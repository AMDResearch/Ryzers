# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""FastWAM LIBERO policy adapter for the simulation/libero harness.

Implements the model-agnostic `sim_libero.Policy` seam by wrapping the FastWAM world-action
model. Reuses the *validated* eval machinery from experiments/libero/eval_libero_single.py
verbatim (config compose, model instantiate + checkpoint load, processor/normalizer, and
`_predict_action_chunk`) so interactive/closed-loop rollouts match the shipped numbers.

Selected at runtime by the sim harness via
  POLICY_FACTORY=fastwam_libero_policy:build_policy

Env: CKPT, DATASET_STATS, MIXED_PRECISION (bf16), SUITE, NUM_INFERENCE_STEPS,
REPLAN_STEPS, NUM_STEPS_WAIT, FASTWAM_REPO (/repos/fastwam).
Requires /repos/fastwam and its experiments/libero dir on PYTHONPATH (the demo sets this).
"""
import os

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

import experiments.libero.eval_libero_single as E
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from sim_libero.policy import Policy

FASTWAM_REPO = os.environ.get("FASTWAM_REPO", "/repos/fastwam")
CONFIG_DIR = os.path.join(FASTWAM_REPO, "configs")
DEFAULT_CKPT = "/models/fastwam_release/libero_uncond_2cam224.pt"
DEFAULT_STATS = "/models/fastwam_release/libero_uncond_2cam224_dataset_stats.json"


class FastwamLiberoPolicy(Policy):
    name = "fastwam"

    def __init__(self, model, processor, cfg, action_horizon, input_w, input_h, device):
        self.model = model
        self.processor = processor
        self.cfg = cfg
        self.action_horizon = action_horizon
        self.input_w = input_w
        self.input_h = input_h
        self.device = device
        self.replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
        self.num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 5))

    def predict_action_chunk(self, obs, instruction):
        action, _imgs, _pred = E._predict_action_chunk(
            obs=obs,
            task_description=instruction,
            model=self.model,
            processor=self.processor,
            cfg=self.cfg,
            action_horizon=self.action_horizon,
            input_w=self.input_w,
            input_h=self.input_h,
            model_device=self.device,
        )
        return np.asarray(action, dtype=np.float32)


def build_policy():
    # ryzers passes optional knobs as empty strings; treat "" as unset.
    ckpt = os.environ.get("CKPT") or DEFAULT_CKPT
    stats = os.environ.get("DATASET_STATS") or DEFAULT_STATS
    mixed = os.environ.get("MIXED_PRECISION") or "bf16"
    suite = os.environ.get("SUITE") or "libero_object"

    overrides = [
        f"ckpt={ckpt}",
        "gpu_id=0",
        f"mixed_precision={mixed}",
        f"EVALUATION.task_suite_name={suite}",
        "EVALUATION.task_id=0",
        "EVALUATION.num_trials=1",
        f"EVALUATION.dataset_stats_path={stats}",
        "EVALUATION.output_dir=/tmp/fastwam_interactive",
    ]
    if os.environ.get("NUM_INFERENCE_STEPS"):
        overrides.append(f"EVALUATION.num_inference_steps={os.environ['NUM_INFERENCE_STEPS']}")
    if os.environ.get("REPLAN_STEPS"):
        overrides.append(f"EVALUATION.replan_steps={os.environ['REPLAN_STEPS']}")
    if os.environ.get("NUM_STEPS_WAIT"):
        overrides.append(f"EVALUATION.num_steps_wait={os.environ['NUM_STEPS_WAIT']}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(config_name="sim_libero", overrides=overrides)

    device = E._resolve_eval_device(cfg)
    dtype = E._mixed_precision_to_model_dtype(mixed)
    model = instantiate(cfg.model, model_dtype=dtype, device=device)
    E._load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(device).eval()

    stats_path = E._resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(stats_path))
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    else:
        action_horizon = int(action_horizon_cfg)

    video_size = cfg.data.train.get("video_size", [224, 224])
    input_h = int(video_size[0])
    input_w = int(video_size[1])

    print(f"[fastwam_libero_policy] model ready (ckpt={ckpt}, horizon={action_horizon}, "
          f"input={input_w}x{input_h}, replan={cfg.EVALUATION.get('replan_steps', 5)})", flush=True)
    return FastwamLiberoPolicy(model, processor, cfg, action_horizon, input_w, input_h, device)
