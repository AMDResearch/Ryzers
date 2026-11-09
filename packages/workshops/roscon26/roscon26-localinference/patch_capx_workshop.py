#!/usr/bin/env python3

# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Apply the ROSCon 2026 workshop modifications to a composed CaP-X checkout.

Run as ``python3 -I patch_capx_workshop.py /ryzers/cap-x``.

These four changes used to live inside the monolithic JupyterHub course image.
They are a patch layer here so that the CaP-X commit, its dependency
resolution and its submodule selection stay owned by ``packages/robotics/capx``
and a capx bump flows through without a fork.

Every edit asserts how many times its anchor appears. An upstream refactor
therefore fails ``docker build`` instead of producing an image that only falls
over once the room is full.

Two of the changes have to tolerate the composed layer having gone first:

* ``capx/integrations/__init__.py`` has already had a ``FrankaControlApiSam2``
  alias appended by the capx ryzer. That line registers a *different* name, so
  the anchors below still match exactly once.
* ``capx/envs/runner.py`` has already had its 120 s wait raised to 900 s by the
  capx ryzer. Accept that, and re-apply only if the upstream value is found.
"""

from __future__ import annotations

import sys
from pathlib import Path


def replace_once(text: str, old: str, new: str, where: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"{where}: expected exactly 1 occurrence of {old[:72]!r}, found {count}. "
            "CaP-X changed upstream - re-derive this patch against CAPX_REF."
        )
    return text.replace(old, new)


# ---------------------------------------------------------------------------
# 1. Perception registry: OWLv2 + box-prompted SAM2 instead of gated SAM3.
#
# CaP-X ships both paths but registers SAM3 by default, and facebook/sam3 is
# gated behind a Hugging Face approval. The four live workshop scenarios name
# the plain FrankaControlApi / FrankaControlSpillWipeApi in their env_configs,
# so flipping the *default* is what removes the token requirement - the capx
# ryzer's extra FrankaControlApiSam2 alias is never reached from those configs.
#
# Only these two move. R1ProControlApi and FrankaLiberoApi keep their SAM3
# default: no R1Pro and no LIBERO scenario is part of this workshop (the chain
# builds capx with CAPX_SIM=robosuite), and changing them would be a behaviour
# change this course cannot test.
# ---------------------------------------------------------------------------
WORKSHOP_APIS = (
    (
        'register_api("FrankaControlApi", lambda env: FrankaControlApi(env, use_sam3={}))',
        "FrankaControlApi",
    ),
    (
        "lambda env: FrankaControlSpillWipeApi("
        "env, tcp_offset=[0.0, 0.0, -0.0158], use_sam3={}),",
        "FrankaControlSpillWipeApi",
    ),
)


def patch_registry(root: Path) -> None:
    path = root / "capx/integrations/__init__.py"
    text = path.read_text()
    for template, api in WORKSHOP_APIS:
        if template.format("False") in text:
            print(f"{path.name}: {api} already grounds with OWLv2/SAM2, skipping")
            continue
        text = replace_once(
            text, template.format("True"), template.format("False"), path.name
        )
    path.write_text(text)
    print("registry: FrankaControlApi + FrankaControlSpillWipeApi -> OWLv2/SAM2")


# ---------------------------------------------------------------------------
# 2. FP16 checkpoint staging.
#
# ROCm takes several minutes to transfer these checkpoints to the APU when
# their CPU-side tensors are float32. Stage them as float16 and restore float32
# on-device: the GPU-side conversion is effectively instantaneous and execution
# stays FP32, so this is a load-time optimisation and not a precision change.
# The workshop's held-out rollouts were validated against the staged weights.
#
# Note "cuda" is the right string on ROCm - torch keeps the CUDA device names.
# ---------------------------------------------------------------------------
def patch_owlvit_server(root: Path) -> None:
    path = root / "capx/serving/launch_owlvit_server.py"
    text = path.read_text()
    if "load_dtype" in text:
        print(f"{path.name}: FP16 staging already present, skipping")
        return
    text = replace_once(
        text,
        "    try:\n        if is_v2:\n",
        '    load_dtype = torch.float16 if device.startswith("cuda") else torch.float32\n\n'
        "    try:\n"
        "        if is_v2:\n",
        path.name,
    )
    for model_class in ("Owlv2ForObjectDetection", "OwlViTForObjectDetection"):
        text = replace_once(
            text,
            f"_MODEL = {model_class}.from_pretrained(model_name)",
            f"_MODEL = {model_class}.from_pretrained(model_name, dtype=load_dtype)",
            path.name,
        )
    text = replace_once(
        text,
        "        _MODEL = _MODEL.to(device)\n        _MODEL.eval()\n",
        "        _MODEL = _MODEL.to(device)\n"
        '        if device.startswith("cuda"):\n'
        "            _MODEL = _MODEL.float()\n"
        "        _MODEL.eval()\n",
        path.name,
    )
    path.write_text(text)
    print("launch_owlvit_server.py: FP16 staging, FP32 on device")


def patch_sam2_server(root: Path) -> None:
    path = root / "capx/serving/launch_sam2_server.py"
    text = path.read_text()
    if "load_dtype" in text:
        print(f"{path.name}: FP16 staging already present, skipping")
        return
    text = replace_once(
        text,
        "    # Initialize Pipeline\n",
        '    load_dtype = torch.float16 if device_arg.startswith("cuda") else torch.float32\n\n'
        "    # Initialize Pipeline\n",
        path.name,
    )
    text = replace_once(
        text,
        '    _GENERATOR = pipeline("mask-generation", model=model_name, device=device_arg)\n',
        "    _GENERATOR = pipeline(\n"
        '        "mask-generation", model=model_name, device=device_arg, dtype=load_dtype\n'
        "    )\n",
        path.name,
    )
    text = replace_once(
        text,
        "        _MODEL = Sam2Model.from_pretrained(model_name).to(device_arg)\n",
        "        _MODEL = Sam2Model.from_pretrained(model_name, dtype=load_dtype).to(device_arg)\n",
        path.name,
    )
    text = replace_once(
        text,
        '    elif hasattr(_MODEL, "to"):\n        _MODEL = _MODEL.to(device_arg)\n\n',
        '    elif hasattr(_MODEL, "to"):\n'
        "        _MODEL = _MODEL.to(device_arg)\n"
        '    if device_arg.startswith("cuda"):\n'
        "        _MODEL = _MODEL.float()\n\n",
        path.name,
    )
    path.write_text(text)
    print("launch_sam2_server.py: FP16 staging, FP32 on device")


# ---------------------------------------------------------------------------
# 3. Service wait.
#
# Loading the perception models onto the GPU can take several minutes on a cold
# start. Upstream begins trials after 120 seconds even when those services are
# still unavailable, which fails the first trials of every run. The composed
# capx ryzer already raises this, so the normal path here is a no-op.
# ---------------------------------------------------------------------------
def patch_service_timeout(root: Path) -> None:
    path = root / "capx/envs/runner.py"
    text = path.read_text()
    upstream = "api_servers: list | None, wait_timeout: float = 120.0"
    patched = "api_servers: list | None, wait_timeout: float = 900.0"
    if patched in text:
        print("runner.py: 900 s service wait already applied by the capx ryzer")
        return
    text = replace_once(text, upstream, patched, path.name)
    path.write_text(text)
    print("runner.py: service wait 120 s -> 900 s")


# ---------------------------------------------------------------------------
# 4. Scenario configs.
#
# Replace the single gated SAM3 server with the OWLv2 text-grounding server and
# the box-prompted SAM2 segmentation server, and write a *_fast.yaml sibling
# carrying the smaller pair so the session can A/B large against small without
# a download. Ports match what rho_demo.SERVICE_PORTS probes: 8113 SAM2,
# 8115 Contact-GraspNet, 8116 PyRoKi, 8117 OWLv2.
# ---------------------------------------------------------------------------
SAM3_SERVER = """  - _target_: capx.serving.launch_sam3_server.main
    device: cuda
    port: 8114
    host: 127.0.0.1
"""

LARGE_SERVERS = """  - _target_: capx.serving.launch_owlvit_server.main
    model_name: google/owlv2-large-patch14-ensemble
    device: cuda
    port: 8117
    host: 127.0.0.1

  - _target_: capx.serving.launch_sam2_server.main
    model_name: facebook/sam2.1-hiera-large
    device: cuda
    port: 8113
    host: 127.0.0.1
"""

FAST_SERVERS = """  - _target_: capx.serving.launch_owlvit_server.main
    model_name: google/owlv2-base-patch16-ensemble
    device: cuda
    port: 8117
    host: 127.0.0.1

  - _target_: capx.serving.launch_sam2_server.main
    model_name: facebook/sam2.1-hiera-small
    device: cuda
    port: 8113
    host: 127.0.0.1
"""

CONFIGS = (
    "env_configs/cube_lifting/franka_robosuite_cube_lifting.yaml",
    "env_configs/cube_restack/franka_robosuite_cube_restack.yaml",
    "env_configs/cube_stack/franka_robosuite_cube_stack.yaml",
    "env_configs/spill_wipe/franka_robosuite_spill_wipe.yaml",
)


def patch_configs(root: Path) -> None:
    for name in CONFIGS:
        config = root / name
        text = config.read_text()
        if "launch_sam3_server.main" not in text:
            print(f"{name}: already uses OWLv2/SAM2, skipping")
            continue
        config.write_text(replace_once(text, SAM3_SERVER, LARGE_SERVERS, name))
        fast = config.with_name(f"{config.stem}_fast{config.suffix}")
        fast.write_text(text.replace(SAM3_SERVER, FAST_SERVERS))
        print(f"{name}: SAM3 -> OWLv2+SAM2 (+ {fast.name})")


def verify(root: Path) -> None:
    """Re-read everything, so the build fails here rather than at runtime."""
    for name in ("capx/serving/launch_owlvit_server.py", "capx/serving/launch_sam2_server.py"):
        text = (root / name).read_text()
        assert "dtype=load_dtype" in text, name
        assert "_MODEL = _MODEL.float()" in text, name

    registry = (root / "capx/integrations/__init__.py").read_text()
    for template, api in WORKSHOP_APIS:
        assert template.format("False") in registry, f"{api} still defaults to SAM3"
        assert template.format("True") not in registry, f"{api} SAM3 default survived"

    assert "wait_timeout: float = 900.0" in (root / "capx/envs/runner.py").read_text()

    for name in CONFIGS:
        config = root / name
        text = config.read_text()
        assert "launch_sam3_server.main" not in text, name
        assert "google/owlv2-large-patch14-ensemble" in text, name
        assert "facebook/sam2.1-hiera-large" in text, name
        fast = config.with_name(f"{config.stem}_fast{config.suffix}")
        fast_text = fast.read_text()
        assert "google/owlv2-base-patch16-ensemble" in fast_text, fast.name
        assert "facebook/sam2.1-hiera-small" in fast_text, fast.name
    print("workshop CaP-X patches verified")


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else "/ryzers/cap-x")
    if not root.is_dir():
        # Only reachable in a standalone build, where there is no CaP-X to
        # patch. Warn loudly and succeed: test.sh reports this as a warning
        # too, and failing here would make `ryzers build roscon26-localinference`
        # impossible rather than merely incomplete. Once the tree *is* present,
        # every anchor below is a hard requirement.
        print(
            f"WARNING: {root} is not a directory - no CaP-X to patch.\n"
            "         Notebooks 3 and 4 will not run. Build the chain:\n"
            "         ryzers build jupyterlab ros o3de rai lemonade-sdk capx "
            "roscon26-localinference"
        )
        return 0
    patch_registry(root)
    patch_owlvit_server(root)
    patch_sam2_server(root)
    patch_service_timeout(root)
    patch_configs(root)
    verify(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
