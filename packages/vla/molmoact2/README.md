### MolmoAct2

This directory contains the Docker configuration files to run
[MolmoAct2 from the Allen Institute for AI](https://huggingface.co/collections/allenai/molmoact)
on AMD ROCm (validated on the Strix Halo `gfx1151` iGPU). It ships three inference
demos: a full-model smoke test, a DROID open-loop replay, and a LIBERO closed-loop
rollout in the MuJoCo simulator. The Ryzers example is adapted from the examples
provided by the MolmoAct team.

Model weights and datasets are not baked into the image; they download once into
the persistent Hugging Face cache volume and are reused across runs.

### Build and run the Docker Image

```sh
ryzers build molmoact2
ryzers run                  # environment sign-of-life (ROCm torch + deps, no weights)
```

### Run the demos

Artifacts (videos and action plots) are written to `workspace/molmoact2/outputs`.
Set `HF_TOKEN` for gated repositories. Optionally pre-fetch all assets first:

```sh
ryzers run /ryzers/download.sh
```

Full-model smoke test — loads the `allenai/MolmoAct2-DROID` checkpoint and runs one
action prediction end to end:

```sh
ryzers run /ryzers/demo_smoke.sh
```

DROID open-loop replay — runs MolmoAct2-DROID on a random dataset episode and writes
the scene video plus a ground-truth-vs-prediction action plot:

```sh
ryzers run /ryzers/demo_droid.sh              # random episode
EPISODE=42 ryzers run /ryzers/demo_droid.sh   # fixed episode
```

LIBERO closed-loop rollout — runs the MolmoAct2-LIBERO policy in the LIBERO/robosuite
MuJoCo simulator (EGL headless) on a random scene and writes the rollout video plus
the executed action plot:

```sh
ryzers run /ryzers/demo_libero.sh                              # random scene
SUITE=libero_object TASK_ID=3 ryzers run /ryzers/demo_libero.sh
```

Suites: `libero_10`, `libero_goal`, `libero_object`, `libero_spatial` (`TASK_ID` 0-9).

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
