### MolmoAct2

This package runs [MolmoAct2](https://huggingface.co/collections/allenai/molmoact)
on AMD ROCm. It is validated on the Strix Halo `gfx1151` iGPU and includes:

- a full MolmoAct2-DROID model smoke test
- DROID open-loop replay against ground-truth actions
- LIBERO closed-loop MuJoCo evaluation with rollout videos
- a browser-based interactive LIBERO simulator
- zero-shot cross-embodiment interactive demos for UR5e and xArm6 native grippers

The Docker image bakes the simulator stack and xArm6 assets. Model weights and
datasets are stored in the persistent Hugging Face cache volume and can be
preloaded before demo runs.

### Build

```sh
ryzers build molmoact2
ryzers run
ryzers run /ryzers/download.sh
```

`ryzers run` with no command is a light ROCm environment check. `download.sh`
preloads the MolmoAct2-DROID, MolmoAct2-DROID-Dataset, and
MolmoAct2-Think-LIBERO assets into the mounted cache.

Artifacts are written to `workspace/molmoact2/outputs`.

### Interactive Demo

![interactive demo](assets/interactive_demo.gif)

Run the synchronous browser demo. The GIF above is the curated
`interactive_demo_webpage.mp4` recording sped up 2x:

```sh
ryzers run /ryzers/demo_interactive.sh
```

Open `http://localhost:8080`. On a remote box, forward the port first:

```sh
ssh -L 8080:localhost:8080 <host>
```

The synchronous demo defaults to the fast path: `THINK=0` and `NUM_STEPS=4`.
Pass `THINK=1` for full depth reasoning.

Example fixed task:

```sh
SUITE=libero_object TASK_ID=3 ryzers run /ryzers/demo_interactive.sh
```

Complex synchronous task execution:

![synchronous complex task](assets/synchronous_demo.gif)

### Cross-Embodiment Interactive Demos

The policy action and state contract remains Panda-style end-effector space. The
embodiment layer swaps the simulator robot, solves each arm home pose to the Panda
LIBERO end-effector start, remaps native gripper state into Panda gripper units,
and keeps the wrist camera aligned to the Panda camera-in-grip-site transform when
needed.

UR5e with native Robotiq85 gripper:

![UR5e cross-embodiment demo](assets/cross_embodiment_ur5e.gif)

```sh
EMBODIMENT=ur5e ryzers run /ryzers/demo_interactive.sh
```

xArm6 with native XArmGripper:

![xArm6 cross-embodiment demo](assets/cross_embodiment_xarm6.gif)

```sh
EMBODIMENT=xarm6 ryzers run /ryzers/demo_interactive.sh
```

Validated xArm6 defaults:

- `EMBODIMENT_GRIPPER=XArmGripper`
- `EMBODIMENT_EXECUTOR=absolute`
- `EMBODIMENT_SERVO_STEPS=2`
- `EMBODIMENT_XARM6_CAMERA_MATCH=1`

Validated UR5e defaults:

- `EMBODIMENT_GRIPPER=Robotiq85Gripper`
- `EMBODIMENT_UR5E_ROBOTIQ_INIT_BLEND=0.5`
- zero end-effector and wrist-camera offsets

The compatibility entrypoint is also available:

```sh
EMBODIMENT=xarm6 ryzers run /ryzers/demo_interactive_embodiment.sh
```

### Real-Time Interactive Option

The real-time server runs the sim at wall-clock speed while policy planning happens
asynchronously. The arm holds pose while the next action chunk is computed.

```sh
ryzers run /ryzers/demo_interactive_rt.sh
EMBODIMENT=ur5e ryzers run /ryzers/demo_interactive_rt.sh
EMBODIMENT=xarm6 ryzers run /ryzers/demo_interactive_rt.sh
```

Open `http://localhost:8081`, or forward it from a remote box:

```sh
ssh -L 8081:localhost:8081 <host>
```

Long-horizon tasks can use:

```sh
RT_HORIZON_MULT=10 ryzers run /ryzers/demo_interactive_rt.sh
```

### Closed-Loop LIBERO Evaluation

![libero closed-loop](assets/libero_closedloop.gif)

Run one closed-loop LIBERO rollout in MuJoCo:

```sh
ryzers run /ryzers/demo_libero.sh
SUITE=libero_object TASK_ID=3 ryzers run /ryzers/demo_libero.sh
```

Suites: `libero_10`, `libero_goal`, `libero_object`, `libero_spatial`.
`TASK_ID` is `0` to `9`.

The same embodiment switches work for closed-loop evaluation:

```sh
EMBODIMENT=ur5e SUITE=libero_object TASK_ID=3 ryzers run /ryzers/demo_libero.sh
EMBODIMENT=xarm6 SUITE=libero_object TASK_ID=3 ryzers run /ryzers/demo_libero.sh
```

### DROID Open-Loop Replay

![droid open-loop](assets/droid_openloop.gif)

Run MolmoAct2-DROID open-loop on a dataset episode:

```sh
ryzers run /ryzers/demo_droid.sh
EPISODE=42 ryzers run /ryzers/demo_droid.sh
```

The demo writes a scene video and a ground-truth versus prediction action plot.

### Model Smoke Test

```sh
ryzers run /ryzers/demo_smoke.sh
```

This loads `allenai/MolmoAct2-DROID` and runs one end-to-end action prediction on
ROCm.

### Common Knobs

- `THINK=1` enables the full depth reasoning path.
- `THINK=0 NUM_STEPS=4` is the fast interactive default.
- `PORT=...` changes the browser port.
- `SEED=...`, `SUITE=...`, and `TASK_ID=...` make runs reproducible.

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
