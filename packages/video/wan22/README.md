# WAN 2.2 TI2V-5B Docker Setup

This Ryzer package runs the WAN 2.2 TI2V-5B text-to-video pipeline on AMD ROCm
systems. It uses the upstream WAN 2.2 source with a ROCm `flash_attn` shim and a
crop-blended tiled VAE decode fallback for Strix Halo.

Weights are not baked into the image. They are downloaded once into the
persistent `/models` volume and reused for prompt-driven generation.

## Build And Smoke Test

Build the image:

```sh
ryzers build wan22
```

Run the default smoke test:

```sh
ryzers run
```

The smoke test verifies ROCm PyTorch, SDPA attention, the `flash_attn` shim, WAN
TI2V config import, CLI entry points, and basic `.mp4` writing. It does not
download weights or run generation.

## Download Weights

Download `Wan-AI/Wan2.2-TI2V-5B` once:

```sh
ryzers run /ryzers/download_wan22.sh
```

If Hugging Face authentication is required, set `HF_TOKEN` before running the
download command.

## Generate Video

Generate a 121-frame, 1280x704 video:

```sh
PROMPT="A dreamy nighttime sky filled with glowing stars over a quiet village. One giant star suddenly flickers, sneezes loudly, and falls from the sky wearing fuzzy pajamas and bunny slippers. Villagers below stare in confusion while a dog howls at it. Soft magical lighting, fantasy animation style, cozy but absurd." \
SIZE=1280x704 \
FRAME_NUM=121 \
ryzers run /ryzers/demo_wan22.sh
```

The generated `.mp4` is written under `/outputs`, which maps to
`$PWD/workspace/wan22/outputs` on the host.

Use `SAVE_FILE` to choose the output path:

```sh
PROMPT="A realistic brown fox jumps over a sleeping dog in a grassy field." \
SIZE=1280x704 \
FRAME_NUM=121 \
SAVE_FILE=/outputs/foxdog.mp4 \
ryzers run /ryzers/demo_wan22.sh
```

If setting variables on separate lines, export them first:

```sh
export PROMPT="A cinematic landscape shot of waves crashing at sunrise."
export SIZE=1280x704
export FRAME_NUM=121
ryzers run /ryzers/demo_wan22.sh
```

Supported generation variables:

- `PROMPT` or `PROMPT_FILE`
- `SIZE`: `1280x704` or `704x1280`
- `FRAME_NUM`: explicit WAN frame count, must be `4n+1`
- `DURATION_SECONDS`: duration rounded up to a valid `4n+1` frame count
- `SAMPLE_STEPS`: default `50`
- `SEED`: default `20260525`
- `SAVE_FILE`: default auto-generated under `/outputs`

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

