#!/usr/bin/env bash

# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -euo pipefail

CKPT_DIR="${CKPT_DIR:-/models/Wan2.2-TI2V-5B}"
OUT_DIR="${OUT_DIR:-/outputs}"
AUTO_DOWNLOAD="${AUTO_DOWNLOAD:-1}"
SIZE="${SIZE:-1280x704}"
FRAME_NUM="${FRAME_NUM:-121}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
SEED="${SEED:-20260525}"
PROMPT_FILE="${PROMPT_FILE:-}"
SAVE_FILE="${SAVE_FILE:-}"
USER_ARGS=("$@")

DURATION_SECONDS="${DURATION_SECONDS:-${WAN_SECONDS:-}}"
if [ -z "$DURATION_SECONDS" ]; then
    DURATION_SECONDS="$(printenv SECONDS || true)"
fi

if [ -z "${PROMPT:-}" ] && [ -n "$PROMPT_FILE" ] && [ -f "$PROMPT_FILE" ]; then
    PROMPT="$(tr '\n' ' ' < "$PROMPT_FILE")"
fi

has_arg() {
    local name="$1"
    local arg
    for arg in "${USER_ARGS[@]}"; do
        if [ "$arg" = "$name" ] || [[ "$arg" == "$name="* ]]; then
            return 0
        fi
    done
    return 1
}

for idx in "${!USER_ARGS[@]}"; do
    case "${USER_ARGS[$idx]}" in
        --ckpt_dir)
            next_idx=$((idx + 1))
            if [ "$next_idx" -lt "${#USER_ARGS[@]}" ]; then
                CKPT_DIR="${USER_ARGS[$next_idx]}"
            fi
            ;;
        --ckpt_dir=*)
            CKPT_DIR="${USER_ARGS[$idx]#--ckpt_dir=}"
            ;;
    esac
done

if [ "${#USER_ARGS[@]}" -eq 0 ] && [ -z "${PROMPT:-}" ]; then
    echo "PROMPT is required for generation." >&2
    echo "Use one of:" >&2
    echo "  PROMPT=\"...\" ryzers run /ryzers/demo_wan22.sh" >&2
    echo "  export PROMPT=\"...\"; ryzers run /ryzers/demo_wan22.sh" >&2
    echo "  PROMPT_FILE=/path/to/prompt.txt ryzers run /ryzers/demo_wan22.sh" >&2
    exit 2
fi

if ! python3 /ryzers/download_wan22.py --ckpt_dir "$CKPT_DIR" --check_only >/dev/null 2>&1; then
    if [ "$AUTO_DOWNLOAD" = "1" ]; then
        echo "Checkpoint not found or incomplete at $CKPT_DIR; downloading first." >&2
        /ryzers/download_wan22.sh
    else
        echo "Checkpoint not found or incomplete: $CKPT_DIR" >&2
        echo "Run /ryzers/download_wan22.sh first, or set AUTO_DOWNLOAD=1." >&2
        exit 2
    fi
fi

mkdir -p "$OUT_DIR"

if [ "${#USER_ARGS[@]}" -gt 0 ]; then
    ARGS=("${USER_ARGS[@]}")
    if ! has_arg "--ckpt_dir"; then
        ARGS+=(--ckpt_dir "$CKPT_DIR")
    fi
    if ! has_arg "--output_dir"; then
        ARGS+=(--output_dir "$OUT_DIR")
    fi
else
    ARGS=(
        --ckpt_dir "$CKPT_DIR"
        --size "$SIZE"
        --sample_steps "$SAMPLE_STEPS"
        --seed "$SEED"
        --prompt "$PROMPT"
        --output_dir "$OUT_DIR"
    )

    if [ -n "$SAVE_FILE" ]; then
        ARGS+=(--save_file "$SAVE_FILE")
    fi

    if [ -n "$DURATION_SECONDS" ]; then
        ARGS+=(--seconds "$DURATION_SECONDS")
    else
        ARGS+=(--frame_num "$FRAME_NUM")
    fi
fi

python3 /ryzers/generate_ti2v_strix.py "${ARGS[@]}"

if ! has_arg "-h" && ! has_arg "--help"; then
    echo "WAN 2.2 TI2V generation complete."
fi

: <<'UPSTREAM_EQUIVALENT'
Equivalent upstream-style intent:

python generate.py \
  --task ti2v-5B \
  --size "$SIZE" \
  --ckpt_dir "$CKPT_DIR" \
  --offload_model True \
  --convert_model_dtype \
  --t5_cpu \
  --prompt "$PROMPT" \
  --save_file "$SAVE_FILE"

The Ryzer entry point keeps those model settings and adds the Strix Halo
crop-blended tiled VAE fallback.
UPSTREAM_EQUIVALENT

