#!/usr/bin/env python3

# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""WAN 2.2 TI2V-5B generation entry point for Strix Halo ROCm systems."""

from __future__ import annotations

import argparse
import logging
import math
import os
import pathlib
import random
import re
import sys
import types
from datetime import datetime

import torch
from tqdm.auto import tqdm


UPSAMPLE = 16
SUPPORTED_SIZES = {
    "1280x704": (1280, 704),
    "1280*704": (1280, 704),
    "704x1280": (704, 1280),
    "704*1280": (704, 1280),
}


def _starts_for(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, max(length - tile + 1, 1), stride))
    final = length - tile
    if starts[-1] != final:
        starts.append(final)
    return starts


def _crop_bounds(start: int, end: int, full: int, margin: int) -> tuple[int, int]:
    crop_start = margin if start > 0 else 0
    crop_end = (end - start) - (margin if end < full else 0)
    if crop_end <= crop_start:
        return 0, end - start
    return crop_start, crop_end


def _frame_count_for_seconds(seconds: float, fps: int) -> int:
    target = max(1, int(math.ceil(seconds * fps)))
    if target <= 1:
        return 1
    return 4 * int(math.ceil((target - 1) / 4)) + 1


def _parse_size(value: str) -> tuple[int, int]:
    normalized = value.lower().replace(" ", "")
    if normalized not in SUPPORTED_SIZES:
        choices = ", ".join(sorted(set(SUPPORTED_SIZES)))
        raise argparse.ArgumentTypeError(f"unsupported TI2V size {value!r}; choose one of: {choices}")
    return SUPPORTED_SIZES[normalized]


def _slugify_prompt(prompt: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", prompt.strip().lower()).strip("_")
    return slug[:48] or "prompt"


def _patch_tiled_vae_decode(
    vae,
    tile_h: int,
    tile_w: int,
    stride_h: int,
    stride_w: int,
    crop_margin: int,
) -> None:
    original_decode = vae.decode

    def tiled_decode(self_vae, zs):
        if not isinstance(zs, list):
            raise TypeError("zs should be a list")

        decoded_all = []
        for z in zs:
            _, _latent_t, height, width = z.shape
            h_starts = _starts_for(height, tile_h, stride_h)
            w_starts = _starts_for(width, tile_w, stride_w)
            logging.info(
                "Tiled VAE decode: latent=%s tile=%sx%s stride=%sx%s crop=%s grid=%sx%s",
                tuple(z.shape),
                tile_h,
                tile_w,
                stride_h,
                stride_w,
                crop_margin,
                len(h_starts),
                len(w_starts),
            )

            values = None
            weights = None
            total_tiles = len(h_starts) * len(w_starts)
            tile_iter = ((h0, w0) for h0 in h_starts for w0 in w_starts)
            for h0, w0 in tqdm(tile_iter, total=total_tiles, desc="VAE tiles", unit="tile"):
                h1 = min(h0 + tile_h, height)
                w1 = min(w0 + tile_w, width)
                tile = z[:, :, h0:h1, w0:w1].contiguous()
                tile_out = original_decode([tile])[0]
                if values is None:
                    values = torch.zeros(
                        (tile_out.shape[0], tile_out.shape[1], height * UPSAMPLE, width * UPSAMPLE),
                        device=tile_out.device,
                        dtype=torch.float32,
                    )
                    weights = torch.zeros_like(values)

                ch0, ch1 = _crop_bounds(h0, h1, height, crop_margin)
                cw0, cw1 = _crop_bounds(w0, w1, width, crop_margin)
                oh0, oh1 = (h0 + ch0) * UPSAMPLE, (h0 + ch1) * UPSAMPLE
                ow0, ow1 = (w0 + cw0) * UPSAMPLE, (w0 + cw1) * UPSAMPLE
                th0, th1 = ch0 * UPSAMPLE, ch1 * UPSAMPLE
                tw0, tw1 = cw0 * UPSAMPLE, cw1 * UPSAMPLE
                values[:, :, oh0:oh1, ow0:ow1] += tile_out[:, :, th0:th1, tw0:tw1].float()
                weights[:, :, oh0:oh1, ow0:ow1] += 1

            if torch.any(weights == 0):
                raise RuntimeError("tiled VAE decode left uncovered pixels; reduce crop margin or stride")
            decoded_all.append((values / weights).clamp_(-1, 1))
        return decoded_all

    vae.decode = types.MethodType(tiled_decode, vae)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate WAN 2.2 TI2V-5B video on AMD Strix Halo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--size", type=_parse_size, default=(1280, 704), help="Upstream TI2V size: 1280x704 or 704x1280.")
    parser.add_argument("--seconds", type=float, default=None, help="Duration rounded up to a valid 4n+1 frame count.")
    parser.add_argument("--frame_num", type=int, default=None, help="Explicit frame count. Must be 4n+1.")
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--ckpt_dir", default="/models/Wan2.2-TI2V-5B")
    parser.add_argument("--output_dir", default="/outputs")
    parser.add_argument("--save_file", default="")
    parser.add_argument("--no_tiled_vae_decode", action="store_true", help="Disable the Strix Halo tiled VAE fallback.")
    parser.add_argument("--tile_h", type=int, default=12)
    parser.add_argument("--tile_w", type=int, default=12)
    parser.add_argument("--stride_h", type=int, default=6)
    parser.add_argument("--stride_w", type=int, default=6)
    parser.add_argument("--crop_margin", type=int, default=2)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    import wan
    from wan.configs import WAN_CONFIGS
    from wan.utils.utils import save_video

    cfg = WAN_CONFIGS["ti2v-5B"]
    width, height = args.size
    if args.frame_num is not None and args.seconds is not None:
        raise SystemExit("choose either --frame_num or --seconds, not both")
    if args.seconds is not None:
        frame_num = _frame_count_for_seconds(args.seconds, cfg.sample_fps)
    elif args.frame_num is not None:
        frame_num = args.frame_num
    else:
        frame_num = cfg.frame_num
    if frame_num % 4 != 1:
        raise SystemExit(f"frame_num must be 4n+1 for WAN VAE temporal stride; got {frame_num}")

    seed = args.seed
    if seed < 0:
        seed = random.randint(0, sys.maxsize)
    torch.cuda.set_device(0)

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_file:
        save_file = pathlib.Path(args.save_file)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_slug = _slugify_prompt(args.prompt)
        save_file = output_dir / f"ti2v5b_{width}x{height}_f{frame_num}_s{args.sample_steps}_{seed}_{prompt_slug}_{stamp}.mp4"
    save_file.parent.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Generating TI2V-5B: size=%sx%s frames=%s fps=%s steps=%s seed=%s output=%s",
        width,
        height,
        frame_num,
        cfg.sample_fps,
        args.sample_steps,
        seed,
        save_file,
    )
    pipe = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=True,
        convert_model_dtype=True,
    )
    if not args.no_tiled_vae_decode:
        _patch_tiled_vae_decode(pipe.vae, args.tile_h, args.tile_w, args.stride_h, args.stride_w, args.crop_margin)

    video = pipe.generate(
        args.prompt,
        img=None,
        size=(width, height),
        max_area=width * height,
        frame_num=frame_num,
        shift=cfg.sample_shift,
        sample_solver="unipc",
        sampling_steps=args.sample_steps,
        guide_scale=cfg.sample_guide_scale,
        seed=seed,
        offload_model=True,
    )

    logging.info("Saving generated video to %s", save_file)
    save_video(
        tensor=video[None],
        save_file=str(save_file),
        fps=cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    del video, pipe
    torch.cuda.synchronize()
    logging.info("Generation complete: %s", save_file)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
