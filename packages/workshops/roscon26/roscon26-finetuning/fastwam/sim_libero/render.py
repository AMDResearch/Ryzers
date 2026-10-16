# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Shared rendering helpers for the LIBERO interactive/sanity harness.

Pure-stdlib + Pillow/imageio; no torch, no policy code. Handles MJPEG frame encoding,
the composed agentview|wrist viewport, the command banner, and even-dimension MP4 saving
(FFmpeg's yuv420p encoder rejects odd width/height).
"""
import io
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font(size):
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def encode_jpeg(rgb, quality=88):
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def compose_view(imgs, height=None):
    """Stitch the {image, wrist_image} dict from get_libero_image side-by-side.

    Optionally upscale to `height` px tall for a crisper live viewport (aspect kept).
    """
    parts = []
    for key in ("image", "wrist_image"):
        if key in imgs:
            arr = np.asarray(imgs[key])
            parts.append(arr)
    view = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
    if height is not None and view.shape[0] != height:
        w = int(round(view.shape[1] * height / view.shape[0]))
        view = np.asarray(Image.fromarray(view).resize((w, height), Image.BILINEAR))
    return np.ascontiguousarray(view)


def banner_frame(rgb, text, size, tag=""):
    """Downscale the frame to `size` px wide and add a top banner with the command.

    Output width/height are forced even so the H.264 yuv420p encoder accepts them.
    """
    img = Image.fromarray(np.ascontiguousarray(rgb))
    w = size
    h = int(round(img.height * size / img.width))
    w += w % 2
    h += h % 2
    img = img.resize((w, h), Image.BILINEAR)
    bh = max(40, h // 10)
    bh += bh % 2
    canvas = Image.new("RGB", (w, h + bh), (15, 15, 18))
    canvas.paste(img, (0, bh))
    d = ImageDraw.Draw(canvas)
    f = font(max(14, w // 40))
    cap = 44 if tag else 70
    msg = text if len(text) <= cap else text[: cap - 3] + "..."
    d.text((10, bh // 2), msg, fill=(240, 240, 240), font=f, anchor="lm")
    if tag:
        color = (255, 180, 80) if tag == "THINKING" else (120, 210, 140)
        d.text((w - 10, bh // 2), tag, fill=color, font=f, anchor="rm")
    return np.asarray(canvas)


def save_mp4(frames, path, fps=20):
    """Save RGB frames to an MP4 (H.264, yuv420p). Frames must share even dimensions.

    imageio-ffmpeg already injects `-pix_fmt yuv420p` for libx264, so we set it via the
    writer's `pixelformat` (not output_params) to avoid ffmpeg's "Multiple -pix_fmt
    options" warning from passing it twice.
    """
    import imageio

    with imageio.get_writer(
        path, fps=fps, codec="libx264", quality=8,
        macro_block_size=1, pixelformat="yuv420p",
    ) as w:
        for fr in frames:
            w.append_data(np.ascontiguousarray(fr))
