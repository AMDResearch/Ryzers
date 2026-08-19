# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Rendering helpers (Pillow/imageio): JPEG encode, viewport upscale, command banner, even-dim MP4 save."""
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


def view_frame(rgb, height=None):
    """Optionally upscale a composed RGB frame to `height` px tall (aspect kept)."""
    view = np.ascontiguousarray(np.asarray(rgb)[:, :, :3])
    if height is not None and view.shape[0] != height:
        w = int(round(view.shape[1] * height / view.shape[0]))
        view = np.asarray(Image.fromarray(view).resize((w, height), Image.BILINEAR))
    return np.ascontiguousarray(view)


def banner_frame(rgb, text, size, tag=""):
    """Downscale to `size` px wide + top command banner; dims forced even for H.264 yuv420p."""
    img = Image.fromarray(np.ascontiguousarray(np.asarray(rgb)[:, :, :3]))
    w = size
    h = int(round(img.height * size / img.width))
    w += w % 2
    h += h % 2
    img = img.resize((w, h), Image.BILINEAR)
    bh = max(40, h // 12)
    bh += bh % 2
    canvas = Image.new("RGB", (w, h + bh), (15, 15, 18))
    canvas.paste(img, (0, bh))
    d = ImageDraw.Draw(canvas)
    f = font(max(14, w // 44))
    cap = 44 if tag else 74
    msg = text if len(text) <= cap else text[: cap - 3] + "..."
    d.text((10, bh // 2), msg, fill=(240, 240, 240), font=f, anchor="lm")
    if tag:
        color = (255, 180, 80) if tag == "THINKING" else (120, 210, 140)
        d.text((w - 10, bh // 2), tag, fill=color, font=f, anchor="rm")
    return np.asarray(canvas)


def save_mp4(frames, path, fps=15):
    """Save RGB frames to an MP4 (H.264, yuv420p); frames must share even dimensions."""
    import imageio

    with imageio.get_writer(
        path, fps=fps, codec="libx264", quality=8,
        macro_block_size=1, pixelformat="yuv420p",
    ) as w:
        for fr in frames:
            w.append_data(np.ascontiguousarray(fr))
