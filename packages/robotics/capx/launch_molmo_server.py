# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""OpenAI-compatible Molmo2 pointing server for CaP-X on ROCm.

CaP-X's `capx/integrations/vision/molmo.py` talks to an OpenAI-compatible
`/v1/chat/completions` endpoint on port 8122 and parses `<points …>` tags out of
the reply. Upstream expects that endpoint to be a separate vLLM process, but the
`molmo` extra pins `vllm` and conflicts with the `robosuite` extra, so it cannot
live in the Robosuite eval image.

This server provides the same endpoint using plain `transformers` on the base
ROCm PyTorch build, which is all the pointing path needs: one short greedy
generation per query, no batching or paged attention.

Usage (matches the other capx.serving launchers):

    python3 capx/serving/launch_molmo_server.py --port 8122
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
import re
import time
from typing import Any

import numpy as np
import torch
import tyro
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

_PROC: Any | None = None
_MODEL: Any | None = None
_MODEL_NAME: str = "allenai/Molmo2-8B"
_DEVICE: str = "cuda"
_MAX_NEW_TOKENS: int = 256

_DATA_URL_RE = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,", re.IGNORECASE)


# --- API models (the subset of the OpenAI schema that capx sends) ---


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]]
    max_tokens: int | None = None
    temperature: float | None = None
    stop: list[str] | str | None = None


def _decode_data_url(url: str) -> Image.Image:
    payload = _DATA_URL_RE.sub("", url)
    try:
        raw = base64.b64decode(payload)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Malformed base64 image: {exc}") from exc
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _extract_prompt_and_image(messages: list[dict[str, Any]]) -> tuple[str, Image.Image | None]:
    """Pull the last text prompt and the last image out of an OpenAI message list."""
    text_parts: list[str] = []
    image: Image.Image | None = None

    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            text_parts.append(content)
            continue
        for part in content or []:
            if not isinstance(part, dict):
                continue
            kind = part.get("type")
            if kind == "text" and part.get("text"):
                text_parts.append(str(part["text"]))
            elif kind == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                if url.startswith("data:"):
                    image = _decode_data_url(url)
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="Only base64 data: image URLs are supported",
                    )

    return "\n".join(text_parts).strip(), image


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok" if _MODEL is not None else "loading", "model": _MODEL_NAME}


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": _MODEL_NAME, "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest) -> dict[str, Any]:
    if _MODEL is None or _PROC is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    prompt, image = _extract_prompt_and_image(req.messages)
    if image is None:
        raise HTTPException(status_code=400, detail="No image supplied in request")

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt or "Point at the object."}]
    content.append({"type": "image", "image": image})

    inputs = _PROC.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,
    )
    inputs = {k: (v.to(_MODEL.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    max_new_tokens = req.max_tokens or _MAX_NEW_TOKENS
    start = time.time()
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if str(_MODEL.device).startswith("cuda")
        else torch.autocast("cpu", enabled=False)
    )
    with torch.inference_mode(), autocast:
        output = _MODEL.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    generated = output[0, inputs["input_ids"].size(1):]
    text = _PROC.decode(generated, skip_special_tokens=True)
    logger.info("point query %.2fs prompt=%r -> %r", time.time() - start, prompt, text)

    return {
        "id": "chatcmpl-molmo",
        "object": "chat.completion",
        "model": req.model or _MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": int(inputs["input_ids"].size(1)),
            "completion_tokens": int(generated.numel()),
            "total_tokens": int(output.numel()),
        },
    }


def main(
    model_name: str = "allenai/Molmo2-8B",
    device: str = "cuda",
    port: int = 8122,
    host: str = "127.0.0.1",
    max_new_tokens: int = 256,
) -> None:
    global _PROC, _MODEL, _MODEL_NAME, _DEVICE, _MAX_NEW_TOKENS

    from transformers import AutoModelForImageTextToText, AutoProcessor

    _MODEL_NAME = model_name
    _DEVICE = device
    _MAX_NEW_TOKENS = max_new_tokens

    logger.info("Loading Molmo pointing model %s on %s …", model_name, device)
    _PROC = AutoProcessor.from_pretrained(
        model_name, trust_remote_code=True, padding_side="left"
    )
    # No device_map= here: that path requires `accelerate`, which the Robosuite
    # eval image does not install. Load on CPU then move, like the other servers.
    _MODEL = AutoModelForImageTextToText.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
    )
    _MODEL = _MODEL.to(device)
    _MODEL.eval()
    logger.info("Molmo model loaded. Starting server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)
