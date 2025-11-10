# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# script modified from https://learnopencv.com/smolvla-lerobot-vision-language-action-model/
# ran from commit: https://github.com/huggingface/lerobot/commits/5c87365cc160617c45dc5d1bbb3788de010271a7 (jun 13)

import torch
import time
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from transformers import AutoProcessor

# Load model (replace with your checkpoint if needed)
# Note: locking to hub revision before pipeline update
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base", revision="3326b10").to("cuda")
policy.eval()

# patch: The loaded policy is missing the language_tokenizer attribute.
policy.language_tokenizer = AutoProcessor.from_pretrained(policy.config.vlm_model_name).tokenizer

# Dummy batch config for a single observation
batch_size = 1
img_shape = (3, 512, 512)  # (C, H, W)
# Infer state_dim from the loaded normalization stats
state_dim = policy.normalize_inputs.buffer_observation_state.mean.shape[-1]

dummy_batch = {
    # a single image observation
    "observation.image": torch.rand(batch_size, *img_shape, device="cuda"),
    # a single state observation
    "observation.state": torch.rand(batch_size, state_dim, device="cuda"),
    "task": ["stack the blocks"] * batch_size,
}

# --- Prepare inputs for the model ---
# The policy expects normalized inputs and specific data preparation.
normalized_batch = policy.normalize_inputs(dummy_batch)
images, img_masks = policy.prepare_images(normalized_batch)
state = policy.prepare_state(normalized_batch)
lang_tokens, lang_masks = policy.prepare_language(normalized_batch)
# ---

# Warmup
for _ in range(3):
    with torch.no_grad():
        output = policy.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state)

# Benchmark
torch.cuda.reset_peak_memory_stats()
start = time.time()
for _ in range(100):
    with torch.no_grad():
        _ = policy.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state)
end = time.time()

avg_inf = (end - start)/100
avg_hz = policy.config.chunk_size/avg_inf

print(f"Chunk size: {policy.config.chunk_size}")
print(f"Avg inference time: {avg_inf:.6f} s")
print(f"Avg inference hz: {avg_hz:.6f} hz")
print(f"Max GPU memory used: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
