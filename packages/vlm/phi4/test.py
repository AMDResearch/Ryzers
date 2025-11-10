# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Example adapted from: https://huggingface.co/microsoft/Phi-4-multimodal-instruct

from PIL import Image
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

# Model to use
model_path = "microsoft/Phi-4-multimodal-instruct"

# Load model and processor
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="auto", 
    torch_dtype="auto", 
    trust_remote_code=True,
    _attn_implementation='eager',
    # _attn_implementation='flash_attention_2', # not supported on gfx1151 yet
)

# Load generation config
generation_config = GenerationConfig.from_pretrained(model_path)

# Define prompt structure
user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'

# Part 1: Image Processing
print("\n--- IMAGE PROCESSING ---")
prompt = f'{user_prompt}<|image_1|>What is shown in this image?{prompt_suffix}{assistant_prompt}'
print(f'>>> Prompt\n{prompt}')

# Open image
image = Image.open('/ryzers/data/toucan.jpg')
inputs = processor(text=prompt, images=image, return_tensors='pt').to(model.device)

# Generate response
generate_ids = model.generate(
    **inputs,
    max_new_tokens=1000,
    generation_config=generation_config,
)
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(f'>>> Response\n{response}')

# Part 2: Audio Processing
print("\n--- AUDIO PROCESSING ---")

speech_prompt = "Transcribe the audio to text, and then translate the audio to French. Use <sep> as a separator between the original transcript and the translation."
prompt = f'{user_prompt}<|audio_1|>{speech_prompt}{prompt_suffix}{assistant_prompt}'
print(f'>>> Prompt\n{prompt}')

# Open audio file
audio, samplerate = sf.read("/ryzers/data/audio.flac")

# Process with the model
inputs = processor(text=prompt, audios=[(audio, samplerate)], return_tensors='pt').to(model.device)

generate_ids = model.generate(
    **inputs,
    max_new_tokens=1000,
    generation_config=generation_config,
)
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(f'>>> Response\n{response}')
