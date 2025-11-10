# DINOv3

Meta's DINOv3 vision foundation model for feature extraction.

## Requirements

**HuggingFace Token Required** - DINOv3 models are gated and require authentication.

1. Accept license at https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m (click on the `Files and versions` tab)
2. Get token from https://huggingface.co/settings/tokens
3. Update `packages/vision/dinov3/config.yaml`:

```yaml
environment_variables:
- "HF_TOKEN=hf_your_actual_token_here"
```

## Build and Run

```bash
ryzers build dinov3
ryzers run
```

## Model

By default test.py uses `facebook/dinov3-vits16-pretrain-lvd1689m` (21.6M parameters - smallest variant).

Find list of models on [Hugging Face](https://huggingface.co/collections/facebook/dinov3), e.g:
- `facebook/dinov3-vitb16-pretrain-lvd1689m` (85.7M)
- `facebook/dinov3-vitl16-pretrain-lvd1689m` (0.3B)
- `facebook/dinov3-vit7b16-pretrain-lvd1689m` (7B)

## References

- [HuggingFace Docs](https://huggingface.co/docs/transformers/main/en/model_doc/dinov3)
- [Meta AI DINOv3](https://ai.meta.com/dinov3/)
- [DINOv3 GitHub](https://github.com/facebookresearch/dinov3)

---

Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
