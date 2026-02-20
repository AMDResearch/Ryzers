# CogACT

This package contains the steps to run Microsoft's CogACT vision-language-action model.

## Requirements

**HuggingFace Token Required** - CogACT depends on Llama2 models which are gated and require authentication.

1. Accept license at https://huggingface.co/meta-llama/Llama-2-7b-hf
2. Get token from https://huggingface.co/settings/tokens
3. Update `config.yaml`:

```yaml
environment_variables:
- "HF_TOKEN=hf_your_actual_token_here"
```

## Build and Run

```bash
ryzers build cogact
ryzers run
```

## References

- [CogACT GitHub repo](https://github.com/microsoft/CogACT)
- [HuggingFace Docs](https://huggingface.co/CogACT/CogACT-Base)

---

Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
