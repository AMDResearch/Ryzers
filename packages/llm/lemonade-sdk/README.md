# Lemonade Server

[Lemonade](https://lemonade-server.ai/) is a lightweight local AI server for running text, vision, image generation, and speech models on AMD hardware. It provides an OpenAI-compatible API at `http://localhost:13305/api/v1`, making it a drop-in backend for hundreds of apps that support the OpenAI protocol.

### Build the Docker Image

```sh
ryzers build lemonade-sdk
ryzers run
```

### Interactive Usage

```sh
ryzers run bash
lemond &
lemonade list
lemonade pull Qwen3-VL-8B-Instruct-GGUF
lemonade run Qwen3-VL-8B-Instruct-GGUF
```

### References

- https://lemonade-server.ai/
- https://github.com/lemonade-sdk/lemonade

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
