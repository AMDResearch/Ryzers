# Lemonade Server

[Lemonade](https://lemonade-server.ai/) is a lightweight local AI server for running text, vision, image generation, and speech models on AMD hardware. It provides an OpenAI-compatible API at `http://localhost:8000/api/v1`, making it a drop-in backend for hundreds of apps that support the OpenAI protocol.

### Build the Docker Image

```sh
ryzers build lemonade-sdk
ryzers run
```

### Interactive Usage

```sh
ryzers run bash
lemonade-server list
lemonade-server pull Qwen3-VL-8B-Instruct-GGUF
lemonade-server serve --no-tray
```

### References

- https://lemonade-server.ai/
- https://github.com/lemonade-sdk/lemonade

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
