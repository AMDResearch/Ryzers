### SAM3

This directory contains the docker configuration files to run Meta's SAM3
image/video segmentation model on RyzenAI platforms.

## Requirements

**HuggingFace Token Required** - SAM3 models are gated and require authentication.

1. Accept license at https://huggingface.co/facebook/sam3
2. Get token from https://huggingface.co/settings/tokens
3. Update `packages/vision/sam3/config.yaml`:

### Build and run the Docker Image

Run:

```sh
ryzers build sam3
ryzers run
```

The model runs an image segmentation query on an example image, and displays
the input image and output overlay-masked image side-by-side on a simple
webpage. You can see the webpage in your browser by navigating to
https://<your_machine_ip_address>:8080 (address will be printed in the
terminal). Once finished, hit Ctrl+C to exit and let docker clean up the
container.

Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
