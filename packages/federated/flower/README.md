### Gemma3 (4B)

This directory contains the docker configuration files to run Gemma3-4B on RyzenAI platforms. Gemma3-4B is the smallest open-weights model released under the Gemma3 suite that offers multimodal support.

### Build and run the Docker Image

To build and run a Docker container with Gemma3-4B-Instruct using a llamacpp backend, run:

```sh
ryzers build llamacpp gemma3
ryzers run
```

### Demo

Additionally, there is a demo.sh included that when run on the host machine, will serve up a webpage that allows a user to interact with Gemma3-4B VLM.  A webcam is required. Credit to https://github.com/ngxson/smolvlm-realtime-webcam.

```sh
git clone https://github.com/ngxson/smolvlm-realtime-webcam <PATH TO REPO>

ryzers build llamacpp gemma3
ryzers run /ryzers/demo_gemma3.sh
```

In a browser, open file://\<PATH TO REPO\>/index.html

Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.