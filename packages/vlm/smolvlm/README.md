###  Smolvlm Docker Setup

This folder contains the necessary files to build and run a Docker container with Smolvlm using a llamacpp backend.

Additionally, there is a demo.sh included that when run on the host machine, will serve up a webpage that allows a user to interact with smolvlm.  A webcam is required.  Credit to https://github.com/ngxson/smolvlm-realtime-webcam.

### Demo

```sh
git clone https://github.com/ngxson/smolvlm-realtime-webcam <PATH TO REPO>

ryzers build llamacpp smolvlm
ryzers run /ryzers/demo_smolvlm.sh
```

In firefox, open file://\<PATH TO REPO\>/index.html



### Build the Docker Image

```sh
ryzers build llamacpp smolvlm
ryzers run
```

Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.