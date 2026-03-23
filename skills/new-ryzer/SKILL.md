# Skill: new-ryzer

> Scaffold a new Ryzer Docker package with all required files.

## Usage

```
new-ryzer <category> <package_name>
```

- `category` — one of the existing directories under `packages/` (e.g. `llm`, `vlm`, `vla`, `robotics`, `vision`, `npu`, `ros`, `graphics`, `ide`, `adaptive-socs`), or a new category name if none fits.
- `package_name` — lowercase, no spaces or hyphens (e.g. `ollama`, `genesis`, `mobilesam`).

---

## What is a Ryzer?

A Ryzer is a self-contained, composable Docker package for running AI/ML/robotics workloads on AMD Ryzen AI hardware. Ryzers are layered — each one builds on top of a base image, and multiple ryzers can be chained together in a single build.

Users build and run ryzers with:

```bash
ryzers build <package_name>        # build a single package
ryzers build pkg1 pkg2 pkg3        # chain multiple packages (each layers on the previous)
ryzers run                         # run the most recently built image
ryzers run bash                    # override CMD to get a shell
```

---

## Required Output

Every ryzer package lives at `packages/<category>/<package_name>/` and **must** contain exactly these 4 files:

```
packages/<category>/<package_name>/
├── Dockerfile
├── config.yaml
├── test.sh (or test.py)
└── README.md
```

Optional extras: `demo.sh`, `demo.py`, `LICENSE`, additional test files.

---

## File Specifications

### 1. Dockerfile

**Rules:**
- MUST start with `ARG BASE_IMAGE` and `FROM ${BASE_IMAGE}` — this enables layered composition.
- MUST include the copyright header.
- Set `WORKDIR /ryzers` before copying test/demo scripts.
- Copy test scripts into `/ryzers/test_<package_name>.<ext>` and `chmod +x` them.
- End with `CMD /ryzers/test_<package_name>.<ext>` so the default action validates the install.
- Use `EXPOSE` for any ports the package serves on.
- Keep installs minimal — only what this package needs.

**Template:**

```dockerfile
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    <system-packages> \
 && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir --break-system-packages \
    "<package1>==<version>" \
    "<package2>==<version>"

# Copy test script
WORKDIR /ryzers
COPY test.sh /ryzers/test_<package_name>.sh
RUN chmod +x /ryzers/test_<package_name>.sh

# EXPOSE <port>  # if applicable

CMD /ryzers/test_<package_name>.sh
```

**Notes:**
- Use `--no-install-recommends` and clean up apt lists to keep image size down.
- Use `--no-cache-dir` and `--break-system-packages` with pip3.
- Pin versions where possible for reproducibility.
- If the package needs build args, declare them with `ARG` and document in config.yaml.

### 2. config.yaml

This file configures Docker build and run flags. All fields are optional — include only what the package needs. An empty/commented config.yaml is valid.

**Template:**

```yaml
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

### ryzers build options

# gpu_support: false          # add GPU support, default is true
# x11_display: false          # add X11 support, default is true
# init_image: "image_name"    # The first docker image to use as a base
                              # default is ryzers.RYZERS_DEFAULT_INIT_IMAGE

# build_arguments:            # List of ARGs to pass to the `docker build` call
# - "ARG=VALUE"               # E.g. - "PYTHON_VERSION=3.12"


### ryzers run options

# environment_variables:              # List of environment variables to set in the container
# - "ENV_VARIABLE=VALUE"              # E.g. "PYTHONPATH=/path/to/python"

# port_mappings:
# - "host_portnum:container_portnum"  # List of port mappings to expose from the docker
                                      # E.g. "8888:8888" for JupyterLab

# volume_mappings:
# - "host_path:container_path"        # List of volume mappings to mount from the host
                                      # E.g. "/path/to/host:/path/to/container"

# docker_extra_run_flags: "run flags" # Additional flags to pass to the `docker run` command
```

**Common patterns — uncomment/add as needed:**

| Scenario | What to add |
|----------|-------------|
| Package serves a web UI or API | `port_mappings: ["8080:8080"]` |
| Package downloads large models | `volume_mappings: ["$PWD/workspace/.cache/huggingface:/root/.cache/huggingface"]` |
| Package needs extra shared memory | `docker_extra_run_flags: "--shm-size=2g"` |
| Package needs a build-time variable | `build_arguments: ["MY_VERSION=1.0"]` |
| Package needs specific env vars at runtime | `environment_variables: ["HSA_OVERRIDE_GFX_VERSION=11.0.0"]` |
| Package does NOT need GPU | `gpu_support: false` |
| Package does NOT need X11 | `x11_display: false` |

### 3. test.sh (or test.py)

A quick smoke test that validates the package installed correctly. Should exit 0 on success, non-zero on failure.

**Bash template (`test.sh`):**

```bash
#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

echo "Running tests for <package_name>..."
<command_to_test_installation>
if [ $? -eq 0 ]; then
  echo "Tests passed!"
else
  echo "Tests failed!"
  exit 1
fi
```

**Python template (`test.py`):**

```python
#!/usr/bin/env python3

# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import <package>
print(f"<package> version: {<package>.__version__}")
# Add a minimal functional check here
```

**Guidelines:**
- Keep it fast — just verify the install, don't run full test suites.
- Import the main module, print its version, maybe run one trivial operation.
- For services (like ollama), start the server, wait briefly, then test the endpoint.

### 4. README.md

**Template:**

```markdown
# <Package Name> Docker Setup

Brief description of what this package provides and what it's useful for.

## Build & Run

```sh
ryzers build <package_name>
ryzers run
```

## References

- <link to upstream project>
- <link to docs>

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
```

---

## After Creating the Package

1. **Update the main `README.md`** — add a link to the new package in the "Supported Packages" table under the appropriate category row.

2. **Test locally** (if possible):
   ```bash
   ryzers build <package_name>
   ryzers run
   ```

---

## Existing Categories

| Category | Directory | What goes here |
|----------|-----------|----------------|
| LLM | `packages/llm/` | Large language models (ollama, llamacpp, etc.) |
| VLM | `packages/vlm/` | Vision-language models (gemma3, phi4, etc.) |
| VLA | `packages/vla/` | Vision-language-action models (openvla, gr00t, etc.) |
| Robotics | `packages/robotics/` | Robotics frameworks (lerobot, act, genesis, etc.) |
| ROS | `packages/ros/` | ROS 2 and related tools (ros, gazebo) |
| Vision | `packages/vision/` | Computer vision (opencv, sam, ultralytics, etc.) |
| NPU | `packages/npu/` | Ryzen AI NPU tools (xdna, iron, etc.) |
| Graphics | `packages/graphics/` | 3D/graphics engines (o3de) |
| IDE | `packages/ide/` | Development environments (jupyterlab) |
| Adaptive SoCs | `packages/adaptive-socs/` | Adaptive SoC tools (pynq-remote) |
| Workshops | `packages/workshops/` | Workshop demos |

Create a new category directory if the package doesn't fit any of these.

---

## Reference Examples

Study these existing packages for patterns:

- **Minimal package:** `packages/init/ryzer_blank/` — bare minimum Dockerfile
- **Typical package:** `packages/robotics/genesis/` — good all-around reference
- **Service with ports:** `packages/llm/ollama/` — port mappings, volume mounts, server test
- **Package with build args:** `packages/vision/opencv/` — uses `ARG OPENCV_VERSION` from config.yaml
- **Web UI package:** `packages/ide/jupyterlab/` — port mapping, environment variables
