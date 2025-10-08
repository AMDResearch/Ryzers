
![](docs/header.png)

# Ryzen AI Robotics, Vision and ML Dockerfiles

This repository provides a collection of composable Dockerfiles and build scripts for deploying and running software, full applications, and select demonstrators on AMD Ryzen AI hardware. The project is designed to streamline the setup of AI workloads, robotics, vision, and other applications optimized for Ryzen AI.

---

## Features

- **Verified AMD Support for Popular Frameworks**: A variety of Robotics and ML software frameworks supported across Ryzen AI platforms.
- **Optimized for Ryzen AI**: Includes support for hardware-accelerated AI workloads and accelerators like the iGPU and NPUs.
- **Minimal Host Software Requirements**: Standard Ubuntu support with minimal software requirements
- **Composable Dockerfiles**: Modular design for reusability across different applications.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Supported Packages](#supported-packages)
5. [Highlighted Packages](#highlighted-packages)
5. [Contributing](#contributing)
6. [License](#license)

---

## Overview

Ryzers is a modular framework for building and running Docker containers tailored for AMD Ryzen AI hardware. It supports a wide range of applications, including machine learning, robotics, vision, and more. The repository is structured to allow easy composition of Dockerfiles, enabling users to build custom containers for their specific needs.

These dockerfiles will also be pushed and actively maintained in their original repository homes whenever possible.  Ryzers will be a collection point of software frameworks to run on AMD hardware.  We are committed to open-source and happy to accept contributions or feedback on packages hosted here.

---

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/AMDResearch/Ryzers
pip install Ryzers/
```

For detailed installation instructions and requirements, refer to the included documentation.

## Usage


### Simple Example
```
ryzers build genesis
ryzers run 
```

```
# Alternatively, override the Dockerfile's CMD to run a custom command
ryzers run bash
```

For detailed build and run instructions, refer to the included documentation.

## Supported Packages

| Category        | Software                                                                                                    |
|-----------------|--------------------------------------------------------------------------------------------------------------------|
| LLM                     | [`ollama`](packages/llm/ollama), [`llamacpp`](packages/llm/llamacpp), [`lmstudio`](packages/llm/lmstudio)   |
| VLM / VLA              | [`OpenVLA`](packages/robotics/openvla), [`SmolVLA`](packages/robotics/smolvla), [`GR00T-N1.5`](packages/robotics/gr00t), [`Gemma3`](packages/vlm/gemma3), [`SmolVLM`](packages/vlm/smolvlm),  |
| Graphics                     | [`O3DE`](packages/graphics/o3de) |
| Robotics                | [`ROS 2`](packages/ros/ros), [`Gazebo`](packages/ros/gazebo), [`LeRobot`](packages/robotics/lerobot). [`ACT`](packages/robotics/act)    |
| Simulation                |  [`Genesis`](packages/robotics/genesis)  |
| Vision                  | [`OpenCV`](packages/vision/opencv), [`SAM`](packages/vision/sam), [`MobileSAM`](packages/vision/mobilesam), [`ncnn`](packages/vision/ncnn) |
| Ryzen AI NPU                |  [`XDNA`](packages/npu/xdna), [`IRON`](packages/npu/iron), [`NPUEval`](packages/npu/npueval)  |
| Adaptive SoCs           | $${\color{red}\small{\texttt{PYNQ.remote}}}$$ |
| Utilities   | [`JupyterLab`](packages/ide/jupyterlab), [`amdgpu_top`](packages/init/amdgpu_top) |

Packages Legend: 
$${\color{red}\small{\texttt{Coming Soon}}}$$, 
$${\color{orange}\small{\texttt{Help Wanted}}}$$


---

## Highlighted Packages

TBD

## Contributing

We welcome contributions to Ryzers! If you have ideas for new features, bug fixes, or improvements, please submit a pull request or open an issue.  The format of a Ryzer package can be quickly learned from existing packages and Co-Pilots and Chatbots are quite good at vibe-coding new packages.  For detailed guidelines, see CONTRIBUTING.md.

---

## License

This project is licensed under the MIT license. See the [`LICENSE`](LICENSE) file for details. 
