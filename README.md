
![](docs/header.png)

# Ryzen AI Robotics, Vision and ML Dockerfiles

This repository provides a collection of composable Dockerfiles and build scripts for deploying and running software, full applications, and select demonstrators on AMD Ryzen AI hardware. The project is designed to streamline the setup of AI workloads, robotics, vision, and other applications optimized for Ryzen AI.

---

## Overview

Ryzers is a modular framework for building and running Docker containers tailored for AMD Ryzen AI hardware. It supports a wide range of applications, including machine learning, robotics, vision, and more. The repository is structured to allow easy composition of Dockerfiles, enabling users to build custom containers for their specific needs.

These dockerfiles will also be pushed and actively maintained in their original repository homes whenever possible.  Ryzers will be a collection point of software frameworks to run on AMD hardware.  We are committed to open-source and happy to accept contributions or feedback on packages hosted here.

---

## Supported Packages

| Category        | Software                                                                                                    |
|-----------------|--------------------------------------------------------------------------------------------------------------------|
| LLM                     | [`ollama`](packages/llm/ollama), [`llamacpp`](packages/llm/llamacpp), [`lmstudio`](packages/llm/lmstudio), [`GEPA`](packages/llm/gepa)   |
| VLM                     | [`Gemma3`](packages/vlm/gemma3), [`SmolVLM`](packages/vlm/smolvlm), [`Phi-4`](packages/vlm/phi4), [`LFM2-VL`](packages/vlm/lfm2vl) |
| VLA                     | [`OpenVLA`](packages/vla/openvla), [`SmolVLA`](packages/vla/smolvla), [`GR00T-N1.5`](packages/vla/gr00t), [`openpi`](packages/vla/openpi), [`CogACT`](packages/vla/cogact) |
| Graphics                     | [`O3DE`](packages/graphics/o3de) |
| Robotics                | [`ROS 2`](packages/ros/ros), [`Gazebo`](packages/ros/gazebo), [`LeRobot`](packages/robotics/lerobot), [`ACT`](packages/robotics/act), [`RAI`](packages/robotics/rai)    |
| Simulation                |  [`Genesis`](packages/robotics/genesis)  |
| Vision                  | [`OpenCV`](packages/vision/opencv), [`SAM`](packages/vision/sam), [`MobileSAM`](packages/vision/mobilesam), [`ncnn`](packages/vision/ncnn), [`DINOv3`](packages/vision/dinov3), [`SAM3`](packages/vision/sam3), [`Ultralytics`](packages/vision/ultralytics) |
| Ryzen AI NPU                |  [`XDNA`](packages/npu/xdna), [`IRON`](packages/npu/iron), [`NPUEval`](packages/npu/npueval), [`Ryzen AI CVML`](packages/npu/ryzenai_cvml)  |
| Adaptive SoCs           | [`PYNQ.remote`](packages/adaptive-socs/pynq-remote) |
| Utilities   | [`JupyterLab`](packages/ide/jupyterlab), [`amdgpu_top`](packages/init/amdgpu_top) |

---

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/AMDResearch/Ryzers
pip install Ryzers/
```

For detailed installation instructions and requirements, refer to the [included documentation](https://amdresearch.github.io/Ryzers/installation.html).

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

For detailed build and run instructions, refer to the instructions included in the package.

## Features

- **Verified AMD Support for Popular Frameworks**: A variety of Robotics and ML software frameworks supported across Ryzen AI platforms.
- **Optimized for Ryzen AI**: Includes support for hardware-accelerated AI workloads and accelerators like the iGPU and NPUs.
- **Minimal Host Software Requirements**: Standard Ubuntu support with minimal software requirements
- **Composable Dockerfiles**: Modular design for reusability across different applications.

---

## Contributing

We welcome contributions to Ryzers! If you have ideas for new features, bug fixes, or improvements, please submit a pull request or open an issue.  The format of a Ryzer package can be quickly learned from existing packages and Co-Pilots and Chatbots are quite good at vibe-coding new packages.  For detailed guidelines, see [CONTRIBUTING.md](docs/contributing.md).

---

## License

This project is licensed under the MIT license. See the [`LICENSE`](LICENSE) file for details. 
