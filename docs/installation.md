# Installation

## System Requirements

We have tested all packages on Ubuntu 24.04 running on Strix-Halo (Ryzen AI Max) AI PCs.  Some packages do run on AMD Ryzen AI Strix and Phoenix devices as well - that support is listed within each package's README located in `packages/<category>/package_name`.  As more devices and OSs become fully supported, we will update this documentation. 

The only software requirement is Docker.

## Clone the Repository

To get started, clone the repository:

```bash
git clone https://github.com/AMDResearch/Ryzers
pip install Ryzers/
```

## Run an Example Ryzers Package

The [usage](usage) section will describe the Ryzers API in detail, but a simple build and run call below can check if the installation is correct.

```sh
ryzers build genesis
ryzers run
```

## Host Prerequisites

Docker is the only software requirement for the tooling itself, but building
and *running* packages also needs the host to pass the GPU into containers.
Run the preflight check before your first build:

```bash
bash scripts/check-host.sh
```

It verifies, in order:

- **OS** — tested on Ubuntu 24.04; other releases print a warning.
- **GPU device nodes** — `/dev/kfd` (ROCm compute) and `/dev/dri/renderD128`
  (render node) must exist on native Linux. Under WSL2 the GPU instead shows
  up as `/dev/dxg`.
- **User groups** — your user must be in the `render` and `video` groups:
  `sudo usermod -aG render,video $USER`, then log out and back in.
- **Container runtime** — Docker with a reachable daemon. Rootless Podman can
  substitute for building (wrap or alias it as `docker`); note that Podman
  does not resolve short image names against Docker Hub by default, so either
  use fully-qualified names (`docker.io/...`) or set
  `unqualified-search-registries = ["docker.io"]` in `registries.conf`.
- **Resources** — at least 30 GB free disk (100 GB recommended: the ROCm base
  images are ~20 GB before any package layers) and 8 GB RAM (16 GB
  recommended for the heavier packages).

The check prints one `[OK]` / `[WARN]` / `[FAIL]` line per item and exits
non-zero if anything failed, so it can gate `ryzers build` in a script or CI.
