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

