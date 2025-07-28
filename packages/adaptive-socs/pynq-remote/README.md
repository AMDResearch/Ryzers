# PYNQ.remote Docker Setup

This folder contains the necessary files to build and run a Docker container with PYNQ.remote installed using Ryzers.

## Building and running the Docker Image

```bash
# Build PYNQ container with JupyterLab for interactive development
ryzers build pynq-remote jupyterlab
ryzers run
```

---

## Running JupyterLab

The PYNQ container has a default command that launches JupyterLab when built with the jupyterlab package.

### Accessing JupyterLab
1. Open your web browser and go to: `http://localhost:8888`
2. Navigate to the `/workspace` directory for your notebooks and PYNQ development

---

### Using PYNQ.remote
For detailed usage instructions, refer to the [PYNQ documentation](https://pynq.readthedocs.io/en/latest/).

## Ports
- **8888**: JupyterLab interface
- **7967**: PYNQ.remote communication