### openpi

This directory contains the docker configuration files to run [openpi from Physical Intelligence](https://github.com/Physical-Intelligence/openpi).

### Build and run the Docker Image

```sh
ryzers build openai
ryzers run
```

### Running other models

For example, to download and convert pi0.5 use the following commands:

```
cd /ryzers/openpi

# Download
uv run scripts/serve_policy.py --download-only policy:checkpoint --policy.config=pi05_droid --policy.dir=gs://openpi-assets/checkpoints/pi05_droid

# Convert JAX model to PyTorch
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /root/.cache/openpi/openpi-assets/checkpoints/pi05_droid \
    --config_name pi05_droid \
    --output_path /root/.cache/openpi/openpi-assets/checkpoints/pi05_droid
```

Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
