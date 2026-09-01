# CONTACT Docker Setup

This Ryzer packages the AMD Track A path for
[CONTACT](https://github.com/AdeeshDesai/CONTACT): training a Diffusion Policy
from pre-collected Zarr observations and reporting offline validation loss and
action error on a ROCm GPU.

The package keeps the NVIDIA IsaacGym and TacSL paths in CONTACT, but disables
simulator rollout at runtime with `USE_SIM=0` and
`training.enable_rollout=false`.

## Supported Scope

- Primary validated target: Strix Halo `gfx1151`
- ROCm 7.14, Python 3.12, PyTorch 2.12
- Vision, tactile RGB, and tactile force-field datasets
- Training and offline evaluation
- No simulator rollout or success-rate evaluation

## Prepare the Dataset

The default smoke test uses the `barbed_flat` dataset. From the Ryzers
repository root:

```bash
python -m pip install gdown
mkdir -p workspace/contact/data
gdown 1hNeYjXp000xy26wsvmtBVorjF-w4TguG \
  -O workspace/contact/data/barbed_flat.zip
unzip workspace/contact/data/barbed_flat.zip -d workspace/contact/data
rm workspace/contact/data/barbed_flat.zip
mkdir -p workspace/contact/outputs
```

The extracted dataset must be available at
`workspace/contact/data/barbed_flat/.zgroup`.

## Build and Run

```bash
ryzers build contact
ryzers run
```

The default command verifies the ROCm GPU, runs a synthetic CONTACT policy
forward/backward step, loads the mounted dataset, and runs one TacFF training
and validation step. Metrics are written under `workspace/contact/outputs`.

To open an interactive shell:

```bash
ryzers run bash
```

## Build Arguments

The package pins both source repositories. Override a pin without editing the
package:

```bash
CONTACT_REPO=https://github.com/your-user/CONTACT.git \
CONTACT_BRANCH=feat/amd-rocm \
CONTACT_REF=<commit> \
ryzers build contact
```

## Limitations

Track A evaluates validation loss and action prediction error only. Reproducing
CONTACT simulator success rates on AMD requires the planned Genesis environment
replacement; results from a different physics engine will be a new AMD
reference baseline rather than numerically identical IsaacGym results.

## References

- [CONTACT](https://github.com/AdeeshDesai/CONTACT)
- [CONTACT paper](https://arxiv.org/abs/2603.08560)
- [Ryzers](https://github.com/AMDResearch/Ryzers)

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
