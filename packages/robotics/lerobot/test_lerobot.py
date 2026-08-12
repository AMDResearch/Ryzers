#!/usr/bin/env python3

from importlib.metadata import version

import torch

from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies.diffusion import DiffusionConfig, DiffusionPolicy
from lerobot.utils.feature_utils import dataset_to_policy_features


if torch.version.hip is None:
    raise RuntimeError(f"PyTorch is not a ROCm build: {torch.__version__}")
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot access the ROCm GPU")

device = torch.device("cuda:0")
properties = torch.cuda.get_device_properties(device)
print(
    f"LeRobot {version('lerobot')} | PyTorch {torch.__version__} | "
    f"HIP {torch.version.hip} | {properties.name} ({properties.gcnArchName})"
)

metadata = LeRobotDatasetMetadata("lerobot/pusht")
features = dataset_to_policy_features(metadata.features)
output_features = {
    key: feature
    for key, feature in features.items()
    if feature.type is FeatureType.ACTION
}
input_features = {
    key: feature for key, feature in features.items() if key not in output_features
}

config = DiffusionConfig(
    input_features=input_features,
    output_features=output_features,
    device="cuda",
    pretrained_backbone_weights=None,
)
policy = DiffusionPolicy(config).to(device).train()
if next(policy.parameters()).device != device:
    raise RuntimeError("LeRobot policy was not moved to the ROCm GPU")

batch_size = 2
batch = {
    "observation.image": torch.rand(
        batch_size, config.n_obs_steps, 3, 96, 96, device=device
    ),
    "observation.state": (
        torch.rand(batch_size, config.n_obs_steps, 2, device=device) * 2 - 1
    ),
    "action": (
        torch.rand(batch_size, config.horizon, 2, device=device) * 2 - 1
    ),
    "action_is_pad": torch.zeros(
        batch_size, config.horizon, dtype=torch.bool, device=device
    ),
}

loss, _ = policy.forward(batch)
if not torch.isfinite(loss):
    raise RuntimeError(f"LeRobot produced a non-finite loss: {loss}")

loss.backward()
print(f"LeRobot GPU forward/backward passed with loss={loss.detach().item():.6f}")
