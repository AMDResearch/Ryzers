# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Flower Quickstart: Client application for federated learning."""

import torch
from flwr.client import ClientApp, NumPyClient

from quickstart_amd.task import DEVICE, Net, load_data, test, train


class FlowerClient(NumPyClient):
    """Flower client for federated learning on CIFAR-10."""

    def __init__(self, partition_id: int, num_partitions: int, batch_size: int):
        self.partition_id = partition_id
        self.num_partitions = num_partitions
        self.batch_size = batch_size
        self.net = Net().to(DEVICE)
        self.trainloader, self.valloader = load_data(
            partition_id, num_partitions, batch_size
        )

    def get_parameters(self, config):
        """Return model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy arrays."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train model on local data."""
        self.set_parameters(parameters)
        lr = config.get("lr", 0.1)
        epochs = config.get("local_epochs", 1)
        train_loss = train(self.net, self.trainloader, epochs, lr, DEVICE)
        return (
            self.get_parameters(config),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        """Evaluate model on local data."""
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader, DEVICE)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context):
    """Create a Flower client for the federation."""
    # Debug: print available config keys
    print(f"[DEBUG] node_config keys: {list(context.node_config.keys())}")
    print(f"[DEBUG] node_config: {dict(context.node_config)}")
    print(f"[DEBUG] run_config: {dict(context.run_config)}")

    # Get partition info with defaults (for deployment mode compatibility)
    partition_id = context.node_config.get("partition-id", 0)
    num_partitions = context.node_config.get("num-partitions", 2)
    batch_size = context.run_config.get("batch-size", 32)

    print(f"[CLIENT] partition_id={partition_id}, num_partitions={num_partitions}, batch_size={batch_size}")
    return FlowerClient(partition_id, num_partitions, batch_size).to_client()


# Create the ClientApp
app = ClientApp(client_fn=client_fn)
