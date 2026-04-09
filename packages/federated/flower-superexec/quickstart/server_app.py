# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Flower Quickstart: Server application for federated learning."""

import torch
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg

from quickstart.task import Net


def server_fn(context):
    """Configure the Flower server."""
    # Read configuration
    num_rounds = context.run_config["num-server-rounds"]
    fraction_evaluate = context.run_config["fraction-evaluate"]

    # Initialize global model parameters
    net = Net()
    ndarrays = [val.cpu().numpy() for _, val in net.state_dict().items()]
    parameters = ndarrays_to_parameters(ndarrays)

    # Configure FedAvg strategy
    strategy = FedAvg(
        fraction_fit=1.0,  # Sample 100% of clients for training
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        initial_parameters=parameters,
        on_fit_config_fn=lambda server_round: {
            "lr": context.run_config["learning-rate"],
            "local_epochs": context.run_config["local-epochs"],
        },
    )

    # Server configuration
    config = ServerConfig(num_rounds=num_rounds)

    return strategy, config


# Create the ServerApp
app = ServerApp(server_fn=server_fn)
