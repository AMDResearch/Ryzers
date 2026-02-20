# Flower
Flower is a framework for federated learning. This Ryzer demonstrates how to use Flower on AMD GPUs. This example is based on the Flower example found [here](https://flower.ai/docs/framework/docker/tutorial-quickstart-docker.html).

## Install Flower
Execute `pip install flower` to install Flower.

In `Ryzers/packages/federated/superexec/` create a getting started project:
`flwr  new  @flwrlabs/quickstart-pytorch`

Edit the `pyproject.toml` to comment out the dependencies on torch and torchvision (lines 13-14), as these packages are installed in the Ryzer already.

## Build the containers
Unlike other Ryzers, Flower requires several containers to be built. From the Ryzers directory, run;

```
ryzers build superexec --name superexec
ryzers build superlink --name superlink
ryzers build supernode --name supernode
```

## Create the network
Flower communicates between the containers using the docker network. Create a dedicated docker network for it to use:
`docker  network  create  --driver  bridge  flwr-network`

## Run the example
Run `Ryzers/packages/federated/superexec/flower.sh`. This will launch a superlink to coordinate the federation, supernodes (clients), and superexecs for scheduling applications per client.

Find your local config file:
`flwr  config  list`

Edit the config file to tell it about your superlink:
```
[superlink.local-deployment]
address  =  "127.0.0.1:9093"
insecure  =  true
```

Launch the quickstart application with 2 clients:

`flwr run . local-deployment --stream`

The example will use the GPU between 2 clients.
