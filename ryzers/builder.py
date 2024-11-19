# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import os
import yaml
import subprocess
from typing import List

class DockerBuilder:
    """
    A class to build Docker images sequentially from a list of Dockerfiles.
    """

    def __init__(self, container_name: str, packages: List[str]):
        """
        Initialize the DockerBuilder with container name, packages, and base image.

        :param container_name: Name of the final container image.
        :param packages: List of package names corresponding to each Dockerfile.
        :param base_image: Base image to use for the Docker builds.
        """
        self.container_name = container_name
        self.packages = packages

    def build(self, dockerfiles: List[str], buildflags: List[str], initial_image: str):
        """
        Build each Dockerfile sequentially, using the last built image as the base image.

        :param dockerfiles: List of paths to Dockerfiles.
        :param buildflags: List of build flags to use during the Docker build.
        """
        current_base_image = initial_image
        for index, dockerfile_path in enumerate(dockerfiles):
            if not os.path.exists(dockerfile_path):
                raise FileNotFoundError(f"Dockerfile not found: {dockerfile_path}")

            image_tag = f"{self.packages[index]}"
            if index == len(dockerfiles) - 1:
                image_tag = self.container_name

            print(f"Building Dockerfile '{index+1}/{len(dockerfiles)} {dockerfile_path}' with base image '{current_base_image}'...")

            # Build the Docker image with the current base image
            package_path = os.path.abspath(os.path.dirname(dockerfile_path))
            log_file = f'ryzers.build.{image_tag}.log'

            cmd = f"docker build -t {image_tag} {buildflags} --build-arg BASE_IMAGE={current_base_image} {package_path}"
            print(cmd)
            with open(log_file, "w") as log:
                subprocess.run(
                    cmd.split(), 
                    check=True
                )

            self._write_last_built()
            print(f"Successfully built image: {image_tag}")

            # Update the base image for the next build
            current_base_image = image_tag

    def _write_last_built(self):
        last_built_file = os.path.join(os.path.dirname(__file__), '_ryzers.yaml')
        data = {'last_built_image': self.container_name}

        with open(last_built_file, 'w') as f:
            yaml.dump(data, f)
