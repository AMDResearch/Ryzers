# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from typing import List, Optional
from .runner import DockerRunner
from .packages import DockerPackageManager
from .builder import DockerBuilder
from . import RYZERS_DEFAULT_INIT_IMAGE

class RyzerManager:
    """
    Manages the building and running of Docker containers using multiple Dockerfiles and configurations.

    Attributes:
        docker_manager (DockerPackageManager): Manages Dockerfiles and configurations.
        docker_builder (DockerBuilder): Builds Docker images from Dockerfiles.
        docker_runner (DockerRunner): Runs Docker containers.
        container_name (str): The name of the container.
        base_image (str): The initial base image to start with.
        packages (List[str]): List of package names to combine.
    """

    def __init__(self, base_path: str, container_name: str, packages: List[str], base_image: Optional[str]=None):
        """
        Initializes the RyzerManager with the base path, container name, packages, and base image.

        Args:
            base_path (str): The base directory to scan for Dockerfiles.
            container_name (str): The name of the container.
            packages (List[str]): List of package names to manage.
            base_image (str, optional): The initial base image to start with. Defaults to RYZERS_DEFAULT_INIT_IMAGE in __init__.py.
        """
        self.packages = ["ryzer_env"] + packages
        self.docker_manager = DockerPackageManager(base_path, self.packages)
        self.docker_builder = DockerBuilder(container_name, self.packages)
        self.docker_runner = DockerRunner(container_name)   

        self.container_name = container_name
        self.base_image = base_image
 

    def build(self):
        """
        Builds Docker images from Dockerfiles and generates a run script.

        Retrieves build and run flags from the Docker manager, builds the Docker images,
        and generates a bash script to run the Docker container with the run flags.
        """
        buildflags = self.docker_manager.get_build_flags()
        runflags = self.docker_manager.get_run_flags()
        initimage = self.base_image if self.base_image is not None else self.docker_manager.get_initial_image()

        print(f'Build Flags: {buildflags}')
        print(f'Run Flags:   {runflags}')       
        print(f'Initial Image: {initimage}')


        self.docker_builder.build(self.docker_manager.dockerfiles, buildflags, initimage)
        self.docker_runner.build_runscript(runflags)
