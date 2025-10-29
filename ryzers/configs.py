# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import os
import yaml
import glob
from typing import List, Tuple

from . import RYZERS_DEFAULT_RUN_FLAGS
from . import RYZERS_DEFAULT_INIT_IMAGE

class ConfigKeyValueEntry():
    def __init__(self, key, value, configyaml_path):
        self.key = key
        self.value = value
        self.configyaml_path = configyaml_path
        self.comments = ""

    def __str__(self):
        return str(self.value)


class ConfigKeyKeyValueEntry(ConfigKeyValueEntry):
    def __init__(self, key, key1, value, configyaml_path, sep="="):
        super().__init__(key, value, configyaml_path)
        self.sep = sep
        self.key1 = key1

    def __str__(self):
        return f"{self.key1}{self.sep}{self.value}"


class ConfigManager:
    """
    A class to manage and parse config.yaml files sitting next to Dockerfiles.

    Attributes:
        dockerfile_paths (List[str]): List of paths to Dockerfiles.
    """
    CONFIG_KEYS = [
        "init_image",
        "gpu_support",
        "x11_display",
        "camera_support",
        "build_arguments",
        "environment_variables",
        "port_mappings",
        "volume_mappings",
        "run_arguments",
        "docker_extra_build_flags",
        "docker_extra_run_flags"
    ]

    def __init__(self, configfiles: List[str]):
        self.configfiles = configfiles
        self.configentries = self.get_configentries()

    def configs_to_initial_image(self):
        """
        Returns the initial image to use for the Docker build.
        If no initial image is specified, returns the default image.
        """
        for c in self.configentries:
            if c.key == "init_image":
                return c.value
        return RYZERS_DEFAULT_INIT_IMAGE

    def configs_to_buildflags(self):

        # get last unique build argument value for each key
        buildflags_dict = {}
        for c in self.configentries:
            if c.key == "build_arguments":
                buildflags_dict[c.key1] = c 

        # overwrite with environment variables
        for e in os.environ:
            if e in buildflags_dict:
                buildflags_dict[e] = ConfigKeyKeyValueEntry("build_arguments", e, os.environ[e], None)  

        return " ".join([f"--build-arg {configentry}" for _, configentry in buildflags_dict.items()])

    def configs_to_runflags(self):
        runflags = RYZERS_DEFAULT_RUN_FLAGS
        gpu = True
        x11 = True
        cameras = True
        render_group = True
        extra_run_flags = ""

        for c in self.configentries:
            print(str(c))
            if c.key == "port_mappings":
                runflags += f" -p {c}" 
            if c.key == "volume_mappings":
                runflags += f" -v {c}"                 
            if c.key == "environment_variables":
                runflags += f" -e {c}"  
            if c.key == "gpu_support":
                gpu = c.value
            if c.key == "x11_display":
                x11 = c.value
            if c.key == "camera_support":
                cameras = c.value
            if c.key == "run_arguments" and c.key1 == "render_group":
                render_group = c.value.lower() == 'true' if isinstance(c.value, str) else bool(c.value)
            if c.key == "docker_extra_run_flags":
                extra_run_flags += f" {c.value}"
        if gpu:
            if render_group:
                runflags += " --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --group-add video --group-add render "
            else:
                runflags += " --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --group-add video "
        if x11:
            runflags += " -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix"
        if cameras:
            vdevices = sorted(glob.glob('/dev/video*'))
            runflags += f" {' '.join(f'--device {vdev}' for vdev in vdevices)}"
        if extra_run_flags:
            runflags += extra_run_flags

        return runflags

    def get_configentries(self):
        """
        Collects all the values associated with an input key from the config.yaml files.
        Args:
            key (str): The key to search for in the config.yaml files.
        Returns:
            List[Tuple[str, str, str]]: A list of tuples where each tuple contains the string value and the file it was found in.
        """

        configs = []

        for cfile_path in self.configfiles:
            try:
                with open(cfile_path, 'r') as file:
                    config = yaml.safe_load(file)

                    if config is None:
                        continue
                    
                    for key in config:

                        if key not in self.CONFIG_KEYS:
                            raise ValueError(f"Invalid key: {key}. Must be one of {', '.join(self.CONFIG_KEYS)}")

                        values = config[key]

                        if isinstance(values, list):
                            for value in values:
                                configs.append(self.get_configentry_obj(key, value, cfile_path))
                        else:
                            configs.append(self.get_configentry_obj(key, values, cfile_path))

            except yaml.YAMLError as e:
                print(f"Error reading YAML file {cfile_path}: {e}")

        return configs
    
    def get_configentry_obj(self, key: str, value: str, config_path: str) -> Tuple[str, str]:
        if key in ["build_arguments", "environment_variables", "run_arguments"]:
            key0, val0 = value.split('=', 1)
            return ConfigKeyKeyValueEntry(key, key0, val0, config_path)
        if key in ["port_mappings", "volume_mappings"]:
            key0, val0 = value.split(':', 1)
            return ConfigKeyKeyValueEntry(key, key0, val0, config_path, sep=':')
        if key in ["gpu_support", "x11_display", "docker_extra_build_flags", "docker_extra_run_flags", "init_image"]:
            return ConfigKeyValueEntry(key, value, config_path)
        else:
            raise ValueError(f"Invalid key: {key}. Must be one of {', '.join(self.CONFIG_KEYS)}")
