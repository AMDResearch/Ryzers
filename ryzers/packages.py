# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import os
from typing import List, Any, Set
from .configs import ConfigManager

class DockerPackageManager:
    """
    Manages a collection of Dockerfiles.
    Scans a base directory for subdirectories containing Dockerfiles and stores
    folder names and paths in a dictionary.
    """

    def __init__(self, base_path, packages: List[str]):
        """
        Initialize the DockerPackageManager with base path and packages.

        :param base_path: The base directory to scan for Dockerfiles.
        :param packages: List of package names to manage.
        """
        self.base_path = base_path
        self.packages = packages

        self._package_path_map = self._find_allpackages()

        self.dockerfiles = self.get_packages_filelist("Dockerfile")
        self.configfiles = self.get_packages_filelist("config.yaml", optional=True)
        self.config_manager = ConfigManager(self.configfiles)

    def _find_allpackages(self, current_path=None, current_depth=0, max_depth=4):
        """
        Recursively finds all Dockerfiles in the base directory up to a maximum depth and returns 
        a dictionary with folder name as key and path as value.

        :param current_path: The current directory path being scanned.
        :param current_depth: The current depth of the directory tree being scanned.
        :param max_depth: The maximum depth to scan.
        :return: A dictionary with folder name as key and path as value.
        """
        if current_path is None:
            current_path = self.base_path

        if current_depth > max_depth:
            return {}

        dockerfile_path_map = {}
        for entry in os.listdir(current_path):
            entry_path = os.path.join(current_path, entry)
            if self._is_directory_readable(entry_path):
                if os.path.exists(os.path.join(entry_path, "Dockerfile")):
                    dockerfile_path_map[entry] = entry_path
                else:
                    dockerfile_path_map.update(self._find_allpackages(entry_path, current_depth + 1, max_depth))
        return dockerfile_path_map

    def get_packages_filelist(self, filename, optional=False):
        """
        Get a list of file paths for the specified filename in the managed packages.

        :param filename: The name of the file to search for.
        :param optional: Whether the file is optional.
        :return: A list of file paths.
        """
        pkg_filename_paths = []

        for pkg in self.packages:
            if pkg not in self._package_path_map:
                raise FileNotFoundError(f'Cannot find required package: {pkg}')

            full_file_path = os.path.join(self._package_path_map[pkg], filename)
            if os.path.isfile(full_file_path):
                pkg_filename_paths.append(full_file_path)
            elif optional:
                pass
            else:
                raise FileNotFoundError(f'Cannot find required file: {full_file_path}')

        return pkg_filename_paths

    def get_build_flags(self):
        return self.config_manager.configs_to_buildflags()

    def get_run_flags(self):
        return self.config_manager.configs_to_runflags()
    
    def get_initial_image(self):
        """
        Returns the initial image to use for the Docker build.
        If no initial image is specified, returns the default image.
        """
        return self.config_manager.configs_to_initial_image()

    def _is_directory_readable(self, path):
        return (os.path.exists(path) and 
                os.path.isdir(path) and 
                os.access(path, os.R_OK))