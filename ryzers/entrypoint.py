# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import os
import sys
import argparse
from .ryzer import RyzerManager
from .runner import DockerRunner

def build(base_path, name, packages, init_image):
    """
    Builds the Docker images using the specified packages path and selected packages.

    Args:
        base_path (str): The base directory to scan for Dockerfiles.
        name (str): The name of the Docker image to build.
        packages (list): List of package names to manage.
        init_image (str, optional): The initial base image to start with.
    """
    mgr = RyzerManager(base_path, name, packages, init_image)
    mgr.build()

def run(name, docker_cmd):
    """
    Runs the Docker container with the specified name.

    Args:
        name (str): The name of the Docker image to run.
        docker_cmd (str): The command to run with the Docker run call
    """
    runner = DockerRunner(name, docker_cmd)
    runner()

def main():
    """
    Main entry point for the command-line tool.

    Parses command-line arguments and executes the appropriate build or run command.
    """
    default_base_path = os.getcwd()

    parser = argparse.ArgumentParser(description="Command-line tool for building and running Ryzen AI Dockers.")
    
    # Define subcommands
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")
    
    # 'build' command
    build_parser = subparsers.add_parser("build", help="Build the project")
    build_parser.add_argument("--base_path", default=default_base_path, help=f"Base path for the project (default: CWD)")
    build_parser.add_argument("--name", default="ryzerdocker", help="Name of the docker image to build")
    build_parser.add_argument("dockerfiles", nargs="*", help="List of Dockerfiles to combine for the build (optional)")
    build_parser.add_argument("--init_image", default=None, help=f"Initial base image to start with (default: None)")

    # 'run' command
    run_parser = subparsers.add_parser("run", help="Run the project")
    run_parser.add_argument("--name", default=None, help="Name of the docker image to run")
    run_parser.add_argument("docker_cmd", nargs="?", default="", help="Overwrite the Docker CMD to run this command (optional)")

    # Parse the arguments
    args = parser.parse_args()

    if args.command == "build":
        build(args.base_path, args.name, args.dockerfiles, args.init_image)
    elif args.command == "run":
        run(args.name, args.docker_cmd)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)
