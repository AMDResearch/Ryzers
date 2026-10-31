# Dev Containers Support for Ryzers

Ryzers now supports attaching to running containers using VS Code's Dev Containers extension or Docker extension.

## Overview

When you build a Ryzer container, the system automatically configures it to be compatible with VS Code's container attachment features. This allows you to:

- Attach VS Code directly to running Ryzer containers
- Use the full IDE experience inside the container
- Access all container resources and mounted volumes
- Debug and develop directly in the containerized environment

## How It Works

### Automatic Configuration

When you run `ryzers build`, the system generates a run script that:

1. **Removes the `--rm` flag** - Containers persist after exit, allowing attachment
2. **Adds a named container** - Format: `ryzer-<container_name>`
3. **Adds Dev Containers labels** - Helps VS Code identify the container
4. **Checks for existing containers** - Automatically removes old instances before starting new ones

### Container Naming

Containers are named with the prefix `ryzer-` followed by your image name:
- Image: `ryzerdocker` → Container: `ryzer-ryzerdocker`
- Image: `my-custom-build` → Container: `ryzer-my-custom-build`

## Usage

### Method 1: VS Code Docker Extension (Recommended)

1. **Install the Docker extension**:
   - Open VS Code Extensions (Ctrl+Shift+X)
   - Search for "Docker" by Microsoft
   - Install the extension

2. **Start your Ryzer container**:
   ```bash
   ryzers build ros o3de rai ollama
   ryzers run bash
   ```

3. **Attach VS Code**:
   - Open the Docker panel in VS Code (Ctrl+Shift+D or click Docker icon)
   - Find your container under "Containers" (e.g., `ryzer-ryzerdocker`)
   - Right-click → "Attach Visual Studio Code"
   - A new VS Code window opens connected to the container

4. **Start developing**:
   - Open folders inside the container (e.g., `/ryzers/rai`)
   - Edit files, run terminals, debug - all inside the container
   - Access mounted volumes like `/ryzers/rai/src/rai_bench/rai_bench/experiments`

### Method 2: Dev Containers Extension

1. **Install the Dev Containers extension**:
   - Open VS Code Extensions (Ctrl+Shift+X)
   - Search for "Dev Containers" by Microsoft
   - Install the extension

2. **Start your Ryzer container**:
   ```bash
   ryzers run bash
   ```

3. **Attach using Command Palette**:
   - Press F1 or Ctrl+Shift+P
   - Type "Dev Containers: Attach to Running Container"
   - Select your container from the list (e.g., `ryzer-ryzerdocker`)

### Method 3: Docker CLI

For quick terminal access without VS Code:

```bash
# List running containers
docker ps

# Attach to your Ryzer container
docker exec -it ryzer-ryzerdocker bash

# Or use the container name pattern
docker exec -it $(docker ps --filter "name=ryzer-" --format "{{.Names}}") bash
```

## Container Management

### Viewing Running Containers

```bash
# List all Ryzer containers
docker ps --filter "name=ryzer-"

# List all containers (including stopped)
docker ps -a --filter "name=ryzer-"
```

### Stopping Containers

```bash
# Stop a specific container
docker stop ryzer-ryzerdocker

# Stop all Ryzer containers
docker stop $(docker ps --filter "name=ryzer-" --format "{{.Names}}")
```

### Removing Containers

```bash
# Remove a specific container
docker rm ryzer-ryzerdocker

# Remove all stopped Ryzer containers
docker rm $(docker ps -a --filter "name=ryzer-" --format "{{.Names}}")
```

The run script automatically removes existing containers with the same name before starting, so you typically don't need to manually clean up.

## Configuration

### Custom Dev Container Settings

A default `.devcontainer/devcontainer.json` file is provided in the repository root. You can customize it for your workflow:

```json
{
  "name": "Ryzers Development Container",
  "workspaceFolder": "/workspace",
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-azuretools.vscode-docker"
      ],
      "settings": {
        "python.defaultInterpreterPath": "/opt/venv/bin/python"
      }
    }
  }
}
```

### Package-Specific Configurations

Each package's `config.yaml` can specify:
- Port mappings (accessible from host)
- Volume mappings (shared directories)
- Environment variables
- GPU/X11 support

These are automatically included in the container configuration.

## Troubleshooting

### Container Not Appearing in VS Code

1. Ensure the container is running: `docker ps --filter "name=ryzer-"`
2. Refresh the Docker extension view
3. Check container logs: `docker logs ryzer-ryzerdocker`

### Permission Issues

If you encounter permission errors:
```bash
# The container runs as root by default
# Files created in mounted volumes will be owned by root
# Use chown on the host if needed:
sudo chown -R $USER:$USER ./experiments
```

### X11 Display Issues

If GUI applications don't work after attaching:
```bash
# Inside the container, ensure DISPLAY is set
echo $DISPLAY

# On host, ensure X11 forwarding is enabled
xhost +local:docker
```

### Container Keeps Restarting

If you run `ryzers run bash` and the container exits immediately:
- The `bash` command requires interactive mode (`-it` flag is included)
- Check if there are errors in the container logs
- Try running without the `bash` override to use the default CMD

## Best Practices

1. **Use named builds** for different projects:
   ```bash
   ryzers build --name my-project ros rai
   ryzers run --name my-project bash
   ```

2. **Mount your workspace** for persistent changes:
   - Add volume mappings in package `config.yaml`
   - Or use `docker run -v` flags in the generated script

3. **Stop containers when done** to free resources:
   ```bash
   docker stop ryzer-ryzerdocker
   ```

4. **Use the Docker extension** for visual container management

## Examples

### Attaching to RAI Benchmark Container

```bash
# Build with all dependencies
ryzers build ros o3de rai ollama

# Run with bash to keep container alive
ryzers run bash

# In VS Code:
# 1. Open Docker panel
# 2. Right-click "ryzer-ryzerdocker"
# 3. Select "Attach Visual Studio Code"
# 4. Open folder: /ryzers/rai
# 5. Edit benchmark scripts, run tests, etc.
```

### Multiple Containers for Different Projects

```bash
# Build different configurations
ryzers build --name rai-dev ros rai
ryzers build --name genesis-sim genesis

# Run both
ryzers run --name rai-dev bash &
ryzers run --name genesis-sim bash &

# Attach to each in separate VS Code windows
```

## See Also

- [VS Code Dev Containers Documentation](https://code.visualstudio.com/docs/devcontainers/containers)
- [Docker Extension Documentation](https://code.visualstudio.com/docs/containers/overview)
- [Ryzers Build Documentation](../README.md)
