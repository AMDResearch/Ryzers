# Package Structure

This section explains the structure and format of the `config.yaml` and `Dockerfile` files used to define packages in the Ryzers repository. Each package is self-contained and includes these files to specify its build and runtime configurations.

---

## `config.yaml`

The `config.yaml` file defines the metadata, build arguments, environment variables, and other configurations for a package. Below is the schema and explanation of its fields:

### Schema for `config.yaml`
```bash
init_image: <string>    
gpu_support: <boolean>  
x11_display: <boolean>  

build_arguments:
- <key1=value1>  
- <key2=value2> 

environment_variables:
- <key1=value1> 
- <key2=value2> 

port_mappings:
- <host_port1:container_port1>  
- <host_port2:container_port2>

volume_mappings:
- <host_path1:container_path1>  
- <host_path2:container_path2>     

docker_extra_run_flags: <string> 
```

### Field Descriptions

All fields are optional and all have a default value if not specified.

- **`init_image`**: (Optional) Specifies the initial base image to use for the build process. Defaults to `RYZER_DEFAULT`. For example, you can overwrite it with a PyTorch ROCm image.

- **`gpu_support`**: (Optional) A boolean value indicating whether GPU support is enabled. Defaults to `true`.

- **`x11_display`**: (Optional) A boolean value indicating whether X11 display forwarding is enabled. Defaults to `true`.

- **`build_arguments`**: A list of build arguments to pass to the Docker build process. Each argument is specified as a key-value pair. For example:
  - `"PYTHON_VERSION=3.10"` 

- **`environment_variables`**: A list of environment variables to set in the container. Each variable is specified as a key-value pair. For example:
  - `"HSA_OVERRIDE_GFX_VERSION=11.0.0"`

- **`port_mappings`**: A list of host-to-container port mappings. Each mapping is specified as `host_port:container_port`. For example:
  - `"8888:8888"`

- **`volume_mappings`**: A list of host-to-container volume mappings. Each mapping is specified as `host_path:container_path`. For example:
  - `"$PWD/workspace/.cache/huggingface:/root/.cache/huggingface"`

- **`docker_extra_run_flags`**: (Optional) A string of additional flags to pass to the `docker run` command. For example:
  - `"--shm-size=2g"`: Increases shared memory size to 2GB 


## Dockerfile
The Dockerfile specifies the steps to build the Docker image for the package. It must start with the ARG BASE_IMAGE line to ensure the ryzers build process which optionally can set an initial image based on the `config.yaml` file.

Example Dockerfile

```
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Addtional package-specific code here...

# Default command that is a test script to validate image build and
# optionally shows a demo of the package
CMD ["test.sh"]
```

### Ryzers Integration

Ryzers automatically adds environment variables and Docker flags to simplify the process of building and running these Dockerfiles. This ensures that hardware-specific configurations, such as GPU support and rendering settings, are applied automatically.

Additionally, Ryzers generates full `docker build` and `docker run` command lines as scripts, allowing users to further customize the build and runtime behavior if needed. These scripts can be found in the calling directory after running `ryzers build` or `ryzers run`.
