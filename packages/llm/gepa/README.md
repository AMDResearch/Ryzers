#  GEPA

This package contains the necessary files to build and run a Docker container for GEPA.

## Configuration

Optionally you can set your `OPENAI_API_KEY` in the config.yaml file to run some of the [out-of-the-box examples](https://github.com/gepa-ai/gepa?tab=readme-ov-file#simple-prompt-optimization-example).

```config
# Uncomment to set your OpenAI API key
# environment_variables:
#   - "OPENAI_API_KEY=your-api-key-here"
```

## Build the Docker Image

Execute the standard ryzers build command to install GEPA in a docker container.

```bash
ryzers build gepa
ryzers run
```

Our default run command will launch pytest to verify that the key components of GEPA are set up correctly.

---

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
