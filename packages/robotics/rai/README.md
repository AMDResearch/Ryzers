# RAI

RAI (Robot Agent Interface) is a flexible AI agent framework to develop and deploy Embodied AI features for robots created and maintained by [Robotec.ai](https://www.robotec.ai/).

## Configuration

### ROS 2 setup

While RAI supports multiple ROS versions, this package has only been tested with the `jazzy` release. When building make sure your [ROS ryzer config](../../ros/ros/config.yaml) is using `jazzy` before proceeding to build this package.

### LLM vendor setup

RAI supports multiple vendors - OpenAI, AWS and Ollama. By default RAI uses OpenAI as the LLM vendor - if that is the one you want to use, make sure to set your `OPENAI_API_KEY` in this package's `config.yaml`:

```config
# Uncomment to set your OpenAI API key
# environment_variables:
#   - "OPENAI_API_KEY=your-api-key-here"
``` 

Local endpoints with Ollama are also supported, in which case simply build your ryzer with the additional ollama package.

## Build

We build RAI on top of the ROS 2 ryzer as a dependency:

```bash
ryzers build ros rai
```

## Run

```bash
# Run the test script
ryzers run
```

The default test will verify the environment and installed packages.

## Manipulation demo

To run the manipulation demo we need to add O3DE to the build 

```bash
ryzers build ros o3de rai
ryzers run /ryzers/manipulation_demo.sh
```

This will launch a streamlit app where you can chat with the robot agent and ask it to perform tasks on the table bench. Note that the simulation can take a while to load, give it a minute or two until the items spawn on the table. 

Once everything is loaded you can use the streamlit chat window to tell the robot agent to, for example, "stack the cubes".

<img src="images/manipulation_demo.gif">

## Run benchmarks with local LLM

### Ollama

We'll append the ollama package to enable locally run models.

```bash
ryzers build ros o3de rai ollama
ryzers run bash
```

It is recommended to visit the RAI [benchmarks documentation](https://robotecai.github.io/rai/tutorials/benchmarking/) for more details on individual benchmarks and options. Below we give an example of running an ollama server and running the 3 benchmarks for tool calling, manipulation and VLMs with models running locally on the iGPU.

```bash
# Start ollama server
ollama serve &

# Make sure to pull the models you want to test before launching benchmarks
ollama pull qwen2.5:7b
ollama pull gemma3:4b

# Setup ROS environment
cd /ryzers/rai
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# Run tool calling benchmark
python src/rai_bench/rai_bench/examples/tool_calling_agent.py --model-name qwen2.5:7b --vendor ollama --extra-tool-calls 5 --task-types basic  --n-shots 5 --prompt-detail descriptive --complexities easy

# Run manipulation benchmark
python src/rai_bench/rai_bench/examples/manipulation_o3de.py --model-name qwen2.5:7b --vendor ollama --levels trivial

# Run VLM benchmark
python src/rai_bench/rai_bench/examples/vlm_benchmark.py --model-name gemma3:4b --vendor ollama
```

By default we mount the benchmark results in an experiments directory on the path you ran `ryzers run` from. You will find a `results_summary.csv` for an overview there along with more detailed logs.

### Lemonade

We'll append the lemonade-sdk package to serve models locally through Lemonade's OpenAI-compatible API.

```bash
ryzers build ros o3de rai lemonade-sdk
ryzers run
```

By default `lemonade_env.sh` downloads and serves **Gemma-4-E2B-it-GGUF**. To use a different model, pass it as an argument (`source lemonade_env.sh <model>`) and update the `--model-name` flag to match. Again, below we give an example for running the lemonade-sdk on the same 3 benchmarks.

```bash
ryzers run bash

source lemonade_env.sh     # or: source lemonade_env.sh <other-model>

# Run tool calling benchmark
python src/rai_bench/rai_bench/examples/tool_calling_agent.py --model-name Gemma-4-E2B-it-GGUF --vendor openai --extra-tool-calls 5 --task-types basic --n-shots 5 --prompt-detail descriptive --complexities easy

# Run manipulation benchmark
python src/rai_bench/rai_bench/examples/manipulation_o3de.py --model-name Gemma-4-E2B-it-GGUF --vendor openai --levels trivial

# Run VLM benchmark (use a vision-capable model)
python src/rai_bench/rai_bench/examples/vlm_benchmark.py --model-name Gemma-4-E2B-it-GGUF --vendor openai
```

Available models and serving config live in `lemonade_env.sh`.

## Documentation

- Official docs: https://robotecai.github.io/rai/
- GitHub: https://github.com/RobotecAI/rai
