#!/usr/bin/env bash
# Source me, don't run me:   export OPENAI_API_KEY=sk-... ; source openai_env.sh
# Preps the container to run RAI benchmarks against real OpenAI CLOUD models,
# then leaves the benchmark command up to you.

# Refuse to run with a missing / dummy key (the Lemonade placeholder)
if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "lemonade" ]; then
    echo "Set your real key first:  export OPENAI_API_KEY=sk-..." >&2
    return 1 2>/dev/null || exit 1
fi

# Point RAI's [openai] base_url back to the real endpoint. Sets it regardless
# of current value, so this works no matter which backend you ran last.
sed -i 's|^base_url = .*|base_url = "https://api.openai.com/v1/"|' /ryzers/rai/config.toml

# ROS env (runtime = interactive bash, so .bash)
cd /ryzers/rai
source /opt/ros/jazzy/setup.bash
source install/setup.bash

echo "OpenAI ready. Run e.g.:"
echo "  python src/rai_bench/rai_bench/examples/manipulation_o3de.py --model-name gpt-4o --vendor openai --levels trivial"
