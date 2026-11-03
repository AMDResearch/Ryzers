#!/usr/bin/env bash
# Source me, don't run me:   source lemonade_env.sh [MODEL]
# Preps the container to run RAI benchmarks against a LOCAL Lemonade model,
# then leaves the benchmark command up to you.

MODEL="${1:-Gemma-4-E2B-it-GGUF}"

# Start lemond if it isn't already up; log to /tmp/lemond.log
if ! lemonade status >/dev/null 2>&1; then
    lemond > /tmp/lemond.log 2>&1 &
    until lemonade status >/dev/null 2>&1; do sleep 1; done   # wait for server
fi

# Load the model (downloads on first use; no-op if already loaded)
lemonade load "$MODEL"

# Point RAI's [openai] base_url at Lemonade. Sets it regardless of current
# value, so this works no matter which backend you ran last.
sed -i 's|^base_url = .*|base_url = "http://localhost:13305/api/v0"|' /ryzers/rai/config.toml
export OPENAI_API_KEY="lemonade"   # dummy; Lemonade ignores it
# Point the [openai] model name to do the same sourced in lemonade.
sed -i '/^\[openai\]/,/^\[/{s|^simple_model = .*|simple_model = "'"$MODEL"'"|; s|^complex_model = .*|complex_model = "'"$MODEL"'"|}' /ryzers/rai/config.toml

# ROS env (runtime = interactive bash, so .bash)
cd /ryzers/rai
source /opt/ros/jazzy/setup.bash
source install/setup.bash

echo "Lemonade ready. Run e.g.:"
echo "  python src/rai_bench/rai_bench/examples/manipulation_o3de.py --model-name $MODEL --vendor openai --levels trivial"
echo "  bash /ryzers/manipulation_demo.sh"