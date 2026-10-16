#!/usr/bin/env bash
# Source me, don't run me:   source lemonade_env.sh [MODEL]
# Preps the container to run RAI benchmarks against a LOCAL Lemonade model,
# then leaves the benchmark command up to you.
#
# Serving the model is useful on its own, without any of the RAI or ROS setup
# that follows it, so that half can be run in a subshell instead:
#     bash lemonade_env.sh --serve-only [MODEL]
# That is what notebook 3 does. It skips the config.toml rewrite and the ROS
# overlay below, which belong to RAI and would put the wrong cwd and the wrong
# Python packages on the CaP-X kernel.

SERVE_ONLY=0
if [ "${1:-}" = "--serve-only" ]; then
    SERVE_ONLY=1
    shift
fi

MODEL="${1:-Gemma-4-E2B-it-GGUF}"
LEMONADE_CACHE="${LEMONADE_CACHE:-/opt/lemonade-cache/lemonade}"
LEMONADE_HF_HOME="${LEMONADE_HF_HOME:-/opt/lemonade-cache/huggingface}"
export HF_HOME="${LEMONADE_HF_HOME}"

# Start lemond if it isn't already up; log to /tmp/lemond.log
if ! lemonade status >/dev/null 2>&1; then
    lemond "${LEMONADE_CACHE}" > /tmp/lemond.log 2>&1 &
    # Bounded, so a server that never binds fails here instead of hanging the
    # shell (or, under --serve-only, the caller waiting on this script)
    for _ in $(seq 300); do
        lemonade status >/dev/null 2>&1 && break
        sleep 1
    done
    if ! lemonade status >/dev/null 2>&1; then
        echo "lemond did not come up - see /tmp/lemond.log" >&2
        return 1 2>/dev/null || exit 1
    fi
fi

# Load the image-cached model (no-op if already loaded).
lemonade load "$MODEL"

if [ "$SERVE_ONLY" -eq 0 ]; then
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
fi

# Headless run only since this script is for the roscon26 conference
echo "Lemonade ready"
