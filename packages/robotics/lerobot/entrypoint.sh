#!/usr/bin/env bash
set -e

if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_USER
    HF_USER="$(python -c 'from huggingface_hub import whoami; print(whoami()["name"])')"
    echo "Logged in as ${HF_USER}"
fi

exec "$@"
