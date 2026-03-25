#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/env.sh"

export FLOWER_SCRIPTS="$SCRIPT_DIR"

# Launch each component in a separate terminal tab
gnome-terminal --tab --title="Superlink" -- bash -c 'echo -ne "\033]0;Superlink\007"; '"$FLOWER_SCRIPTS"'/start_superlink.sh; exec bash'

sleep 5
gnome-terminal --tab --title="Supernodes" -- bash -c 'echo -ne "\033]0;Supernodes\007"; '"$FLOWER_SCRIPTS"'/start_supernodes.sh; exec bash'

sleep 5
gnome-terminal --tab --title="Superexecs" -- bash -c 'echo -ne "\033]0;Superexecs\007"; '"$FLOWER_SCRIPTS"'/start_superexecs.sh; exec bash'

echo "All Flower components started in separate terminal tabs"
echo "To view logs, check each terminal tab"
