#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

source env.sh
source $SPSDEMO_VENV/bin/activate

export SPSDEMO_SCRIPTS=$SPSDEMO_SCRIPTS

gnome-terminal --tab --title="SmolVLM" -- bash -c 'echo -ne "\033]0;SmolVLM\007"; $SPSDEMO_SCRIPTS/start_smolvlm.sh; exec bash'

sleep 5
gnome-terminal --tab --title="Genesis" -- bash -c 'echo -ne "\033]0;Genesis\007"; $SPSDEMO_SCRIPTS/start_genesis.sh; exec bash'

sleep 5
gnome-terminal --tab --title="CVML" -- bash -c 'echo -ne "\033]0;CVML\007"; $SPSDEMO_SCRIPTS/start_cvml.sh; exec bash'
