#!/usr/bin/env bash
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Fetch RoboTwin 2.0 configs + assets into the mounted volume and wire them into the
# de-vendored RoboTwin at /opt/RoboTwin. Idempotent (marker-gated).
set -euo pipefail
RT_COMMIT="${RT_COMMIT:-bf44be51cf5717a5595ce59447f2cf5263d2aa95}"
RAW="https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/${RT_COMMIT}"
REPO_RT="${ROBOTWIN_ROOT:-/opt/RoboTwin}"
ASSETS="${ROBOTWIN_ASSETS_DIR:-/models/robotwin_assets}"
mkdir -p "$ASSETS/assets" "$ASSETS/task_config"

if [ ! -f "$ASSETS/.ready" ]; then
  echo "== fetch task_config (@ ${RT_COMMIT:0:8}) =="
  for f in _camera_config.yml _config_template.yml _embodiment_config.yml _eval_step_limit.yml \
           demo_clean.yml demo_randomized.yml create_task_config.sh; do
    curl -fsSL "$RAW/task_config/$f" -o "$ASSETS/task_config/$f" && echo "  got $f"
  done

  echo "== fetch + extract embodiments/objects (TianxingChen/RoboTwin2.0) =="
  python - "$ASSETS/assets" <<'PY'
import sys, zipfile, os
from huggingface_hub import snapshot_download
dst = sys.argv[1]
snapshot_download(repo_id="TianxingChen/RoboTwin2.0", repo_type="dataset",
                  allow_patterns=["embodiments.zip", "objects.zip"],
                  local_dir=dst, resume_download=True)
for z in ("embodiments.zip", "objects.zip"):
    p = os.path.join(dst, z)
    print("extract", z); zipfile.ZipFile(p).extractall(dst); os.remove(p)
PY
  touch "$ASSETS/.ready"
fi

echo "== wire assets/configs into the de-vendored RoboTwin at $REPO_RT =="
rm -rf "$REPO_RT/assets" "$REPO_RT/task_config"
ln -sfn "$ASSETS/assets" "$REPO_RT/assets"
ln -sfn "$ASSETS/task_config" "$REPO_RT/task_config"

# aloha-agilex uses curobo by default; route to mplib on ROCm (asset-level config).
CFG="$REPO_RT/assets/embodiments/aloha-agilex/config.yml"
if [ -f "$CFG" ] && grep -q 'planner: "curobo"' "$CFG"; then
  sed -i 's/planner: "curobo"/planner: "mplib_RRT"/' "$CFG" && echo "  aloha-agilex -> mplib_RRT"
fi
( cd "$REPO_RT" && python ./script/update_embodiment_config_path.py >/dev/null 2>&1 || true )

echo "PASS: RoboTwin assets ready under $ASSETS"
