#!/bin/bash
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# One-shot single-machine smoke test for the Flower federated-learning
# Ryzers. Each long-running component (SuperLink, ServerApp superexec,
# and one SuperNode + ClientApp superexec per partition) is launched in
# its own terminal window so you can watch them individually. Once the
# federation is up, the quickstart-pytorch run is submitted from this
# terminal.
#
# Multi-machine deployment uses the same three Ryzers — point
# SUPERLINK_IP at the server and run the appropriate role on each box.
# See each Ryzer's README for the distributed flow.
#
# Usage:
#   cd packages/federated
#   ./run-local.sh                          # 2 partitions, auto-pick terminal
#   FLOWER_NUM_PARTITIONS=4 ./run-local.sh  # 4 partitions
#   RYZERS_TERMINAL=gnome-terminal ./run-local.sh   # force a terminal emulator

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
NUM_PARTITIONS="${FLOWER_NUM_PARTITIONS:-2}"

cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Pick a terminal emulator. xterm is preferred because it honours $DISPLAY
# reliably over SSH X-forwarding; the others are tried as fallbacks.
# ---------------------------------------------------------------------------
if [ -z "${DISPLAY}" ]; then
  echo "ERROR: \$DISPLAY is not set — can't open terminal windows." >&2
  echo "       Run over X-forwarding (ssh -X) or set DISPLAY." >&2
  exit 1
fi

TERM_EMU="${RYZERS_TERMINAL:-}"
if [ -z "${TERM_EMU}" ]; then
  for t in xterm gnome-terminal xfce4-terminal konsole x-terminal-emulator; do
    if command -v "${t}" >/dev/null 2>&1; then
      TERM_EMU="${t}"
      break
    fi
  done
fi
if [ -z "${TERM_EMU}" ] || ! command -v "${TERM_EMU}" >/dev/null 2>&1; then
  echo "ERROR: no terminal emulator found (tried xterm, gnome-terminal, ...)." >&2
  echo "       Install one or set RYZERS_TERMINAL." >&2
  exit 1
fi
echo "Using terminal emulator: ${TERM_EMU}"

LOG_DIR="$(mktemp -d -t flower-local-XXXXXX)"
echo "Per-component launch scripts: ${LOG_DIR}"

# spawn TITLE "shell-command"
#   Writes a small wrapper script (so we avoid cross-emulator quoting
#   hell), then opens it in a new terminal window that stays open after
#   the component exits.
spawn() {
  local title="$1"
  local body="$2"
  local script="${LOG_DIR}/${title}.sh"
  cat > "${script}" <<EOF
#!/bin/bash
cd "${REPO_ROOT}"
echo "### ${title}"
${body}
status=\$?
echo
echo "[${title} exited (status \${status}) — press Enter to close]"
read -r
EOF
  chmod +x "${script}"

  case "${TERM_EMU}" in
    xterm|x-terminal-emulator)
      "${TERM_EMU}" -T "${title}" -e bash "${script}" &
      ;;
    gnome-terminal)
      gnome-terminal --title="${title}" -- bash "${script}" &
      ;;
    xfce4-terminal)
      xfce4-terminal --title="${title}" --command="bash ${script}" &
      ;;
    konsole)
      konsole -p tabtitle="${title}" -e bash "${script}" &
      ;;
    *)
      "${TERM_EMU}" -e bash "${script}" &
      ;;
  esac
}

wait_for_port() {
  local port="$1"
  local timeout="${2:-120}"
  local i
  for ((i = 1; i <= timeout; i++)); do
    if (echo > /dev/tcp/127.0.0.1/"${port}") 2>/dev/null; then
      return 0
    fi
    if ((i % 10 == 0)); then
      echo "  ... still waiting for 127.0.0.1:${port} (${i}/${timeout}s)"
    fi
    sleep 1
  done
  return 1
}

# ---------------------------------------------------------------------------
# Clean up stale containers FIRST, before building. A rebuild moves the
# flower-* tags onto new image IDs and orphans the old ones, so an
# `--filter ancestor=<tag>` cleanup run *after* the build would no longer
# match containers from the previous run (they still reference the old
# image ID) and they'd keep holding the 909x ports.
#
# Primary match is the ryzers-flower-local label (set via each role's
# docker_extra_run_flags) which survives rebuilds. The ancestor/ryzerdocker
# pass is a fallback for containers created before the label existed.
# ---------------------------------------------------------------------------
echo "== Cleaning up stale containers =="
docker ps -aq --filter "label=ryzers-flower-local=1" | xargs -r docker rm -f >/dev/null 2>&1 || true
for name in flower-superlink flower-supernode flower-superexec ryzerdocker; do
  docker ps -aq --filter "ancestor=${name}" | xargs -r docker rm -f >/dev/null 2>&1 || true
done

# ---------------------------------------------------------------------------
# Build the three role images. NB: `ryzers build` names the *final* image
# after --name (default "ryzerdocker"), NOT after the last package — so
# --name is required here, otherwise all three would clobber the same
# "ryzerdocker" image and `ryzers run --name <role>` would not find its
# generated run-script.
# ---------------------------------------------------------------------------
echo "== Building Ryzers =="
ryzers build --name flower-superlink flower-base flower-superlink
ryzers build --name flower-supernode flower-base flower-supernode
ryzers build --name flower-superexec flower-base flower-superexec

# Start from a clean SuperLink state. A partial run left over from a
# previous (e.g. failed) submit can make the SuperLink raise
# KeyError('config') -> "Exception calling application: 'config'" when a
# SuperNode connects. This matches the volume mount in
# flower-superlink/config.yaml ($PWD/workspace/flower/state).
echo "== Resetting SuperLink state =="
rm -rf "${REPO_ROOT}/workspace/flower/state"/* 2>/dev/null || true

echo "== Starting SuperLink =="
spawn "flower-superlink" "ryzers run --name flower-superlink"
if ! wait_for_port 9092 120; then
  echo "ERROR: SuperLink Fleet API (9092) never came up." >&2
  echo "       Check the 'flower-superlink' terminal window for the error." >&2
  exit 1
fi
echo "  SuperLink up (9092 bound)."

echo "== Starting ServerApp superexec =="
spawn "flower-serverapp" \
  "FLOWER_PLUGIN_TYPE=serverapp FLOWER_APPIO_ADDR=127.0.0.1:9091 ryzers run --name flower-superexec"

for ((i = 0; i < NUM_PARTITIONS; i++)); do
  echo "== Starting SuperNode partition ${i} =="
  spawn "flower-supernode-${i}" \
    "FLOWER_PARTITION_ID=${i} FLOWER_NUM_PARTITIONS=${NUM_PARTITIONS} ryzers run --name flower-supernode"

  echo "== Starting ClientApp superexec partition ${i} =="
  spawn "flower-clientapp-${i}" \
    "FLOWER_PLUGIN_TYPE=clientapp FLOWER_PARTITION_ID=${i} FLOWER_APPIO_ADDR=127.0.0.1:$((9094 + i)) ryzers run --name flower-superexec"
done

echo "== Waiting for SuperLink ExecApi (9093) =="
if ! wait_for_port 9093 120; then
  echo "ERROR: SuperLink ExecApi (9093) never came up." >&2
  exit 1
fi

# Give the SuperNodes a beat to finish the Fleet handshake before
# submitting — otherwise `flwr run` can race node registration.
echo "  Waiting for SuperNodes to register..."
sleep 8

echo "== Submitting quickstart-pytorch run =="
FLOWER_PLUGIN_TYPE=submit ryzers run --name flower-superexec

echo
echo "== Run submitted. Component windows are still open. =="
echo "To stop everything:"
echo "  for n in flower-superlink flower-supernode flower-superexec; do \\"
echo "    docker ps -q --filter ancestor=\$n | xargs -r docker kill; done"
