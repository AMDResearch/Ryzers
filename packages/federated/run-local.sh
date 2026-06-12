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
# If the orchestrator dropped the shutdown sentinel, this component was
# stopped as part of a normal end-of-run teardown — exit immediately so the
# terminal window closes on its own. Otherwise (e.g. an early crash) stay
# open so the error stays readable.
if [ -f "${LOG_DIR}/.shutdown" ]; then
  echo "[${title} stopped for shutdown (status \${status}) — closing window]"
  exit \${status}
fi
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

port_in_use() {
  # 0 (true) if something is already listening on 127.0.0.1:<port>.
  local port="$1"
  (echo > /dev/tcp/127.0.0.1/"${port}") 2>/dev/null
}

# require_ports_free PORT...
#   Abort if any of the given ports already has a listener. Because gRPC
#   uses SO_REUSEPORT, a leftover listener would be silently co-bound to by
#   the component we are about to start — the exact failure mode this script
#   guards against — so we fail fast with diagnostics instead of starting on
#   top of it. `wait_for_port` cannot catch this: it treats a stale listener
#   as "the service is up".
require_ports_free() {
  local p busy=()
  for p in "$@"; do
    if port_in_use "${p}"; then
      busy+=("${p}")
    fi
  done
  if ((${#busy[@]} > 0)); then
    echo "ERROR: these ports are still in use after cleanup: ${busy[*]}" >&2
    echo "       A stale listener here would be silently co-bound via" >&2
    echo "       SO_REUSEPORT and answer some calls with errors like" >&2
    echo "       \"Exception calling application: 'config'\"." >&2
    echo >&2
    echo "       Find and remove what is holding them, e.g.:" >&2
    echo "         docker ps -a --format '{{.ID}}  {{.Image}}  {{.Names}}'" >&2
    for p in "${busy[@]}"; do
      echo "         ss -tlnp 'sport = :${p}'   # (or: lsof -iTCP:${p} -sTCP:LISTEN)" >&2
    done
    exit 1
  fi
}

# ---------------------------------------------------------------------------
# Clean up stale containers FIRST, before building. A rebuild moves the
# flower-* tags onto new image IDs and orphans the old ones, so an
# `--filter ancestor=<tag>` cleanup run *after* the build would no longer
# match containers from the previous run (they still reference the old
# image ID) and they'd keep holding the 909x ports.
#
# This MUST be thorough. Every component runs with `--network host`, and
# gRPC enables SO_REUSEPORT by default (flwr does not disable it), so a
# stale SuperLink/SuperNode from a previous run can silently CO-BIND 909x
# alongside the freshly started one. The kernel then load-balances
# connections across both, and the older / half-broken instance answers
# some calls with cryptic gRPC errors such as
#   "Exception calling application: 'config'"   (a server-side KeyError).
# That is invisible in the new SuperLink's window (it starts fine), which
# makes it very hard to diagnose — so we remove flower containers by EVERY
# signal we have, not just the label.
# ---------------------------------------------------------------------------
echo "== Cleaning up stale containers =="
# 1) Primary: the label set via each role's docker_extra_run_flags
#    (survives image rebuilds).
docker ps -aq --filter "label=ryzers-flower-local=1" | xargs -r docker rm -f >/dev/null 2>&1 || true
# 2) Fallback: anything whose image references a flower-* tag (or the
#    default "ryzerdocker" tag) — for containers created before the label
#    existed, or built from a differently-tagged cached base.
for name in flower-superlink flower-supernode flower-superexec flower-base ryzerdocker; do
  docker ps -aq --filter "ancestor=${name}" | xargs -r docker rm -f >/dev/null 2>&1 || true
done
# 3) Last resort: any remaining container whose image name contains
#    "flower" (catches odd tags from earlier iterations). `--filter` has no
#    image wildcard, so match on the formatted list instead.
docker ps -a --format '{{.ID}} {{.Image}}' \
  | awk 'tolower($2) ~ /flower/ {print $1}' \
  | xargs -r docker rm -f >/dev/null 2>&1 || true

# ---------------------------------------------------------------------------
# Remove stale generated run-scripts BEFORE building. `ryzers run` does NOT
# regenerate `ryzers.run.<name>.sh` — it blindly `bash`-executes whatever is
# already in the cwd (runner.py). `ryzers build` overwrites the three current
# scripts, but a script left over from an earlier package layout or image
# name is never touched and can be picked up by a stray `ryzers run`,
# reintroducing the cryptic gRPC "Exception calling application: 'config'" /
# "'script'" failures. Wipe them so every run starts from freshly generated
# scripts.
# ---------------------------------------------------------------------------
echo "== Removing stale ryzers run-scripts =="
rm -f "${REPO_ROOT}"/ryzers.run.*.sh 2>/dev/null || true

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

# Confirm cleanup actually freed every host port we are about to bind.
# 9091 ServerAppIo, 9092 Fleet, 9093 ExecApi, plus one ClientAppIo per
# partition (9094 + i). If any is still held, abort before we start — a
# survivor would be co-bound via SO_REUSEPORT and intermittently serve
# stale responses.
echo "== Verifying ports are free =="
PORTS_TO_CHECK=(9091 9092 9093)
for ((i = 0; i < NUM_PARTITIONS; i++)); do
  PORTS_TO_CHECK+=($((9094 + i)))
done
require_ports_free "${PORTS_TO_CHECK[@]}"
echo "  All required ports are free: ${PORTS_TO_CHECK[*]}"

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

# The submit uses `flwr run ... --stream`, so this call blocks until the run
# finishes and the ServerApp has written final_model.pt to disk.
echo "== Submitting quickstart-pytorch run (streaming until complete) =="
FLOWER_PLUGIN_TYPE=submit ryzers run --name flower-superexec

# Run finished — tear the whole federation down so every per-component window
# closes on its own. Dropping the sentinel tells each spawned wrapper to exit
# (closing its terminal) instead of waiting on a keypress; killing the
# containers makes each wrapper's `docker run` return so it reaches that check.
echo
echo "== Run complete. Shutting down all components... =="
touch "${LOG_DIR}/.shutdown"
docker ps -q --filter "label=ryzers-flower-local=1" | xargs -r docker kill >/dev/null 2>&1 || true

echo "== Done. Model written to disk; all component windows are closing. =="
