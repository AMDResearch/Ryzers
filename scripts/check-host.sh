#!/usr/bin/env bash
# =============================================================================
# check-host.sh — preflight host checks for building and running Ryzers packages.
#
# Verifies the things that commonly fail *before* a long `ryzers build`:
# GPU device nodes, user groups, container runtime, disk and memory.
#
# Usage:  bash scripts/check-host.sh
#
# Output: one [OK] / [WARN] / [FAIL] line per check.
#         Exits 1 if any [FAIL] was reported, 0 otherwise.
#         [WARN] items are worth fixing but do not block the build.
# =============================================================================
set -u

fails=0
warns=0
ok()   { printf '[OK]   %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; warns=$((warns + 1)); }
fail() { printf '[FAIL] %s\n' "$*"; fails=$((fails + 1)); }

echo "=== Ryzers host preflight ==="

# --- OS -------------------------------------------------------------------
# All packages are tested on Ubuntu 24.04 / Strix-Halo; other releases may work.
if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    if [ "${ID:-}" = "ubuntu" ] && [ "${VERSION_ID:-}" = "24.04" ]; then
        ok "OS: Ubuntu 24.04 (${PRETTY_NAME:-})"
    else
        warn "OS: ${PRETTY_NAME:-unknown} — tested target is Ubuntu 24.04"
    fi
else
    warn "OS: /etc/os-release not readable — cannot verify distro"
fi

# --- WSL2 -----------------------------------------------------------------
# Ryzers is tested on native Linux; under WSL2 the GPU is exposed differently.
if grep -qi microsoft /proc/version 2>/dev/null; then
    warn "WSL2 detected — Ryzers is tested on native Linux; GPU passthrough uses /dev/dxg"
fi

# --- GPU device nodes -----------------------------------------------------
if [ -e /dev/kfd ]; then
    ok "GPU compute node /dev/kfd present"
else
    if [ -e /dev/dxg ]; then
        warn "GPU: /dev/kfd missing but /dev/dxg present (WSL2 GPU passthrough)"
    else
        fail "GPU: /dev/kfd missing — AMD ROCm workloads cannot run (install/enable the AMD GPU driver)"
    fi
fi

if [ -e /dev/dri/renderD128 ]; then
    ok "GPU render node /dev/dri/renderD128 present"
else
    fail "GPU: /dev/dri/renderD128 missing — container GPU rendering will fail"
fi

# --- User groups ----------------------------------------------------------
if [ "$(id -u)" -eq 0 ]; then
    warn "running as root — Ryzers packages are normally built and run as a regular user"
else
    for grp in render video; do
        if id -nG 2>/dev/null | tr ' ' '\n' | grep -qx "$grp"; then
            ok "user group '$grp' membership active"
        elif getent group "$grp" >/dev/null 2>&1; then
            fail "user not in group '$grp' — run: sudo usermod -aG $grp \$USER (then re-login)"
        else
            warn "group '$grp' does not exist on this host"
        fi
    done
fi

# --- Container runtime ----------------------------------------------------
# Docker is the documented requirement; rootless Podman is a drop-in for
# building (see docs/installation.md).
if command -v docker >/dev/null 2>&1; then
    if docker info >/dev/null 2>&1; then
        ok "container runtime: docker ($(docker --version 2>/dev/null | head -n1))"
    else
        fail "docker found but the daemon is unreachable (start it, or fix permissions / use rootless podman)"
    fi
elif command -v podman >/dev/null 2>&1; then
    ok "container runtime: podman ($(podman --version 2>/dev/null))"
    warn "docker not found — podman can substitute; use fully-qualified image names (docker.io/...) or declare docker.io in registries.conf"
else
    fail "no container runtime — install Docker (docs/installation.md) or Podman"
fi

# --- Resources ------------------------------------------------------------
# A single ROCm base image is ~20 GB; builds add more layers on top.
avail_kb=$(df -Pk "$HOME" 2>/dev/null | awk 'NR==2 {print $4}')
if [ -n "${avail_kb:-}" ]; then
    avail_gb=$((avail_kb / 1024 / 1024))
    if [ "$avail_gb" -ge 100 ]; then
        ok "disk: ${avail_gb} GB free"
    elif [ "$avail_gb" -ge 30 ]; then
        warn "disk: only ${avail_gb} GB free — image builds may exhaust space (100 GB recommended)"
    else
        fail "disk: ${avail_gb} GB free — at least 30 GB required for base images"
    fi
fi

mem_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null)
if [ -n "${mem_kb:-}" ]; then
    mem_gb=$((mem_kb / 1024 / 1024))
    if [ "$mem_gb" -ge 16 ]; then
        ok "memory: ${mem_gb} GB"
    elif [ "$mem_gb" -ge 8 ]; then
        warn "memory: ${mem_gb} GB — heavy packages (e.g. O3DE-based) may struggle"
    else
        fail "memory: ${mem_gb} GB — at least 8 GB required"
    fi
fi

echo "=== result: $fails failure(s), $warns warning(s) ==="
[ "$fails" -eq 0 ] || exit 1
exit 0
