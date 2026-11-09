#!/usr/bin/env bash
#
# link_rocm_tree.sh: make the pip-installed ROCm 7.13 SDK usable by the Genesis AMDGPU backend.
#
# Usage (from the workshop folder, after torch and gslab are installed into .venv):
#
#   ./link_rocm_tree.sh                          # default: .venv
#   VENV=/path/to/venv ./link_rocm_tree.sh       # another virtualenv
#   LLD=/usr/bin/ld.lld-20 ./link_rocm_tree.sh   # choose the linker explicitly
#
# Idempotent; re-run after reinstalling the wheels.
#
# Background
#
#   PyTorch finds the HIP runtime through the rocm_sdk Python package and just works.
#   Genesis compiles its physics kernels at runtime with quadrants, which expects a
#   classic /opt/rocm layout and needs two things the pip SDK does not provide:
#
#     1. dlopen("libamdhip64.so") by bare name. Wheels cannot carry symlinks, so the
#        SDK ships only libamdhip64.so.7; the unversioned alias is missing.
#     2. ${ROCM_PATH}/llvm/bin/ld.lld to link its GPU code objects. The pip SDK has no
#        linker (only the 1+ GB rocm[devel] extra does), and the objects use AMDGPU
#        code object v6, which lld older than LLVM 19 rejects ("unknown abi version").
#
# What this script produces
#
#   <venv>/lib/python3.x/site-packages/_rocm_sdk_*/lib/lib*.so   unversioned aliases
#   <venv>/rocm/{lib,lib-<gfx>,bin,share}  ->  the pip SDK directories
#   <venv>/rocm/llvm/bin/ld.lld            ->  a system ld.lld of LLVM 19 or newer
#   <venv>/rocm/env.sh                          exports ROCM_PATH and LD_LIBRARY_PATH
#
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

VENV="$(readlink -f "${VENV:-$PWD/.venv}")"
PYTHON="${VENV}/bin/python"
ROCM_TREE="${VENV}/rocm"

# Set by the steps below.
SITE=""      # site-packages of the venv
CORE=""      # ${SITE}/_rocm_sdk_core            (HIP runtime, tools)
LIBS=""      # ${SITE}/_rocm_sdk_libraries_<gfx>  (math libraries for one GPU target)
TARGET=""    # <gfx>, e.g. gfx1151
LIB_PATH=""  # value for LD_LIBRARY_PATH

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }
indent() { sed 's/^/  /'; }

# --------------------------------------------------------------------------------------
# 1. Find the pip ROCm SDK inside the venv.
# --------------------------------------------------------------------------------------
locate_sdk() {
    step "Locate the pip-installed ROCm SDK in ${VENV}"
    [ -x "$PYTHON" ] || die "no Python in ${VENV}. Create the virtualenv and install torch first (INSTALL2.md, step 2)."

    SITE="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    CORE="${SITE}/_rocm_sdk_core"
    [ -d "${CORE}/lib" ] || die "${CORE}/lib not found. Install torch from the ROCm wheel index first; it pulls in rocm-sdk-core."

    # There is one rocm-sdk-libraries-<gfx> package per GPU family; take the installed one.
    for d in "${SITE}"/_rocm_sdk_libraries_*; do
        [ -d "${d}/lib" ] && LIBS="$d" && break
    done
    [ -n "$LIBS" ] || die "no _rocm_sdk_libraries_<gfx> package in ${SITE}."
    TARGET="${LIBS##*_rocm_sdk_libraries_}"

    echo "core:      ${CORE}"
    echo "libraries: ${LIBS} (target ${TARGET})"
}

# --------------------------------------------------------------------------------------
# 2. Add libfoo.so next to every libfoo.so.N so dlopen() by bare name works.
# --------------------------------------------------------------------------------------
add_so_aliases() {
    step "Unversioned .so aliases for dlopen()"
    local so base added=0
    for so in "${CORE}"/lib/lib*.so.[0-9]* "${LIBS}"/lib/lib*.so.[0-9]*; do
        base="${so%%.so.*}.so"
        [ -e "$base" ] && continue
        ln -s "$(basename "$so")" "$base"
        added=$((added + 1))
    done
    echo "added ${added} aliases (0 on a re-run is expected)"
    [ -e "${CORE}/lib/libamdhip64.so" ] || die "libamdhip64.so alias missing in ${CORE}/lib"
}

# --------------------------------------------------------------------------------------
# 3. Assemble <venv>/rocm, an /opt/rocm look-alike pointing at the pip SDK.
# --------------------------------------------------------------------------------------
link_tree() {
    step "ROCm tree at ${ROCM_TREE}"
    mkdir -p "${ROCM_TREE}/llvm/bin"
    ln -sfn "${CORE}/lib" "${ROCM_TREE}/lib"
    ln -sfn "${LIBS}/lib" "${ROCM_TREE}/lib-${TARGET}"
    ln -sfn "${CORE}/bin" "${ROCM_TREE}/bin"
    [ -d "${CORE}/share" ] && ln -sfn "${CORE}/share" "${ROCM_TREE}/share"
    LIB_PATH="${ROCM_TREE}/lib:${ROCM_TREE}/lib-${TARGET}"
    ls -l "${ROCM_TREE}" | indent
}

# --------------------------------------------------------------------------------------
# 4. Link an ld.lld of LLVM 19 or newer into the tree.
# --------------------------------------------------------------------------------------
lld_major_version() {
    "$1" --version 2>/dev/null | grep -oE 'LLD [0-9]+' | grep -oE '[0-9]+' | head -n1 || true
}

# Candidates in order of preference: explicit $LLD, the system ROCm, lld-2x/lld-19 from
# apt.llvm.org, then whatever is on PATH. Globs that match nothing are dropped.
lld_candidates() {
    local c
    for c in ${LLD:-} /opt/rocm/llvm/bin/ld.lld /opt/rocm-*/llvm/bin/ld.lld \
             /usr/bin/ld.lld-2? /usr/bin/ld.lld-19 /usr/bin/ld.lld "$(command -v ld.lld || true)"; do
        [ -n "$c" ] && [ -x "$c" ] && echo "$c"
    done
}

find_lld() {
    local c major
    for c in $(lld_candidates); do
        major="$(lld_major_version "$c")"
        if [ -n "$major" ] && [ "$major" -ge 19 ]; then
            echo "$c"
            return 0
        fi
        echo "skip $c: $("$c" --version 2>/dev/null | head -n1) (need LLD >= 19)" >&2
    done
    return 1
}

link_lld() {
    step "ld.lld of LLVM 19 or newer"
    local lld
    if ! lld="$(find_lld)"; then
        cat >&2 <<'MSG'
ERROR: no ld.lld >= 19 found.
  Either install the system ROCm (provides /opt/rocm/llvm/bin/ld.lld), or install lld-20
  from https://apt.llvm.org (sudo apt install lld-20), then re-run this script.
  Ubuntu 24.04's stock lld-18 is too old: it rejects the code objects Genesis emits.
MSG
        exit 1
    fi
    ln -sfn "$lld" "${ROCM_TREE}/llvm/bin/ld.lld"
    echo "ld.lld -> $lld"
    echo "         $("$lld" --version | head -n1)"
}

# --------------------------------------------------------------------------------------
# 5. Write env.sh with the variables every Genesis process needs.
# --------------------------------------------------------------------------------------
write_env_file() {
    step "Environment file ${ROCM_TREE}/env.sh"
    cat > "${ROCM_TREE}/env.sh" <<ENV
# Generated by link_rocm_tree.sh. Source this before running Genesis outside Jupyter:
#   source ${VENV}/bin/activate && source ${ROCM_TREE}/env.sh
# quadrants (Genesis' compiler) runs \${ROCM_PATH}/llvm/bin/ld.lld and dlopens libamdhip64.so.
export ROCM_PATH="${ROCM_TREE}"
export LD_LIBRARY_PATH="${LIB_PATH}\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
# Raise the compiled-kernel cache limit above its default so kernels persist between runs.
export QD_OFFLINE_CACHE_MAX_SIZE_OF_FILES=2147483647
ENV
    indent < "${ROCM_TREE}/env.sh"
}

# --------------------------------------------------------------------------------------
# 6. Check that the two things quadrants needs actually work.
# --------------------------------------------------------------------------------------
verify() {
    step "Verify"
    "${ROCM_TREE}/llvm/bin/ld.lld" --version >/dev/null || die "ld.lld does not run"
    LD_LIBRARY_PATH="${LIB_PATH}" "$PYTHON" -c 'import ctypes; ctypes.CDLL("libamdhip64.so")' \
        || die "dlopen(libamdhip64.so) failed with LD_LIBRARY_PATH=${LIB_PATH}"
    echo "ld.lld runs; libamdhip64.so loads by bare name."
}

print_summary() {
    cat <<SUMMARY

Done. Genesis needs these two variables in every process that uses it:

  ROCM_PATH=${ROCM_TREE}
  LD_LIBRARY_PATH=${LIB_PATH}

  * Shell:   source ${ROCM_TREE}/env.sh
  * Jupyter: bake them into the kernel (INSTALL2.md step 4, or install_local.sh):
      python -m ipykernel install --user --name gslab-roscon \\
          --display-name "gslab ROSCon (ROCm 7.13)" \\
          --env ROCM_PATH "${ROCM_TREE}" \\
          --env LD_LIBRARY_PATH "${LIB_PATH}" \\
          --env QD_OFFLINE_CACHE_MAX_SIZE_OF_FILES 2147483647
SUMMARY
}

main() {
    locate_sdk
    add_so_aliases
    link_tree
    link_lld
    write_env_file
    verify
    print_summary
}

main "$@"
