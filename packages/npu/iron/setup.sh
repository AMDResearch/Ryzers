# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

NEW_CMAKE_DIR="/ryzers/mlir-aie/ironenv/bin/cmake"
export MLIR_AIE_BUILD_DIR="/ryzers/mlir-aie"

source /opt/xilinx/xrt/setup.sh
export PATH="${NEW_CMAKE_DIR}/bin":"${PATH}"

source ${MLIR_AIE_BUILD_DIR}/ironenv/bin/activate

export MLIR_AIE_INSTALL_DIR="$(pip show mlir_aie | grep ^Location: | awk '{print $2}')/mlir_aie"
export PEANO_INSTALL_DIR="$(pip show llvm-aie | grep ^Location: | awk '{print $2}')/llvm-aie"
source ${MLIR_AIE_BUILD_DIR}/utils/env_setup.sh $MLIR_AIE_INSTALL_DIR $PEANO_INSTALL_DIR

export AIE_API_DIR="${MLIR_AIE_BUILD_DIR}/third_party/aie_api"

export WARNING_FLAGS="-Wno-parentheses -Wno-attributes -Wno-macro-redefined"
export PEANOWRAP2_FLAGS="-O2 -std=c++20 --target=aie2-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${MLIR_AIE_INSTALL_DIR}/include " 
export PEANOWRAP2P_FLAGS="-O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${MLIR_AIE_INSTALL_DIR}/include "

examine_output=$(xrt-smi examine)
if echo $examine_output | grep -E "Phoenix|Hawk" > /dev/null; then
    export NPU=npu1
elif echo $examine_output | grep -E "Strix|Krackan" > /dev/null; then
    export NPU=npu2
else
    echo "No recognized NPU found"
fi