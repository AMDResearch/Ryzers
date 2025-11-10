#!/bin/bash
#
# Build script for CVML samples
#

set -e  # Exit on error

echo "Building CVML samples..."

# Create build directories
BUILD_DIR="build"
OUTPUT_DIR="../build"
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
mkdir -p $OUTPUT_DIR

# Run cmake
echo "Running CMake..."
cmake -S $PWD -B $PWD/$BUILD_DIR

# Build
echo "Building executables..."
cmake --build $PWD/$BUILD_DIR --config Release

echo ""
echo "Build complete!"
echo ""
echo "Executables created in notebooks/build/:"
echo "  - depth_estimation_npu"
echo "  - depth_estimation_cpu"
echo "  - face_detection_npu"
echo "  - face_detection_cpu"
echo ""
echo "Usage examples (from notebooks/ directory):"
echo "  ./build/depth_estimation_npu -i Ryzen-AI-CVML-Library/samples/video_call.mp4 -o output_npu.mp4"
echo "  ./build/depth_estimation_cpu -i Ryzen-AI-CVML-Library/samples/video_call.mp4 -o output_cpu.mp4"
echo "  ./build/face_detection_npu -i Ryzen-AI-CVML-Library/samples/video_call.mp4 -m fast -o face_npu.mp4"
echo "  ./build/face_detection_cpu -i Ryzen-AI-CVML-Library/samples/video_call.mp4 -m fast -o face_cpu.mp4"
echo ""
