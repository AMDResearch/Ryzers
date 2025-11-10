# CMakeLists.txt for CVML samples with prebuilt NPU and CPU backends

cmake_minimum_required(VERSION 3.18)
project(cvml-samples)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED)

# Set output directory to parent notebooks/build/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../build)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/../build)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/../build)

# Set CVML library root
set(AMD_CVML_SDK_ROOT /ryzers/RyzenAI-SW/Ryzen-AI-CVML-Library)

# Find OpenCV
file(TO_CMAKE_PATH "$ENV{OPENCV_INSTALL_ROOT}" OPENCV_INSTALL_ROOT)
find_package(OpenCV PATHS ${OPENCV_INSTALL_ROOT})
if (NOT OpenCV_FOUND)
    set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
    message(STATUS "First attempt to find OpenCV failed. Trying again with OpenCV_DIR=${OpenCV_DIR}")
    find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR})
endif()

# Find Ryzen AI CVML library
find_package(RyzenAILibrary REQUIRED PATHS ${AMD_CVML_SDK_ROOT})

set(CMAKE_CXX_FLAGS "-D_UNICODE -DUNICODE")

# Include directories
include_directories(
    ${AMD_CVML_SDK_ROOT}/include
    ${AMD_CVML_SDK_ROOT}/samples/common-sample-utils/include
    ${OpenCV_INCLUDE_DIRS}
)

# Link directories
link_directories(
    ${AMD_CVML_SDK_ROOT}/linux/onnx/ryzen14
    ${AMD_CVML_SDK_ROOT}/linux/onnx/cpu
    ${AMD_CVML_SDK_ROOT}/linux
)

# Common sample utils library
set(COMMON_UTILS_LIB ${AMD_CVML_SDK_ROOT}/samples/build-samples/common-sample-utils/src/libcommon-sample-utils.a)

# Depth Estimation - NPU backend
add_executable(depth_estimation_npu depth_estimation_npu.cpp)
target_link_libraries(depth_estimation_npu
    ${COMMON_UTILS_LIB}
    ${RyzenAILibrary_LIBS}
    ${OpenCV_LIBS}
)

# Depth Estimation - CPU backend
add_executable(depth_estimation_cpu depth_estimation_cpu.cpp)
target_link_libraries(depth_estimation_cpu
    ${COMMON_UTILS_LIB}
    ${RyzenAILibrary_LIBS}
    ${OpenCV_LIBS}
)

# Face Detection - NPU backend
add_executable(face_detection_npu face_detection_npu.cpp)
target_link_libraries(face_detection_npu
    ${COMMON_UTILS_LIB}
    ${RyzenAILibrary_LIBS}
    ${OpenCV_LIBS}
)

# Face Detection - CPU backend
add_executable(face_detection_cpu face_detection_cpu.cpp)
target_link_libraries(face_detection_cpu
    ${COMMON_UTILS_LIB}
    ${RyzenAILibrary_LIBS}
    ${OpenCV_LIBS}
)

# Set RPATH for runtime library finding
set_target_properties(
    depth_estimation_npu depth_estimation_cpu
    face_detection_npu face_detection_cpu
    PROPERTIES
    BUILD_RPATH "${AMD_CVML_SDK_ROOT}/linux:${AMD_CVML_SDK_ROOT}/linux/onnx/ryzen14:${AMD_CVML_SDK_ROOT}/linux/onnx/cpu"
    INSTALL_RPATH "${AMD_CVML_SDK_ROOT}/linux:${AMD_CVML_SDK_ROOT}/linux/onnx/ryzen14:${AMD_CVML_SDK_ROOT}/linux/onnx/cpu"
)
