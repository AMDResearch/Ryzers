# ROCm Upgrade Reference

These are lessons from the Ryzers ROCm 7.2.2 / PyTorch 2.10 to ROCm 7.14 / PyTorch 2.12 migration. Revalidate them against the next target image; they are examples, not permanent version rules.

## Wheel runtime and HIP development tools

The ROCm 7.14 PyTorch image used wheel-packaged ROCm. Runtime libraries were present, but downstream HIP builds needed matching `rocm[devel]` components.

The working sequence was:

1. Read the installed `rocm` distribution version.
2. Install that exact `rocm[devel]` version from AMD's wheel repository.
3. Run `rocm-sdk init` during the build.
4. Expose the SDK root through `/opt/rocm`.

Pre-initialization mattered because parallel native builds could otherwise race while lazy SDK content was unpacked.

## Paths and shared libraries

Most native packages worked with:

- HIP root: `/opt/rocm`
- HIP compiler: `/opt/rocm/lib/llvm/bin/clang`
- ROCm CMake prefix: `/opt/rocm/lib/cmake`

OpenSplat additionally needed wheel directories containing `host-math` and `rocm_sysdeps`. This was established from missing-library errors, not added speculatively.

Do not copy Python 3.12 paths into a future upgrade without checking the new image.

## Gfx overrides

`HSA_OVERRIDE_GFX_VERSION=11.0.0` was removed when the new stack supported `gfx1151` directly. Native builds still needed explicit architecture flags because GPU detection is unavailable during ordinary Docker builds.

Removing an override is valid only after:

- The runtime identifies the real GPU architecture
- Kernels compile for that architecture
- The package runs without the override

## Torch, torchvision, and Triton

Pure Python packages such as MobileSAM accepted the ROCm wheels inherited from the base. Reinstalling torch was unnecessary and would have downgraded them to the old ROCm nightly index.

RAI dependencies could pull CUDA torch, so its final layer restored matching ROCm wheels. The selected torch wheel declared its matching ROCm Triton dependency; the old explicit uninstall/install swap was removed after verifying:

- Triton distribution and version
- Triton module path
- `torch._inductor` import

Always inspect final package state, not just the base image.

## Native package examples

### llama.cpp

The build moved to the `/opt/rocm` compiler and CMake paths. Its test moved from a large 8B model to a 1B model for faster verification.

Downstream SmolVLM also needed a test update because current llama.cpp no longer built `llama-run`. `llama-cli` used bounded generation and noninteractive options.

### GR00T

The shared `rocm[devel]` installation made apt `rocm-dev` redundant. Removing it avoided mixing an apt toolchain with the wheel runtime.

### NCNN

The old ROCm base contained `libopencv-dev`; the new base did not. NCNN's image examples require either OpenCV or `NCNN_SIMPLEOCV=ON`. Enabling SimpleOCV preserved the existing `squeezenet` test without installing full OpenCV.

### OpenSplat

PyTorch 2.10's HIP header declared `c10::hip::HIPCachingAllocator`. The PyTorch 2.12 ROCm wheel declared the allocator under `c10::cuda::CUDACachingAllocator`. The downstream patch was required for the new wheel, but an upstream solution should be version-compatible. Check [OpenSplat issue #228](https://github.com/pierotofy/OpenSplat/issues/228) before carrying the patch forward.

The test dataset was moved to `/opt/opensplat-test-data` because the configured `/ryzers/data` mount hid files baked into that directory. Output was explicitly written back to `/ryzers/data` for persistence.

### OpenPI and MolmoAct2

Strict torch and torchvision constraints required minimal patches for the newer compatible versions. OpenPI also needed uv to use the existing `/opt/venv`, the AMD wheel index, and the actual Python version.

Review lockfiles, uv source selection, JAX extras, and dependency overrides together. A package manager can silently replace the correct ROCm stack even when the Docker base is correct.

## Runtime and test examples

### Ultralytics

The new base lacked the GL library required by `opencv-python`, so `libgl1` was installed. The model and image were placed in `/opt/ultralytics-test-data`; `/ryzers/data_ultralytics` remained a mount for output and user data.

### ACT

Newer MuJoCo defaulted to a GLFW path that required X11. The automated test set `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl` locally instead of changing every container session.

### LM Studio

The unpinned Electron build needed NSS, GTK, audio, accessibility, and related runtime libraries. Automated execution required `xvfb-run`; an interactive GUI still requires a graphical host session.

### Gazebo

The top-level `gz --version` command failed. `gz sim --version` checked the installed simulator directly. Avoid fallbacks that convert a failed version check into a passing test.

### LeRobot

The release bump changed import locations and CLI behavior. Tests were updated to the release's canonical APIs, required a ROCm-enabled torch build and accessible GPU, and disabled pretrained backbone downloads. Documentation used the release's rollout command and current option names.

## Reproducibility checks

For every package:

```text
base image
  -> ryzer_env
  -> package dependency installation
  -> final torch/ROCm inspection
  -> package build
  -> default smoke test
  -> normal ryzers run configuration
```

Check the final stage for:

- CUDA or CPU torch replacing ROCm torch
- Hidden test assets caused by bind mounts
- Runtime network downloads
- Interactive hangs
- Tests that report success after the real command failed
- Generated files lost when an `--rm` container exits
- README commands that assume the wrong working directory

## Performance validation

The migration compared matched old and new images across more than one VLA/VLM workload. Results included both gains and regressions. Preserve this discipline:

- Keep workload and model revisions fixed
- Cache weights outside disposable containers
- Warm up before timing
- Synchronize GPU operations
- Capture memory and throughput as well as latency
- Do not claim a general ROCm improvement from a single favorable model
