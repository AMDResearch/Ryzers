---
name: rocm-upgrade
description: Upgrade the Ryzers default ROCm/PyTorch image and migrate Docker packages safely. Use when bumping ROCm, PyTorch ROCm wheels, gfx targets, HIP toolchains, or validating all Ryzers packages after a ROCm base-image change.
---

# ROCm Upgrade

Upgrade Ryzers without modifying the host ROCm installation, kernel, or firmware. Treat every base image as a new environment: inspect it rather than assuming the previous image layout still applies.

Read [reference.md](reference.md) for migration patterns and failures observed during the ROCm 7.2.2 to 7.14 upgrade.

## 1. Establish scope and baseline

1. Confirm the target ROCm image, GPU architecture, package exclusions, and whether old Docker images may be removed.
2. Work on a dedicated branch. Do not modify `main` or other worktrees.
3. Exclude NPU packages unless the user explicitly includes them; their XDNA stack is independent of the GPU ROCm bump.
4. Record:
   - Current default image in `ryzers/__init__.py`
   - Python, PyTorch, torchvision, torchaudio, ROCm, and HIP versions
   - `torch.version.hip`, `torch.cuda.is_available()`, and GPU name
   - Existing build and test results
5. Do not install ROCm on the host, update the kernel, or reboot.

## 2. Inspect the new base image

Before editing packages, run a temporary container and determine:

- Whether ROCm is apt-based or wheel-based
- Python and virtual-environment paths
- Installed `rocm`, torch, torchvision, torchaudio, and Triton distributions
- Locations of `hipcc`, `clang`, HIP headers, shared libraries, and CMake modules
- Whether `/opt/rocm` exists
- Whether common development packages such as OpenCV were removed
- The native GPU architecture supported by the new stack

For a wheel-based image, install development components matching the installed `rocm` distribution exactly:

```dockerfile
RUN ROCM_VERSION="$(python -c "import importlib.metadata as m; print(m.version('rocm'))" 2>/dev/null)" || exit 0; \
    pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "rocm[devel]==${ROCM_VERSION}"; \
    rocm-sdk init; \
    test -e /opt/rocm || ln -s "$(rocm-sdk path --root)" /opt/rocm
```

Keep this compatibility setup in `ryzer_env`, not in every native package. Verify the exact repository and installation mechanism again on future releases.

## 3. Inventory version-coupled code

Search the repository for:

- Old ROCm and PyTorch version strings
- Package-level `init_image` overrides
- `HSA_OVERRIDE_GFX_VERSION`
- `rocm-dev` and other apt ROCm packages
- Old PyTorch ROCm indexes and nightly indexes
- Explicit `pytorch-triton-rocm` installs
- Hardcoded `/opt/rocm-*`, Python-version, virtualenv, and CMake paths
- Torch and torchvision upper bounds
- HIP architecture flags
- Native extensions and patched upstream source
- Tests using renamed or removed commands

Classify every match:

- Remove a package `init_image` when it only duplicates the global default.
- Keep a pin only when the package intentionally requires a different base, and document why.
- Remove a gfx override only after native support is verified on the target GPU.
- Remove apt `rocm-dev` when `ryzer_env` already supplies the matching wheel development toolchain.
- Do not blindly remove old compatibility patches; reproduce the failure first.

## 4. Update the shared environment first

1. Update `RYZERS_DEFAULT_INIT_IMAGE`.
2. Update `ryzer_env` for the new runtime layout.
3. Initialize lazy SDK content during the image build. Do not let parallel downstream compiles race while unpacking it.
4. Provide a conventional `/opt/rocm` view for packages that use standard HIP and CMake paths.
5. Verify:
   - `/opt/rocm/bin/hipcc --version`
   - HIP headers and CMake packages are present
   - A minimal HIP/CMake project configures without GPU access during `docker build`
   - Runtime libraries resolve

Do not export package-specific compiler flags globally. Keep architecture and compiler choices in the package that needs them.

## 5. Migrate packages by dependency type

### Pure Python packages

Build first. Existing base wheels should remain installed when dependency constraints accept them. Remove redundant torch reinstalls rather than replacing a correct ROCm stack.

After installation, verify that pip did not replace ROCm torch with CUDA or CPU wheels.

### Packages that may replace torch

Inspect the final environment after all package dependencies are installed. If CUDA torch was pulled in, restore the exact matching wheels from the same repository as the base image.

Do not install Triton separately unless metadata and imports prove it is missing or incorrect. Modern ROCm torch wheels may declare the matching ROCm Triton dependency directly.

### Native HIP and CMake packages

Prefer conventional paths exposed by `ryzer_env`:

```dockerfile
-DHIP_ROOT_DIR=/opt/rocm
-DHIP_PATH=/opt/rocm
-DCMAKE_PREFIX_PATH="/opt/rocm/lib/cmake;<torch-cmake-path>"
-DCMAKE_HIP_ARCHITECTURES=<target-gfx>
```

Set the architecture explicitly because Docker builds normally cannot detect the host GPU. Inspect linker errors for nested wheel library directories such as `host-math` or `rocm_sysdeps`; add only paths proven necessary.

### Upstream version constraints and C++ APIs

Patch the smallest possible constraint or API surface. Confirm:

- The new dependency version is actually compatible
- The patch does not break the previous supported version unnecessarily
- An upstream issue or fix already exists

Prefer a version-compatible upstream fix over an unconditional downstream `sed` replacement.

## 6. Make tests deterministic

Every package test must:

- Exit nonzero on real failure
- Exercise the GPU when the package claims GPU support
- Check `torch.version.hip` and GPU availability where applicable
- Use bounded iterations or token counts
- Avoid interactive prompts and GUI requirements
- Avoid unnecessary large model downloads
- Produce useful diagnostics without masking errors through `|| true`

Use `xvfb-run` for desktop applications in headless tests. Set MuJoCo EGL variables in the test command when only tests need headless rendering.

Commands and APIs may change independently of ROCm. Check current `--help` output rather than preserving a command that no longer exists.

### Test assets and volume mounts

Never bake test inputs into a directory replaced by `config.yaml` at runtime. A bind mount hides image contents.

- Put immutable smoke-test inputs under `/opt/<package>-test-data`.
- Keep user data and generated outputs in the mounted directory.
- Test with an empty mount at the configured destination.
- Update the README with the exact interactive-shell command and persisted host output path.

## 7. Build and test in layers

1. Build `ryzer_env`.
2. Build native compilation packages early.
3. Respect package chains; for example, a package consuming llama.cpp must build on the updated llama.cpp image.
4. Build all in-scope packages.
5. Run GPU tests sequentially to avoid memory contention.
6. For each package, record build status, test status, GPU detection, and any compatibility patch.
7. Rebuild and retest after changing shared environment layers.

Use direct container tests for diagnostics, then test through normal `ryzers run` semantics so environment variables and volume mappings are covered.

## 8. Benchmark regressions

When performance validation is requested:

1. Build matched old and new images.
2. Use identical models, inputs, precision, warmups, and iteration counts.
3. Synchronize the GPU around timing.
4. Measure latency, throughput, and peak memory.
5. Run workloads sequentially and reuse model caches.
6. Report regressions as clearly as improvements; do not generalize from one model.

## 9. Final audit

Before handoff:

- Review every diff and explain why it is necessary.
- Confirm no excluded package changed.
- Confirm redundant package base-image pins are gone.
- Confirm no stale gfx overrides or old ROCm indexes remain unintentionally.
- Validate README commands and output paths.
- Run formatting, lint, and `git diff --check`.
- Keep comments brief and factual; remove banner comments and generated-sounding prose.
- Use package-prefixed commit subjects such as `opensplat: support ROCm <version>`.
- Group only truly mechanical multi-package changes.
- Do not commit, rewrite history, force-push, prune images, or create a PR unless the user requests it.

The final report should list changed packages, test evidence, known regressions, upstream patches, and anything intentionally left pinned.
