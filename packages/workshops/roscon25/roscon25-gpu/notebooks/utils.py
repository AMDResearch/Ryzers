import torch
import matplotlib.pyplot as plt
import cv2

@torch.no_grad()
def sanitize_norm(policy):
    # images
    buf_imgs = getattr(policy.normalize_inputs, "buffer_images", {})
    for k, buf in buf_imgs.items():
        for name in ["mean", "std"]:
            t = getattr(buf, name)
            t = t.clone()
            # replace non-finite
            t[~torch.isfinite(t)] = 0 if name == "mean" else 1
            # guard zero/near-zero std
            if name == "std":
                t = torch.where(t.abs() < 1e-6, torch.ones_like(t), t)
            setattr(buf, name, torch.nn.Parameter(t, requires_grad=False))

    # state
    m = policy.normalize_inputs.buffer_observation_state.mean.clone()
    s = policy.normalize_inputs.buffer_observation_state.std.clone()
    m[~torch.isfinite(m)] = 0
    s[~torch.isfinite(s)] = 1
    s = torch.where(s.abs() < 1e-6, torch.ones_like(s), s)
    policy.normalize_inputs.buffer_observation_state.mean = torch.nn.Parameter(m, requires_grad=False)
    policy.normalize_inputs.buffer_observation_state.std  = torch.nn.Parameter(s, requires_grad=False)


def plot_image(image_path: str):
    """Image plotting helper"""
    img = cv2.imread(image_path)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

import os, sys, ctypes, glob

def ros_setup(install_dir: str):
    """
    Dynamically sets up a ROS 2 workspace environment inside Jupyter.
    Mirrors the effect of: source <install_dir>/setup.bash
    - Fixes PYTHONPATH
    - Fixes LD_LIBRARY_PATH
    - Preloads all generated .so libraries (generator + typesupport)
    """

    lib_dir = os.path.join(install_dir, "lib")
    py_dir = os.path.join(lib_dir, "python3.12", "site-packages")

    # --- 1. Ensure environment paths
    os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    if py_dir not in sys.path:
        sys.path.insert(0, py_dir)
    if install_dir in sys.path:
        sys.path.remove(install_dir)

    def _safe_load(so_path):
        """Helper: safely preload shared libs with RTLD_GLOBAL"""
        try:
            ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
            print(f"Loaded: {os.path.basename(so_path)}")
        except OSError as e:
            print(f"Failed: {os.path.basename(so_path)} → {e}")

    # --- 2. Load core generators first
    core_libs = [
        "libvlm_ros__rosidl_generator_c.so",
        "libvlm_ros__rosidl_generator_py.so",
    ]
    for name in core_libs:
        path = os.path.join(lib_dir, name)
        if os.path.exists(path):
            _safe_load(path)

    # --- 3. Load all type-support libs (depend on core)
    for name in [
        "libvlm_ros__rosidl_typesupport_c.so",
        "libvlm_ros__rosidl_typesupport_cpp.so",
        "libvlm_ros__rosidl_typesupport_fastrtps_c.so",
        "libvlm_ros__rosidl_typesupport_fastrtps_cpp.so",
        "libvlm_ros__rosidl_typesupport_introspection_c.so",
        "libvlm_ros__rosidl_typesupport_introspection_cpp.so",
    ]:
        path = os.path.join(lib_dir, name)
        if os.path.exists(path):
            _safe_load(path)

    print("\n ROS environment initialized for:", install_dir)
    print("LD_LIBRARY_PATH head:", ":".join(os.environ["LD_LIBRARY_PATH"].split(":")[:5]))
