# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Fetch the xArm6 mesh binaries from their official UFACTORY upstreams.

We do NOT vendor any of the xArm6 mesh binaries (visual STL, vhacd collision
OBJ/MTL, gripper STL) in this package. They are downloaded here, byte-for-byte,
from the permissively-licensed UFACTORY repositories and verified against a
pinned SHA-256 manifest. Only the AMD-authored MuJoCo wrappers (`robot.xml`,
`xarm_gripper.xml`) live in the package; they reference the meshes this script
materializes under `meshes/`.

Provenance (all 30 files byte-identical to the pinned upstream commits):
  * arm visual STL + collision vhacd OBJ/MTL (23 files)
      xArm-Developer/uf-gym (MIT), urdf/xarm/xarm_description/meshes/xarm6/
  * gripper STL (7 files)
      xArm-Developer/xarm_ros2 @ humble (BSD-3-Clause),
      xarm_description/meshes/gripper/xarm/

Run at image-build time (see the Dockerfile) so the meshes are baked in and no
network access is needed when a demo runs. Idempotent: a file whose SHA-256
already matches is left untouched, so re-runs are cheap and the demo's first
`EMBODIMENT=xarm6` launch can also self-heal a missing mesh.

Usage:  python fetch_meshes.py [--dest DIR]   (default DIR = this file's dir)
"""
import argparse
import hashlib
import os
import sys
import time
import urllib.request

# Pinned upstream commits — frozen so the fetched bytes can never drift.
UF_GYM_COMMIT = "5d93e0419193bfed8a7cefc7bbe9b045a650f049"      # xArm-Developer/uf-gym (MIT)
XARM_ROS2_COMMIT = "d0b95117dabd3883f41155125aa3f67d37901c18"   # xArm-Developer/xarm_ros2 @ humble (BSD-3)

_ARM_BASE = (
    "https://raw.githubusercontent.com/xArm-Developer/uf-gym/"
    f"{UF_GYM_COMMIT}/urdf/xarm/xarm_description/meshes"
)
_GRIPPER_BASE = (
    "https://raw.githubusercontent.com/xArm-Developer/xarm_ros2/"
    f"{XARM_ROS2_COMMIT}/xarm_description/meshes"
)

# suffix (relative to meshes/) -> sha256. The suffix doubles as the upstream path
# under each repo's meshes/ root, so the layout maps 1:1 (no per-file URL table).
# Arm meshes (suffix starts with "xarm6/") come from uf-gym; everything else
# ("gripper/...") comes from xarm_ros2.
MANIFEST = {
    # --- arm visual (uf-gym) ---
    "xarm6/visual/base.stl": "d3d61f2888bc39eecb8583babc7dddccb92f2ccd55aa32cd6f1dce6193feb3b6",
    "xarm6/visual/link1.stl": "103a81f9f00c217247f8b6c8f19b3afcbc9288852082389044d88a4a321cd8cd",
    "xarm6/visual/link2.stl": "20ef55e34f8a63bb65108ed3134c4b22d911b55cd496ca5562b3978cba88ac77",
    "xarm6/visual/link3.stl": "f1503bc0c1fe8d39f4c509dba7288d9130954a35a1bcbaa0fca95d083fed3ffe",
    "xarm6/visual/link4.stl": "2a9952085fef91045380dac9825d9ad5a75867196933bc2cb1c3a635c23b0b7e",
    "xarm6/visual/link5.stl": "bfca01aae4255803ba2f5391c93836b91e1761e069275e38a9656ff2d53f45a4",
    "xarm6/visual/link6.stl": "b90cb1b373e65c035d7965bde92f4e2aedacdbbd57785a9018cd996b6b0b688e",
    # --- arm collision: vhacd OBJ + MTL (uf-gym) ---
    "xarm6/collision/base_vhacd.obj": "4e64122a1689aa822c6d8b88fdf4e0fbc54fc94b02d92be0e0e8e1308b7e3e29",
    "xarm6/collision/link1_vhacd.obj": "8cb5eceaa5fccafa5cbe94adec4e7701ade06c932a385b940b12eeb441a529ed",
    "xarm6/collision/link2_vhacd.obj": "fc03ef74436b1abfd45842a3e555fb257eff7f3ecd76eb394a771afccf1c89ed",
    "xarm6/collision/link2_vhacd2.obj": "2fc7451ef8b21d2cbfd8ca34f5d28267215a355f8ba3b8b192e9cc0984edc284",
    "xarm6/collision/link3_vhacd.obj": "932799e0038645092293eb3e197f5c550fd3a8a53cec97822f797ac63a537c92",
    "xarm6/collision/link4_vhacd.obj": "5907222062d2528bb4cf2179a2f3e5fdb963b27e2d906c1348075e3dd6197463",
    "xarm6/collision/link5_vhacd.obj": "10250a224167c8e14f00dbb277a15b71f18b3db102ee7730abbabe9850459895",
    "xarm6/collision/link6_vhacd.obj": "cce755e58e1719514531b4eb96e32e9fbb71228af586691221aceea8272c07d9",
    "xarm6/collision/base.mtl": "e23b5279777ea3bcbaeb3a0c748f95a51d1dc3bcce1db8c035a9361512e73a66",
    "xarm6/collision/link1.mtl": "e23b5279777ea3bcbaeb3a0c748f95a51d1dc3bcce1db8c035a9361512e73a66",
    "xarm6/collision/link2.mtl": "e23b5279777ea3bcbaeb3a0c748f95a51d1dc3bcce1db8c035a9361512e73a66",
    "xarm6/collision/link2_vhacd2.mtl": "fdbc8ef1d184ea57a0bbb1b6e688b3058da58ac7c1d6981a697e419cb473916f",
    "xarm6/collision/link3.mtl": "e23b5279777ea3bcbaeb3a0c748f95a51d1dc3bcce1db8c035a9361512e73a66",
    "xarm6/collision/link4.mtl": "e23b5279777ea3bcbaeb3a0c748f95a51d1dc3bcce1db8c035a9361512e73a66",
    "xarm6/collision/link5.mtl": "e23b5279777ea3bcbaeb3a0c748f95a51d1dc3bcce1db8c035a9361512e73a66",
    "xarm6/collision/link6.mtl": "e23b5279777ea3bcbaeb3a0c748f95a51d1dc3bcce1db8c035a9361512e73a66",
    # --- gripper STL (xarm_ros2 @ humble) ---
    "gripper/xarm/base_link.stl": "cdaa4cff22f7c9cff05c6a8ed32f94fd2b11a69d37dd97a159ccee2d8dd32f13",
    "gripper/xarm/left_finger.stl": "a44756eb72f9c214cb37e61dc209cd7073fdff3e4271a7423476ef6fd090d2d4",
    "gripper/xarm/left_inner_knuckle.stl": "e8e48692ad26837bb3d6a97582c89784d09948fc09bfe4e5a59017859ff04dac",
    "gripper/xarm/left_outer_knuckle.stl": "501665812b08d67e764390db781e839adc6896a9540301d60adf606f57648921",
    "gripper/xarm/right_finger.stl": "c5dee87c7f37baf554b8456ebfe0b3e8ed0b22b8938bd1add6505c2ad6d32c7d",
    "gripper/xarm/right_inner_knuckle.stl": "b41dd2c2c550281bf78d7cc6fa117b14786700e5c453560a0cb5fd6dfa0ffb3e",
    "gripper/xarm/right_outer_knuckle.stl": "75ca1107d0a42a0f03802a9a49cab48419b31851ee8935f8f1ca06be1c1c91e8",
}


def _url_for(suffix):
    base = _ARM_BASE if suffix.startswith("xarm6/") else _GRIPPER_BASE
    return f"{base}/{suffix}"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url, retries=4, timeout=60):
    last = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "molmoact2-xarm6-fetch"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # network hiccup / transient 5xx
            last = exc
            if attempt < retries:
                wait = 2 ** attempt
                print(f"  ! {url} failed ({exc}); retry {attempt}/{retries - 1} in {wait}s", flush=True)
                time.sleep(wait)
    raise RuntimeError(f"download failed after {retries} attempts: {url} ({last})")


def fetch(dest_root):
    meshes_root = os.path.join(dest_root, "meshes")
    have, pulled = 0, 0
    for suffix, want in sorted(MANIFEST.items()):
        out = os.path.join(meshes_root, *suffix.split("/"))
        if os.path.isfile(out) and _sha256(out) == want:
            have += 1
            continue
        url = _url_for(suffix)
        data = _download(url)
        got = hashlib.sha256(data).hexdigest()
        if got != want:
            raise SystemExit(
                f"SHA-256 mismatch for {suffix}\n  url:      {url}\n"
                f"  expected: {want}\n  got:      {got}"
            )
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "wb") as fh:
            fh.write(data)
        pulled += 1
        print(f"  + meshes/{suffix}  ({len(data)} B, sha256 ok)", flush=True)
    print(f"xArm6 meshes ready: {pulled} downloaded, {have} already present, "
          f"{len(MANIFEST)} total (all sha256-verified).", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Fetch + verify xArm6 mesh binaries from UFACTORY upstreams.")
    ap.add_argument("--dest", default=os.path.dirname(os.path.abspath(__file__)),
                    help="assets/xarm6 root that holds robot.xml (default: this file's dir)")
    args = ap.parse_args()
    print(f"Fetching {len(MANIFEST)} xArm6 meshes into {os.path.join(args.dest, 'meshes')}", flush=True)
    print(f"  uf-gym    @ {UF_GYM_COMMIT[:12]} (MIT)        -> arm visual + collision", flush=True)
    print(f"  xarm_ros2 @ {XARM_ROS2_COMMIT[:12]} (BSD-3)      -> gripper", flush=True)
    fetch(args.dest)


if __name__ == "__main__":
    sys.exit(main())
