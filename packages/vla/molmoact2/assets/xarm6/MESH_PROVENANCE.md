# xArm6 asset provenance

This directory contains the MuJoCo model for the `EMBODIMENT=xarm6` demo
(`EMBODIMENT=xarm6 ryzers run /ryzers/demo_interactive.sh`). To avoid hosting
third-party geometry in this repository, the verbatim third-party mesh binaries
are **fetched at image-build time** and verified against a pinned SHA-256
manifest; only AMD-authored files and AMD-generated derivative geometry are
tracked here.

## Tracked in this repo

| File(s) | Origin | Notes |
| --- | --- | --- |
| `robot.xml`, `xarm_gripper.xml` | AMD-authored | MuJoCo wrappers that reference the meshes below. |
| `fetch_meshes.py` | AMD-authored | Downloads + sha256-verifies the 23 verbatim third-party meshes. |
| `meshes/xarm6/collision/base_vhacd.obj`<br>`meshes/xarm6/collision/link1_vhacd.obj`<br>`meshes/xarm6/collision/link2_vhacd.obj`<br>`meshes/xarm6/collision/link3_vhacd.obj`<br>`meshes/xarm6/collision/link4_vhacd.obj`<br>`meshes/xarm6/collision/link5_vhacd.obj`<br>`meshes/xarm6/collision/link6_vhacd.obj` | AMD-generated (CoACD) | Convex-decomposition collision hulls produced with [CoACD](https://github.com/SarahWeiii/CoACD) from the MIT-licensed uf-gym arm **visual** meshes (`meshes/xarm6/visual/*.stl`). Derivative geometry, not a verbatim third-party binary, so kept in-tree. |

## Fetched at build time (NOT tracked; see `.gitignore`)

23 files, each byte-identical to the pinned upstream commit and verified by
`fetch_meshes.py`:

- **Arm visual STL** (7) and **collision `.mtl` + `link2_vhacd2.obj`** (9) ->
  [`xArm-Developer/uf-gym`](https://github.com/xArm-Developer/uf-gym) (MIT),
  `urdf/xarm/xarm_description/meshes/xarm6/`, commit
  `5d93e0419193bfed8a7cefc7bbe9b045a650f049`.
- **Gripper STL** (7) ->
  [`xArm-Developer/xarm_ros2`](https://github.com/xArm-Developer/xarm_ros2) @ humble
  (BSD-3-Clause), `xarm_description/meshes/gripper/xarm/`, commit
  `d0b95117dabd3883f41155125aa3f67d37901c18`.

The fetch runs in the Dockerfile so the meshes are baked into the image; it is
idempotent and also self-heals a missing mesh on first demo launch.
