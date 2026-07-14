# API Documentation

## `run.py`

- **Purpose:** Single command-line entry point for the full pipeline.
- **Inputs:** `--input`, optional `--output`, optional `--config`.
- **Outputs:** run folder with human, retarget, simulation, logs, and manifest artifacts.
- **Example:** `python run.py --input input/my_video.mp4 --output outputs/my_video`.

## `video2humanoid.interfaces`

- **Purpose:** Defines `MotionExtractor`, `MotionRetargeter`, and `Simulator` plugin contracts.
- **Extension:** Add a new implementation class and register it in `video2humanoid/registry.py`.

## `video2humanoid.extractors.gemx.GemXExtractor`

- **Purpose:** Runs GEM-X or reuses existing `hpe_results.pt`.
- **Inputs:** video path, extractor config.
- **Outputs:** `human/hpe_results.pt`.
- **Dependencies:** GEM-X, `gemx` conda environment, SOMA assets.

## `video2humanoid.retargeters.soma_g1.SomaG1Retargeter`

- **Purpose:** Converts GEM-X SOMA parameters to SOMA BVH, retargets to Unitree G1, then applies contact-aware ground alignment by default.
- **Inputs:** `hpe_results.pt`.
- **Outputs:** SOMA BVH, raw G1 CSV, ground-aligned G1 CSV, `robot.npy`, contact report, metadata.
- **Dependencies:** soma-retargeter, Newton, G1 assets.

## `video2humanoid.simulators.isaac_kinematic.IsaacKinematicReplay`

- **Purpose:** Generates an Isaac-openable kinematic mesh replay using Newton FK.
- **Inputs:** G1 CSV.
- **Outputs:** animated USD.
- **Dependencies:** Newton, USD Python libraries, trimesh.

## `tools/export_gem_soma_bvh.py`

- **Purpose:** Convert GEM-X `hpe_results.pt` to SOMA BVH.
- **Inputs:** `--hpe`, `--output-bvh`, `--gemx-root`, `--reference-bvh`, `--fps`.
- **Outputs:** BVH with SOMA skeleton hierarchy and local rotations.
- **Algorithm:** Uses GEM-X SOMA body pose, identity, scale, and root translation; builds a personalized SOMA skeleton; writes BVH motion channels.

## `tools/export_g1_robot_npy.py`

- **Purpose:** Convert soma-retargeter Unitree G1 CSV to compact NumPy trajectory.
- **Inputs:** `--csv`, `--output-npy`, `--output-json`, `--fps`.
- **Outputs:** `robot.npy` with root xyz, root quaternion, joint radians; metadata JSON.

## `tools/ground_align_g1_csv.py`

- **Purpose:** Apply contact-aware ground alignment to a Unitree G1 CSV after retargeting and before simulation.
- **Inputs:** `--input-csv`, `--output-csv`, `--report`, optional contact thresholds and `--fps`.
- **Outputs:** Ground-aligned CSV plus JSON contact report with left/right support segments, sole heights, contact slip, and root XY/Z corrections.
- **Algorithm:** Evaluates Newton FK using the actual G1 foot meshes. A foot is a support candidate only when its sole is near its local floor and its vertical and horizontal speeds are low. During each support segment, root XY is adjusted to keep the support foot fixed, and root Z is shifted so the sole reaches the ground without allowing downward jumps.

## `tools/export_g1_kinematic_usd.py`

- **Purpose:** Export G1 retargeted motion as kinematic Isaac USD.
- **Inputs:** `--csv`, `--output-usd`, `--fps`, optional `--ground-aligned`, `--ground-scope`, and `--sole-clearance`.
- **Outputs:** animated mesh USD.
- **Algorithm:** Loads Unitree G1 assets through Newton, evaluates FK per frame, and writes animated mesh transforms with no gravity or balance simulation. With `--ground-aligned`, a final feet-only per-frame upward safety correction prevents mesh penetration below the ground plane.

## `scripts/open_retarget_demo.sh`

- **Purpose:** Open an animated USD in Isaac Sim and start playback.
- **Inputs:** USD path argument and `ISAAC_ROOT` environment variable.
- **Outputs:** Isaac Sim GUI session.
