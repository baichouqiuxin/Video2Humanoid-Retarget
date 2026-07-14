# Video2Humanoid-Retarget

**A Modular Framework for Video-based Humanoid Motion Reconstruction, Motion Retargeting and Robot Simulation.**

Video2Humanoid-Retarget turns a monocular video into reusable humanoid robot motion artifacts. The current implementation uses GEM-X for human motion extraction, NVIDIA soma-retargeter/Newton for Unitree G1 retargeting, and Isaac Sim for kinematic replay visualization.

## Features

- Modular extractor / retargeter / simulator interfaces.
- Current backend: video → GEM-X SOMA → SOMA BVH → Unitree G1 CSV/NPY → Isaac USD replay.
- Contact-aware ground alignment after retargeting: Newton FK evaluates G1 foot-sole meshes, infers support from sole height and speed, locks support-foot XY, and aligns root Z to the ground.
- Kinematic Isaac replay mode for retarget validation without uncontrolled robot falling.
- Flattened ground-plane human motion overlay in the Isaac replay for visual comparison; it preserves horizontal motion only and does not recover human height.
- YAML-driven configuration and reproducible output folders.
- Designed for future DuoMo, WHAM, HMR, GMR, MuJoCo, PyBullet, and Genesis backends.

## Architecture

```text
Video
  ↓
MotionExtractor
  ↓
Human Motion Artifacts
  ↓
MotionRetargeter
  ↓
Robot Motion Artifacts
  ↓
Simulator / Replay Backend
  ↓
USD, video, screenshots, statistics
```

## Quick Start

```bash
python run.py --input input/my_video.mp4 --output outputs/my_video
```

or:

```bash
bash scripts/run_pipeline.sh input/my_video.mp4 outputs/my_video
```

Open a generated Isaac replay:

```bash
ISAAC_ROOT=/path/to/isaacsim \
scripts/open_retarget_demo.sh outputs/my_video/simulation/usd/my_video_g1_kinematic_replay.usd
```

## Installation

```bash
bash scripts/download_dependencies.sh
bash scripts/install.sh
```

Then update `configs/config.yaml` with local paths for GEM-X, soma-retargeter, Isaac Sim, and assets.

## Environment Check

```bash
export GEMX_ROOT=/path/to/GEM-X
export SOMA_RETARGETER_ROOT=/path/to/soma-retargeter
export ISAAC_ROOT=/path/to/isaacsim
export SOMA_ASSETS_ROOT=/path/to/soma_assets
bash scripts/check_environment.sh
```

## Folder Structure

```text
Video2Humanoid-Retarget/
  configs/       YAML configuration
  docs/          API, architecture, diagrams, roadmap
  examples/      example input/configs
  scripts/       install, dependency, launch helpers
  tools/         reusable conversion utilities
  video2humanoid Python package
  outputs/       generated run outputs
  tests/         smoke tests
```

## Output Structure

Each run creates:

```text
outputs/sample_name/
  human/
    hpe_results.pt
  retarget/
    bvh/
    csv/
      sample_soma_raw.csv
      sample_soma_ground_aligned.csv
    motion/
    robot.npy
    logs/
      ground_alignment.json
  simulation/
    usd/
    screenshots/
  logs/
  manifest.json
```

## Examples

The local `file3`, `file4`, and `file5` runs are examples only. They are not hardcoded into the framework.

## Troubleshooting

- **Robot falls in Isaac:** use `*_g1_kinematic_replay.usd`, not dynamic PhysX target-position stages.
- **Foot sliding or penetration:** inspect `retarget/logs/ground_alignment.json`. It records inferred support intervals, sole heights, root corrections, and contact-foot slip; tune the `ground_alignment` thresholds in `configs/config.yaml` for unusual motions.
- **Human overlay has no height:** `/World/HumanMotion_FlattenedGround` intentionally projects SOMA joints onto the ground plane, so it is a 2D motion reference rather than a 3D body-height estimate.
- **Missing SMPL assets:** download licensed SMPL/SOMA assets from official sources and update config paths.
- **No robot visible:** select `/World/G1_KinematicReplay` in Stage and press `F`.
- **GEM-X fails:** verify the `gemx` conda environment and checkpoint/assets.

## Ground Alignment

The current pipeline applies ground alignment after SOMA-to-G1 retargeting, not during GEM-X reconstruction. It preserves the raw CSV and writes a separate aligned CSV. Newton FK provides real left/right foot-sole mesh positions; support candidates require low sole height plus bounded vertical and horizontal speed. During a support segment the root XY is compensated so the contact foot remains fixed, and root Z is shifted so the supporting sole reaches the configured clearance. Isaac export then applies only an upward per-frame safety clamp for the configured ground scope, so numerical penetration is removed without replacing the contact decision.

This is kinematic contact consistency, not dynamic balance control. A controller is still required before deployment on a physical floating-base robot.

## Roadmap

See `docs/ROADMAP.md`.

## Citation

If you publish results from this framework, cite the upstream methods used in your configured backend, including GEM-X, SOMA/SOMA-X, soma-retargeter, Newton, ProtoMotions, and Isaac Sim where applicable.

## Credits

This framework integrates with upstream NVIDIA/NVlabs robotics and human-motion projects through clean adapters. It does not vendor those projects.

## License

Intended license: Apache-2.0. Replace `LICENSE` with the full official license text before release.
