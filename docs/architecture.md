# Architecture

Video2Humanoid-Retarget is intentionally not a GEM-X wrapper. It is a plugin-style framework with replaceable backends.

## Design Goals

1. Accept arbitrary monocular video input.
2. Keep motion extraction, retargeting, and simulation independent.
3. Avoid vendoring large third-party research repositories.
4. Preserve reproducible intermediate artifacts.
5. Support research iteration without changing user-facing commands.

## Modules

### Motion Extraction

The `MotionExtractor` interface converts a video into human motion artifacts. The current backend is `GemXExtractor`, but future extractors can provide the same result object from DuoMo, WHAM, HMR, or other methods.

### Motion Retargeting

The `MotionRetargeter` interface converts extracted human motion into robot motion. The current backend uses SOMA BVH and NVIDIA soma-retargeter/Newton to produce Unitree G1 CSV and NPY trajectories.

The current G1 backend then performs contact-aware ground alignment before simulation. Newton FK evaluates the actual foot-sole mesh vertices; a low, slow foot is treated as support, support-foot XY is anchored, and root Z is aligned to the sole clearance. Both raw and aligned CSV files are retained with a JSON contact/slip report.

### Simulation

The `Simulator` interface converts robot motion into simulator artifacts. The current Isaac backend generates kinematic replay USD files for visual validation. Future backends can target MuJoCo, PyBullet, Genesis, or controller-driven Isaac simulations.

## Why Kinematic Replay?

Retargeted motion is a reference trajectory, not a balance policy. A floating-base humanoid in PhysX will fall unless a stabilizing controller tracks the reference. The kinematic replay backend validates retarget pose quality without confusing it with robot control failure.

## Adding a New Backend

1. Implement one of the abstract interfaces in `video2humanoid/interfaces.py`.
2. Add the class to `video2humanoid/registry.py`.
3. Add YAML configuration in `configs/`.
4. Document dependencies and output format.

## Data Contract

Each run writes a manifest and keeps intermediate artifacts so any stage can be inspected, replaced, or re-run.
