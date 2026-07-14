# Environment Notes

This repository does not vendor heavy third-party dependencies.

Expected external components:

- GEM-X installation and checkpoints.
- soma-retargeter installation.
- Newton and Newton assets.
- Isaac Sim 5.x.
- SOMA/SMPL assets where required by upstream tools.

Use `scripts/download_dependencies.sh` as a guide, then update `configs/config.yaml`.
