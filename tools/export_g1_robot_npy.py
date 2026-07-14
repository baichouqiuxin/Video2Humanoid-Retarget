#!/usr/bin/env python3
"""Export soma-retargeter Unitree G1 CSV to a compact numpy trajectory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def export_robot_npy(csv_path: Path, output_npy: Path, output_json: Path, fps: float):
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        raw = np.asarray([[float(value) for value in row] for row in reader], dtype=np.float32)

    trajectory = np.zeros((raw.shape[0], 7 + (raw.shape[1] - 7)), dtype=np.float32)
    trajectory[:, 0:3] = raw[:, 1:4] * 0.01
    trajectory[:, 3:7] = Rotation.from_euler("xyz", raw[:, 4:7], degrees=True).as_quat().astype(np.float32)
    trajectory[:, 7:] = np.deg2rad(raw[:, 7:])

    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, trajectory)
    output_json.write_text(
        json.dumps(
            {
                "source_csv": str(csv_path),
                "format": "root_xyz_m_root_quat_xyzw_joint_dofs_rad",
                "fps": float(fps),
                "shape": list(trajectory.shape),
                "csv_header": header,
                "joint_names": [name.removesuffix("_dof") for name in header[7:]],
                "has_nan": bool(np.isnan(trajectory).any()),
                "joint_rad_min": float(np.nanmin(trajectory[:, 7:])),
                "joint_rad_max": float(np.nanmax(trajectory[:, 7:])),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved robot trajectory: {output_npy} shape={trajectory.shape}")
    print(f"Saved metadata: {output_json}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-npy", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    export_robot_npy(args.csv, args.output_npy, args.output_json, args.fps)


if __name__ == "__main__":
    main()
