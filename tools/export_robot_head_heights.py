#!/usr/bin/env python3
"""Export Unitree G1 head-height curves from retargeted robot motion arrays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import newton
import numpy as np
from scipy.spatial.transform import Rotation

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_g1_kinematic_usd import (  # noqa: E402
    build_newton_model,
    compute_ground_offset,
    evaluate_body_transforms,
    load_visual_geometries,
)


def load_motion(path: Path) -> np.ndarray:
    motion = np.load(path)
    if motion.ndim != 2 or motion.shape[1] < 8:
        raise ValueError(f"Expected T x DOF robot motion at {path}, got shape {motion.shape}")
    if np.isnan(motion).any():
        raise ValueError(f"NaN values found in {path}")
    return motion.astype(np.float32, copy=False)


def transformed_vertices_z(
    geom: dict,
    body_q_frame: np.ndarray,
    body_name_to_index: dict[str, int],
    z_offset: float,
) -> np.ndarray | None:
    body_index = body_name_to_index.get(geom["body_name"])
    if body_index is None:
        return None
    body_pos = body_q_frame[body_index, :3].astype(np.float64)
    body_rot = Rotation.from_quat(body_q_frame[body_index, 3:7])
    final_pos = body_pos + body_rot.apply(geom["local_pos"])
    final_pos[2] += z_offset
    final_rot = body_rot * geom["local_rot"]
    vertices = final_rot.apply(geom["vertices"].astype(np.float64)) + final_pos
    return vertices[:, 2]


def compute_head_heights(motion_path: Path) -> dict:
    joint_q = load_motion(motion_path)
    asset_root = newton.utils.download_asset("unitree_g1")
    geometries, mjcf_path = load_visual_geometries(asset_root)
    model, state, body_name_to_index = build_newton_model(mjcf_path)
    body_q = evaluate_body_transforms(model, state, joint_q)
    z_offset = compute_ground_offset(geometries, body_name_to_index, body_q[0])

    head_geometries = [
        geom
        for geom in geometries
        if "head" in geom["mesh_name"].lower() or "head" in geom["body_name"].lower()
    ]
    if not head_geometries:
        head_geometries = [geom for geom in geometries if geom["body_name"] == "torso_link"]
    if not head_geometries:
        raise RuntimeError("Could not locate G1 head/torso geometry")
    head_geometry_ids = {id(geom) for geom in head_geometries}

    head_heights = []
    robot_top_heights = []
    for frame_body_q in body_q:
        frame_head_max = []
        frame_robot_max = []
        for geom in geometries:
            z_values = transformed_vertices_z(geom, frame_body_q, body_name_to_index, z_offset)
            if z_values is None:
                continue
            frame_robot_max.append(float(z_values.max()))
            if id(geom) in head_geometry_ids:
                frame_head_max.append(float(z_values.max()))
        if not frame_head_max:
            raise RuntimeError("Head geometry vanished during FK evaluation")
        head_heights.append(max(frame_head_max))
        robot_top_heights.append(max(frame_robot_max) if frame_robot_max else max(frame_head_max))

    head_array = np.asarray(head_heights, dtype=np.float64)
    robot_top_array = np.asarray(robot_top_heights, dtype=np.float64)
    return {
        "label": "Isaac Sim G1 head",
        "source_motion": str(motion_path),
        "asset_mjcf": str(mjcf_path),
        "method": "Newton FK on Unitree G1 torso_link/head_link mesh; ground offset matches USD export.",
        "head_geometries": [f"{geom['body_name']}/{geom['mesh_name']}" for geom in head_geometries],
        "fps": 30.0,
        "frame_count": int(head_array.shape[0]),
        "initial_height_m": float(head_array[0]),
        "height_m": head_array.round(6).tolist(),
        "delta_height_m": (head_array - head_array[0]).round(6).tolist(),
        "robot_top_height_m": robot_top_array.round(6).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = compute_head_heights(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.output} frames={result['frame_count']} initial={result['initial_height_m']:.3f}m")


if __name__ == "__main__":
    main()
