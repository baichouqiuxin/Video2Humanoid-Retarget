#!/usr/bin/env python3
"""Contact-aware G1 ground alignment for soma-retargeter CSV trajectories.

The tool uses Newton FK and real G1 foot meshes to infer support feet. It then
locks the support-foot XY position and shifts the root Z so the supporting sole
lies on the requested floor clearance. The source CSV is never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import newton
import numpy as np
from scipy.spatial.transform import Rotation

from export_g1_kinematic_usd import (
    build_newton_model,
    evaluate_body_transforms,
    load_csv_as_joint_q,
    load_visual_geometries,
)


def frame_speed(values: np.ndarray, fps: float) -> np.ndarray:
    if len(values) < 2:
        return np.zeros(len(values), dtype=np.float64)
    delta = np.diff(values, axis=0)
    return np.linalg.norm(np.vstack([delta[:1], delta]), axis=1) * fps


def rolling_min(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or len(values) < 2:
        return values.copy()
    padded = np.pad(values, (radius, radius), mode="edge")
    windows = np.stack([padded[index : index + len(values)] for index in range(2 * radius + 1)])
    return windows.min(axis=0)


def clean_contact_mask(mask: np.ndarray, max_gap: int, min_frames: int) -> np.ndarray:
    result = mask.astype(bool, copy=True)
    index = 0
    while index < len(result):
        if result[index]:
            index += 1
            continue
        start = index
        while index < len(result) and not result[index]:
            index += 1
        if start > 0 and index < len(result) and index - start <= max_gap:
            result[start:index] = True
    index = 0
    while index < len(result):
        if not result[index]:
            index += 1
            continue
        start = index
        while index < len(result) and result[index]:
            index += 1
        if index - start < min_frames:
            result[start:index] = False
    return result


def geometry_vertices(frame_body_q: np.ndarray, geometries: list[dict], body_indices: dict[str, int], body: str) -> np.ndarray:
    vertices: list[np.ndarray] = []
    for geometry in geometries:
        if geometry["body_name"] != body:
            continue
        body_index = body_indices.get(body)
        if body_index is None:
            continue
        body_pos = frame_body_q[body_index, :3].astype(np.float64)
        body_rot = Rotation.from_quat(frame_body_q[body_index, 3:7])
        position = body_pos + body_rot.apply(geometry["local_pos"])
        rotation = body_rot * geometry["local_rot"]
        vertices.append(rotation.apply(geometry["vertices"].astype(np.float64)) + position)
    if not vertices:
        raise ValueError(f"no visual geometry found for foot body {body!r}")
    return np.concatenate(vertices, axis=0)


def foot_statistics(
    body_q: np.ndarray, geometries: list[dict], body_indices: dict[str, int], body: str
) -> tuple[np.ndarray, np.ndarray]:
    centers = np.empty((len(body_q), 3), dtype=np.float64)
    sole_z = np.empty(len(body_q), dtype=np.float64)
    for frame, frame_body_q in enumerate(body_q):
        vertices = geometry_vertices(frame_body_q, geometries, body_indices, body)
        centers[frame] = vertices.mean(axis=0)
        sole_z[frame] = vertices[:, 2].min()
    return centers, sole_z


def infer_contacts(
    left_centers: np.ndarray,
    left_sole_z: np.ndarray,
    right_centers: np.ndarray,
    right_sole_z: np.ndarray,
    fps: float,
    height_m: float,
    vertical_speed_mps: float,
    horizontal_speed_mps: float,
    gap_frames: int,
    min_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local_floor = rolling_min(np.minimum(left_sole_z, right_sole_z), max(10, int(round(fps * 0.75))))

    def mask(centers: np.ndarray, sole_z: np.ndarray) -> np.ndarray:
        height = sole_z - local_floor
        vertical = frame_speed(sole_z[:, None], fps)
        horizontal = frame_speed(centers[:, :2], fps)
        return clean_contact_mask(
            (height <= height_m) & (vertical <= vertical_speed_mps) & (horizontal <= horizontal_speed_mps),
            gap_frames,
            min_frames,
        )

    return mask(left_centers, left_sole_z), mask(right_centers, right_sole_z), local_floor


def contact_segments(mask: np.ndarray) -> list[dict[str, int]]:
    segments: list[dict[str, int]] = []
    index = 0
    while index < len(mask):
        if not mask[index]:
            index += 1
            continue
        start = index
        while index + 1 < len(mask) and mask[index + 1]:
            index += 1
        segments.append({"start": int(start), "end": int(index), "frames": int(index - start + 1)})
        index += 1
    return segments


def align(input_csv: Path, output_csv: Path, report: Path, args: argparse.Namespace) -> None:
    header, joint_q = load_csv_as_joint_q(input_csv)
    fps = float(args.fps)
    asset_root = newton.utils.download_asset("unitree_g1")
    geometries, mjcf_path = load_visual_geometries(asset_root)
    model, state, body_indices = build_newton_model(mjcf_path)
    body_q = evaluate_body_transforms(model, state, joint_q)
    left_centers, left_sole_z = foot_statistics(body_q, geometries, body_indices, args.left_foot_body)
    right_centers, right_sole_z = foot_statistics(body_q, geometries, body_indices, args.right_foot_body)
    left_contact, right_contact, local_floor = infer_contacts(
        left_centers,
        left_sole_z,
        right_centers,
        right_sole_z,
        fps,
        args.contact_height_m,
        args.max_vertical_speed_mps,
        args.max_horizontal_speed_mps,
        args.contact_gap_frames,
        args.min_contact_frames,
    )

    with input_csv.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        source_header = next(reader)
        rows = [[float(value) for value in row] for row in reader]
    if len(rows) != len(joint_q):
        raise ValueError(f"CSV frames={len(rows)} but FK frames={len(joint_q)}")

    correction_xy = np.zeros((len(rows), 2), dtype=np.float64)
    correction_z = np.zeros(len(rows), dtype=np.float64)
    anchors: list[np.ndarray | None] = [None, None]
    previous = [False, False]
    applied_xy = np.zeros(2, dtype=np.float64)
    for frame in range(len(rows)):
        active = (bool(left_contact[frame]), bool(right_contact[frame]))
        centers = (left_centers[frame, :2], right_centers[frame, :2])
        sole_z = (left_sole_z[frame], right_sole_z[frame])
        candidates: list[tuple[int, np.ndarray, np.ndarray]] = []
        for side in (0, 1):
            if not active[side]:
                anchors[side] = None
                continue
            if not previous[side] or anchors[side] is None:
                anchors[side] = centers[side] + applied_xy
            candidates.append((side, anchors[side], centers[side]))
        if candidates:
            continuing = [item for item in candidates if previous[item[0]]]
            selected = continuing if continuing else candidates
            applied_xy = np.mean([anchor - center for _, anchor, center in selected], axis=0)
            support_z = [sole_z[side] for side, _, _ in selected]
        else:
            support_z = [min(sole_z)]
        desired_z = float(args.sole_clearance_m) - min(support_z)
        correction_z[frame] = desired_z if frame == 0 else max(desired_z, correction_z[frame - 1] - args.max_down_step_m)
        correction_xy[frame] = applied_xy
        # soma-retargeter CSV stores root translation in centimeters at columns 1:4.
        rows[frame][1] += float(applied_xy[0] * 100.0)
        rows[frame][2] += float(applied_xy[1] * 100.0)
        rows[frame][3] += float(correction_z[frame] * 100.0)
        previous = [active[0], active[1]]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(source_header or header)
        writer.writerows(rows)

    def slip(centers: np.ndarray, active: np.ndarray) -> dict[str, float | int]:
        corrected = centers[:, :2] + correction_xy
        steps = np.linalg.norm(np.diff(corrected, axis=0), axis=1)
        mask = active[:-1] & active[1:]
        values = steps[mask]
        return {
            "mean_m_per_frame": float(values.mean()) if values.size else 0.0,
            "max_m_per_frame": float(values.max()) if values.size else 0.0,
            "active_steps": int(values.size),
        }

    payload = {
        "method": "Newton FK foot-sole contact inference, support-foot XY lock, and Z ground alignment",
        "units": "meters",
        "source_csv": str(input_csv),
        "output_csv": str(output_csv),
        "asset_mjcf": str(mjcf_path),
        "fps": fps,
        "foot_bodies": {"left": args.left_foot_body, "right": args.right_foot_body},
        "contact_policy": "local sole-height envelope plus vertical and horizontal foot speed",
        "left_contact_frames": int(left_contact.sum()),
        "right_contact_frames": int(right_contact.sum()),
        "left_contact_segments": contact_segments(left_contact),
        "right_contact_segments": contact_segments(right_contact),
        "left_contact_slip": slip(left_centers, left_contact),
        "right_contact_slip": slip(right_centers, right_contact),
        "left_sole_z_min_after_m": float((left_sole_z + correction_z).min()),
        "right_sole_z_min_after_m": float((right_sole_z + correction_z).min()),
        "root_xy_correction_max_m": float(np.linalg.norm(correction_xy, axis=1).max()),
        "root_z_correction_min_m": float(correction_z.min()),
        "root_z_correction_max_m": float(correction_z.max()),
        "local_floor_z_min_m": float(local_floor.min()),
        "local_floor_z_max_m": float(local_floor.max()),
        "parameters": {
            "contact_height_m": args.contact_height_m,
            "max_vertical_speed_mps": args.max_vertical_speed_mps,
            "max_horizontal_speed_mps": args.max_horizontal_speed_mps,
            "contact_gap_frames": args.contact_gap_frames,
            "min_contact_frames": args.min_contact_frames,
            "max_down_step_m": args.max_down_step_m,
            "sole_clearance_m": args.sole_clearance_m,
        },
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=True), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--left-foot-body", default="left_ankle_roll_link")
    parser.add_argument("--right-foot-body", default="right_ankle_roll_link")
    parser.add_argument("--contact-height-m", type=float, default=0.06)
    parser.add_argument("--max-vertical-speed-mps", type=float, default=0.5)
    parser.add_argument("--max-horizontal-speed-mps", type=float, default=0.8)
    parser.add_argument("--contact-gap-frames", type=int, default=2)
    parser.add_argument("--min-contact-frames", type=int, default=2)
    parser.add_argument("--max-down-step-m", type=float, default=0.03)
    parser.add_argument("--sole-clearance-m", type=float, default=0.0)
    args = parser.parse_args()
    align(args.input_csv, args.output_csv, args.report, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
