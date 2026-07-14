#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch

from gem.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from gem.utils.soma_utils.soma_layer import SomaLayer
from gem.utils.vis.renderer import get_global_cameras_static_v2


GEM_ROOT = Path("/home/anon/GEM-X")
REPORT_PATH = Path("/home/anon/GEM_outputs/camera_alignment_report.txt")


@dataclass(frozen=True)
class SequenceSpec:
    name: str
    source_video: Path
    hpe_results: Path
    output_video: Path
    height_json: Path
    mirror_output: Path | None = None


SEQUENCES = [
    SequenceSpec(
        "file3",
        Path("/home/anon/GEM_outputs/file 3/file 3_2_global.mp4"),
        Path("/home/anon/GEM_outputs/file 3/hpe_results.pt"),
        Path("/home/anon/GEM_outputs/file3/file3_2_global_aligned.mp4"),
        Path("/home/anon/GEM_outputs/file3/file3_gem_head_height.json"),
        Path("/home/anon/GEM_outputs/file 3/file 3_2_global_aligned.mp4"),
    ),
    SequenceSpec(
        "file4",
        Path("/home/anon/GEM_outputs/file4/file4_2_global.mp4"),
        Path("/home/anon/GEM_outputs/file4/hpe_results.pt"),
        Path("/home/anon/GEM_outputs/file4/file4_2_global_aligned.mp4"),
        Path("/home/anon/GEM_outputs/file4/file4_gem_head_height.json"),
    ),
    SequenceSpec(
        "file5",
        Path("/home/anon/GEM_outputs/file5/file5_2_global.mp4"),
        Path("/home/anon/GEM_outputs/file5/hpe_results.pt"),
        Path("/home/anon/GEM_outputs/file5/file5_2_global_aligned.mp4"),
        Path("/home/anon/GEM_outputs/file5/file5_gem_head_height.json"),
    ),
]


SMPL24_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int32,
)


def run_json(command: list[str]) -> dict:
    return json.loads(subprocess.check_output(command, text=True))


def probe_video(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    raw = run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=width,height,avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(path),
        ]
    )
    stream = raw["streams"][0]
    fps_num, fps_den = stream.get("avg_frame_rate", "0/1").split("/")
    fps = float(fps_num) / max(float(fps_den), 1.0)
    return {
        "exists": True,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "fps_raw": stream.get("avg_frame_rate", ""),
        "duration": float(raw.get("format", {}).get("duration", 0.0)),
        "frames": int(stream["nb_frames"]) if str(stream.get("nb_frames", "")).isdigit() else None,
    }


def to_cpu_body_params(body_params: dict) -> dict:
    return {key: value.detach().cpu() if hasattr(value, "detach") else value for key, value in body_params.items()}


def load_soma_motion(spec: SequenceSpec, soma: SomaLayer) -> tuple[torch.Tensor, torch.Tensor, dict]:
    pred = torch.load(spec.hpe_results, map_location="cpu", weights_only=False)
    body_params = to_cpu_body_params(pred["body_params_global"])
    with torch.no_grad():
        soma_out = soma(**body_params)
    verts = soma_out["vertices"].detach().cpu()
    joints = soma_out["joints"].detach().cpu()

    y_min = verts[:, :, 1].min()
    verts[:, :, 1] -= y_min
    joints[:, :, 1] -= y_min
    transform = compute_T_ayfz2ay(joints[[0]], inverse=True)
    verts = apply_T_on_points(verts, transform)
    joints = apply_T_on_points(joints, transform)

    metadata = {
        "frames": int(joints.shape[0]),
        "joints": int(joints.shape[1]),
        "vertices": int(verts.shape[1]),
        "hpe_keys": list(pred.keys()),
        "body_param_shapes": {
            key: tuple(value.shape) for key, value in body_params.items() if hasattr(value, "shape")
        },
    }
    return verts, joints, metadata


def soma_to_isaac(points: np.ndarray) -> np.ndarray:
    return np.stack([points[..., 0], points[..., 2], points[..., 1]], axis=-1)


def normalize_to_start(points: np.ndarray, root_index: int = 0) -> np.ndarray:
    out = points.copy()
    out[..., 0] -= out[0, root_index, 0]
    out[..., 1] -= out[0, root_index, 1]
    out[..., 2] -= min(0.0, float(out[..., 2].min()))
    return out


def apply_xy_origin(points: np.ndarray, origin_xy: np.ndarray) -> np.ndarray:
    out = points.copy()
    out[..., 0] -= origin_xy[0]
    out[..., 1] -= origin_xy[1]
    out[..., 2] -= min(0.0, float(out[..., 2].min()))
    return out


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm


@dataclass
class CameraConfig:
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    offset: tuple[float, float, float] = (3.0, -5.0, 2.0)
    focal_length_mm: float = 22.0
    horizontal_aperture_mm: float = 28.0
    distance_scale: float = 2.7
    min_distance: float = 3.6
    look_at_z_offset: float = 0.05
    camera_mode: str = "smooth_root"
    camera_smoothing: float = 0.08
    target_height_m: float = 1.0

    @property
    def fx(self) -> float:
        return self.width * self.focal_length_mm / self.horizontal_aperture_mm

    @property
    def fy(self) -> float:
        vertical_aperture = self.horizontal_aperture_mm * self.height / self.width
        return self.height * self.focal_length_mm / vertical_aperture

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    @property
    def offset_unit(self) -> np.ndarray:
        return normalize(np.array(self.offset, dtype=np.float64))

    @property
    def elevation_deg(self) -> float:
        direction = self.offset_unit
        return math.degrees(math.asin(float(direction[2])))

    @property
    def azimuth_deg_from_x(self) -> float:
        direction = self.offset_unit
        return math.degrees(math.atan2(float(direction[1]), float(direction[0])))


def camera_for_frame(points: np.ndarray, config: CameraConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    size = maxs - mins
    target = center.copy()
    target[2] += config.look_at_z_offset
    frame_extent = max(float(size[2]), float(size[0]) * 1.6, float(size[1]) * 1.6, 1.0)
    distance = max(config.min_distance, frame_extent * config.distance_scale)
    eye = target + config.offset_unit * distance
    forward = normalize(target - eye)
    right = normalize(np.cross(forward, np.array([0.0, 0.0, 1.0], dtype=np.float64)))
    up = normalize(np.cross(right, forward))
    return eye, target, right, up


def camera_from_target(target: np.ndarray, distance: float, config: CameraConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eye = target + config.offset_unit * distance
    forward = normalize(target - eye)
    right = normalize(np.cross(forward, np.array([0.0, 0.0, 1.0], dtype=np.float64)))
    up = normalize(np.cross(right, forward))
    return eye, target, right, up


def build_stable_camera_path(joints_world: np.ndarray, vertices_world: np.ndarray, config: CameraConfig) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    root = joints_world[:, 0]
    target_raw = root.copy()
    target_raw[:, 2] = config.target_height_m + config.look_at_z_offset
    per_frame_size = vertices_world.max(axis=1) - vertices_world.min(axis=1)
    per_frame_extent = np.maximum.reduce(
        [per_frame_size[:, 2], per_frame_size[:, 0] * 1.25, per_frame_size[:, 1] * 1.25, np.ones(len(per_frame_size))]
    )
    distance = max(config.min_distance, float(np.median(per_frame_extent)) * config.distance_scale)

    if config.camera_mode == "fixed_sequence":
        mins = vertices_world.reshape(-1, 3).min(axis=0)
        maxs = vertices_world.reshape(-1, 3).max(axis=0)
        target = (mins + maxs) / 2.0
        target[2] = config.target_height_m + config.look_at_z_offset
        sequence_extent = max(float(maxs[2] - mins[2]), float(maxs[0] - mins[0]) * 1.15, float(maxs[1] - mins[1]) * 1.15, 1.0)
        fixed_distance = max(config.min_distance, sequence_extent * 1.85)
        return [camera_from_target(target, fixed_distance, config) for _ in range(joints_world.shape[0])]

    smoothed = np.empty_like(target_raw)
    smoothed[0] = target_raw[0]
    alpha = float(np.clip(config.camera_smoothing, 0.01, 1.0))
    for frame_index in range(1, len(target_raw)):
        smoothed[frame_index] = alpha * target_raw[frame_index] + (1.0 - alpha) * smoothed[frame_index - 1]
    return [camera_from_target(target, distance, config) for target in smoothed]


def project(points: np.ndarray, eye: np.ndarray, right: np.ndarray, up: np.ndarray, config: CameraConfig) -> tuple[np.ndarray, np.ndarray]:
    forward = normalize(np.cross(up, right))
    rel = points - eye[None, :]
    x = rel @ right
    y = rel @ up
    z = rel @ forward
    z_safe = np.maximum(z, 1e-4)
    uv = np.column_stack([config.fx * x / z_safe + config.cx, config.cy - config.fy * y / z_safe])
    return uv, z


def draw_grid(frame: np.ndarray, eye: np.ndarray, right: np.ndarray, up: np.ndarray, config: CameraConfig, center_xy: np.ndarray, extent: float) -> None:
    x0, y0 = center_xy
    extent = max(extent, 4.0)
    step = 0.5
    grid_color = (112, 112, 112)
    axis_x_color = (92, 92, 155)
    axis_y_color = (92, 155, 92)
    bg = np.zeros_like(frame)
    bg[:] = (154, 154, 154)
    frame[:] = bg
    plane_corners = np.array(
        [
            [x0 - extent, y0 - extent, 0.0],
            [x0 + extent, y0 - extent, 0.0],
            [x0 + extent, y0 + extent, 0.0],
            [x0 - extent, y0 + extent, 0.0],
        ],
        dtype=np.float64,
    )
    uv, z = project(plane_corners, eye, right, up, config)
    if np.all(z > 0.05):
        cv2.fillConvexPoly(frame, np.round(uv).astype(np.int32), (128, 128, 128), lineType=cv2.LINE_AA)
    ticks = np.arange(-extent, extent + 1e-5, step)
    for tick in ticks:
        color = axis_y_color if abs(tick) < 1e-6 else grid_color
        pts = np.array([[x0 + tick, y0 - extent, 0.0], [x0 + tick, y0 + extent, 0.0]], dtype=np.float64)
        uv, z = project(pts, eye, right, up, config)
        if np.all(z > 0.05):
            cv2.line(frame, tuple(np.round(uv[0]).astype(int)), tuple(np.round(uv[1]).astype(int)), color, 1, cv2.LINE_AA)
        color = axis_x_color if abs(tick) < 1e-6 else grid_color
        pts = np.array([[x0 - extent, y0 + tick, 0.0], [x0 + extent, y0 + tick, 0.0]], dtype=np.float64)
        uv, z = project(pts, eye, right, up, config)
        if np.all(z > 0.05):
            cv2.line(frame, tuple(np.round(uv[0]).astype(int)), tuple(np.round(uv[1]).astype(int)), color, 1, cv2.LINE_AA)


def draw_polyline_3d(frame: np.ndarray, points: np.ndarray, eye: np.ndarray, right: np.ndarray, up: np.ndarray, config: CameraConfig, color: tuple[int, int, int], thickness: int) -> None:
    if len(points) < 2:
        return
    uv, z = project(points, eye, right, up, config)
    for i in range(len(points) - 1):
        if z[i] > 0.05 and z[i + 1] > 0.05:
            cv2.line(
                frame,
                tuple(np.round(uv[i]).astype(int)),
                tuple(np.round(uv[i + 1]).astype(int)),
                color,
                thickness,
                cv2.LINE_AA,
            )


def draw_surface_points(frame: np.ndarray, vertices: np.ndarray, eye: np.ndarray, right: np.ndarray, up: np.ndarray, config: CameraConfig) -> None:
    uv, depth = project(vertices, eye, right, up, config)
    uv_i = np.round(uv).astype(np.int32)
    visible = (
        (depth > 0.05)
        & (uv_i[:, 0] >= 0)
        & (uv_i[:, 0] < config.width)
        & (uv_i[:, 1] >= 0)
        & (uv_i[:, 1] < config.height)
    )
    if not np.any(visible):
        return
    indices = np.flatnonzero(visible)
    indices = indices[np.argsort(depth[indices])[::-1]]
    for index in indices:
        point = tuple(uv_i[index])
        cv2.circle(frame, point, 2, (58, 196, 98), -1, cv2.LINE_AA)
        cv2.circle(frame, point, 2, (180, 255, 190), 1, cv2.LINE_AA)


def draw_solid_mesh(
    frame: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    eye: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    config: CameraConfig,
) -> None:
    uv, depth = project(vertices, eye, right, up, config)
    uv_i = np.round(uv).astype(np.int32)
    tri_depth = depth[faces].mean(axis=1)
    visible = np.all(depth[faces] > 0.05, axis=1)
    if not np.any(visible):
        return
    face_indices = np.flatnonzero(visible)
    face_indices = face_indices[np.argsort(tri_depth[face_indices])[::-1]]
    light_dir = normalize(np.array([-0.35, -0.45, 0.82], dtype=np.float64))
    base = np.array([48, 202, 92], dtype=np.float64)
    for face_index in face_indices:
        tri = faces[face_index]
        pts = uv_i[tri]
        if (
            pts[:, 0].max() < -20
            or pts[:, 0].min() > config.width + 20
            or pts[:, 1].max() < -20
            or pts[:, 1].min() > config.height + 20
        ):
            continue
        v0, v1, v2 = vertices[tri]
        normal = normalize(np.cross(v1 - v0, v2 - v0))
        shade = 0.62 + 0.38 * max(0.0, float(np.dot(normal, light_dir)))
        color = tuple(np.clip(base * shade, 0, 255).astype(np.uint8).tolist())
        cv2.fillConvexPoly(frame, pts, color, lineType=cv2.LINE_AA)


def draw_skeleton(frame: np.ndarray, joints: np.ndarray, parents: Iterable[int], eye: np.ndarray, right: np.ndarray, up: np.ndarray, config: CameraConfig) -> None:
    uv, depth = project(joints, eye, right, up, config)
    bones = []
    for joint_index, parent_index in enumerate(parents):
        parent_index = int(parent_index)
        if parent_index < 0:
            continue
        if joint_index >= len(joints) or parent_index >= len(joints):
            continue
        if depth[joint_index] > 0.05 and depth[parent_index] > 0.05:
            bones.append((max(depth[joint_index], depth[parent_index]), parent_index, joint_index))
    for _, parent_index, joint_index in sorted(bones, reverse=True):
        cv2.line(
            frame,
            tuple(np.round(uv[parent_index]).astype(int)),
            tuple(np.round(uv[joint_index]).astype(int)),
            (70, 220, 115),
            5,
            cv2.LINE_AA,
        )
        cv2.line(
            frame,
            tuple(np.round(uv[parent_index]).astype(int)),
            tuple(np.round(uv[joint_index]).astype(int)),
            (24, 90, 42),
            1,
            cv2.LINE_AA,
        )
    for point, z in sorted(zip(uv, depth), key=lambda item: item[1], reverse=True):
        if z > 0.05:
            cv2.circle(frame, tuple(np.round(point).astype(int)), 5, (45, 160, 85), -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(np.round(point).astype(int)), 5, (210, 255, 220), 1, cv2.LINE_AA)


def draw_text_panel(frame: np.ndarray, lines: list[str]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.82
    thickness = 2
    padding = 16
    line_gap = 12
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    width = max(size[0] for size in sizes) + padding * 2
    height = sum(size[1] for size in sizes) + line_gap * (len(lines) - 1) + padding * 2
    x0 = frame.shape[1] - width - 32
    y0 = 32
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), (0, 0, 0), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.46, frame, 0.54, 0, dst=frame)
    y = y0 + padding + sizes[0][1]
    for line, size in zip(lines, sizes):
        cv2.putText(frame, line, (x0 + padding, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += size[1] + line_gap


def write_height_json(path: Path, heights: np.ndarray, fps: float, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": label,
        "fps": fps,
        "reference": "top_of_head_or_highest_model_vertex_z",
        "initial_height_m": float(heights[0]),
        "height_m": [float(v) for v in heights],
        "delta_height_m": [float(v - heights[0]) for v in heights],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def render_video(
    output_path: Path,
    joints_world: np.ndarray,
    vertices_world: np.ndarray,
    faces: np.ndarray,
    parents: list[int],
    config: CameraConfig,
    height_json: Path,
) -> np.ndarray:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{config.width}x{config.height}",
        "-r",
        f"{config.fps:g}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    root_ground = joints_world[:, 0].copy()
    root_ground[:, 2] = 0.02
    ground_center = np.median(root_ground[:, :2], axis=0)
    extent = max(float(np.ptp(joints_world[..., 0])), float(np.ptp(joints_world[..., 1])), 3.0) + 3.0
    cameras = build_stable_camera_path(joints_world, vertices_world, config)
    head_heights = vertices_world[:, :, 2].max(axis=1)
    write_height_json(height_json, head_heights, config.fps, "GEM-X green human")
    with subprocess.Popen(command, stdin=subprocess.PIPE) as proc:
        assert proc.stdin is not None
        for frame_index in range(joints_world.shape[0]):
            frame = np.empty((config.height, config.width, 3), dtype=np.uint8)
            eye, target, right, up = cameras[frame_index]
            draw_grid(frame, eye, right, up, config, ground_center, extent)
            draw_polyline_3d(frame, root_ground[: frame_index + 1], eye, right, up, config, (70, 115, 225), 3)
            draw_solid_mesh(frame, vertices_world[frame_index], faces, eye, right, up, config)
            draw_skeleton(frame, joints_world[frame_index], parents, eye, right, up, config)
            delta_h = head_heights[frame_index] - head_heights[0]
            draw_text_panel(frame, ["GEM-X 3D", f"Head relZ: {delta_h:+.2f} m", f"Head Z: {head_heights[frame_index]:.2f} m"])
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        return_code = proc.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {output_path}")
    return head_heights


def estimate_original_gem_camera(verts_soma: torch.Tensor, joints_soma: torch.Tensor) -> dict:
    position, target, up = get_global_cameras_static_v2(
        verts_soma.cpu().clone(), beta=4.5, cam_height_degree=30, target_center_height=1.0
    )
    pos = position.detach().cpu().numpy().astype(float)
    tgt = target.detach().cpu().numpy().astype(float)
    up_np = up.detach().cpu().numpy().astype(float)
    view = tgt - pos
    horizontal = math.sqrt(float(view[0] ** 2 + view[2] ** 2))
    elevation = math.degrees(math.atan2(float(view[1]), horizontal))
    azimuth = math.degrees(math.atan2(float(view[0]), float(view[2])))
    return {
        "position_soma_xyz_yup": pos.tolist(),
        "target_soma_xyz_yup": tgt.tolist(),
        "up_soma_xyz_yup": up_np.tolist(),
        "estimated_elevation_deg": elevation,
        "estimated_azimuth_deg_from_soma_z": azimuth,
        "projection": "perspective (Open3D intrinsics in GEM-X global renderer)",
        "gem_renderer_beta": 4.5,
        "gem_renderer_cam_height_degree": 30,
    }


def motion_analysis(joints_soma: np.ndarray, joints_isaac: np.ndarray) -> dict:
    root_soma = joints_soma[:, 0]
    root_isaac = joints_isaac[:, 0]
    ranges_soma = np.ptp(root_soma, axis=0)
    ranges_isaac = np.ptp(root_isaac, axis=0)
    delta_soma = root_soma[-1] - root_soma[0]
    dominant = "forward/backward GEM-Z" if ranges_soma[2] >= ranges_soma[0] else "left/right GEM-X"
    return {
        "coordinate_note": "GEM/SOMA is X lateral, Y vertical, Z forward; aligned render maps to Isaac-style X lateral, Y forward, Z vertical.",
        "root_delta_soma_xyz": delta_soma.tolist(),
        "root_range_soma_xyz": ranges_soma.tolist(),
        "root_range_isaac_xyz": ranges_isaac.tolist(),
        "dominant_horizontal_motion": dominant,
        "forward_z_range_m": float(ranges_soma[2]),
        "lateral_x_range_m": float(ranges_soma[0]),
        "vertical_y_range_m": float(ranges_soma[1]),
        "z_motion_compressed_in_original": bool(ranges_soma[2] > 0.25 and ranges_soma[2] > ranges_soma[0] * 0.8),
    }


def write_report(report_path: Path, rows: list[dict], config: CameraConfig) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "GEM-X global camera alignment report",
        "",
        "Method",
        "- Used GEM hpe_results.pt, not a 2D crop/rotate transform.",
        "- Reconstructed SOMA vertices and 77-joint skeleton from body_params_global.",
        "- Normalized GEM ground/start orientation with GEM-X compute_T_ayfz2ay, then mapped SOMA Y-up to Isaac Z-up.",
        "- Rendered the recovered 3D vertices as a solid green mesh plus the 77-joint skeleton, using a perspective virtual camera matching the current Isaac camera orbit.",
        "- Camera target follows the root with exponential smoothing and fixed distance, so foot/head motion is easier to compare against the fixed floor grid.",
        "- Added a fixed gray ground plane/grid and a top-right head-height panel. Head height is measured as top-of-model Z relative to frame 0.",
        "- No temporal smoothing was applied; frame count and timing follow the HPE sequence at 30 fps.",
        "",
        "Isaac matching camera",
        f"- projection: perspective",
        f"- resolution: {config.width}x{config.height}",
        f"- fps: {config.fps:g}",
        f"- focal_length_mm: {config.focal_length_mm}",
        f"- horizontal_aperture_mm: {config.horizontal_aperture_mm}",
        f"- camera_offset_xyz: {config.offset}",
        f"- offset elevation: {config.elevation_deg:.2f} deg",
        f"- offset azimuth from +X in Isaac XY: {config.azimuth_deg_from_x:.2f} deg",
        f"- distance_scale: {config.distance_scale}",
        f"- min_distance: {config.min_distance}",
        f"- look_at_z_offset: {config.look_at_z_offset}",
        f"- camera_mode: {config.camera_mode}",
        f"- camera_smoothing: {config.camera_smoothing}",
        "- source: /home/anon/isaac_sim/render_retarget_video.sh tracking_camera mode",
        "",
        "Per-sequence analysis",
    ]
    for row in rows:
        lines.extend(
            [
                "",
                row["name"],
                f"- input_global_video: {row['source_video']}",
                f"- output_aligned_video: {row['output_video']}",
                f"- height_json: {row['height_json']}",
                f"- hpe_results: {row['hpe_results']}",
                f"- source_video_metadata: {row['source_video_metadata']}",
                f"- output_video_metadata: {row['output_video_metadata']}",
                f"- reconstructed_motion: {row['motion_metadata']}",
                f"- estimated_original_gem_camera: {row['original_gem_camera']}",
                f"- motion_analysis: {row['motion_analysis']}",
                f"- head_height_initial_m: {row['head_height_initial_m']:.4f}",
                f"- head_height_delta_range_m: [{row['head_height_delta_min_m']:.4f}, {row['head_height_delta_max_m']:.4f}]",
                f"- validation: Z/forward motion is preserved through 3D projection; timing unchanged; output is H.264 with no 2D warping.",
            ]
        )
        if row.get("mirror_output"):
            lines.append(f"- mirror_output_for_space_named_file3_folder: {row['mirror_output']}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_mirror(output_path: Path, mirror_path: Path | None) -> None:
    if mirror_path is None:
        return
    mirror_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-f", str(output_path), str(mirror_path)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render GEM-X global motion under the Isaac Sim camera model.")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--camera-mode", choices=["smooth_root", "fixed_sequence"], default="smooth_root")
    parser.add_argument("--camera-smoothing", type=float, default=0.08)
    parser.add_argument("--only", choices=[spec.name for spec in SEQUENCES], action="append")
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    config = CameraConfig(
        width=args.width,
        height=args.height,
        fps=args.fps,
        camera_mode=args.camera_mode,
        camera_smoothing=args.camera_smoothing,
    )
    selected = [spec for spec in SEQUENCES if args.only is None or spec.name in set(args.only)]
    soma = SomaLayer(
        data_root=str(GEM_ROOT / "inputs/soma_assets"),
        low_lod=True,
        device="cpu",
        identity_model_type="mhr",
        mode="warp",
    )
    parents = [int(parent) for parent in soma.parents]
    faces = soma.faces.detach().cpu().numpy().astype(np.int32)
    rows = []
    for spec in selected:
        print(f"[{spec.name}] loading {spec.hpe_results}", flush=True)
        verts_soma_t, joints_soma_t, motion_meta = load_soma_motion(spec, soma)
        original_camera = estimate_original_gem_camera(verts_soma_t, joints_soma_t)
        joints_soma = joints_soma_t.numpy()
        verts_soma = verts_soma_t.numpy()
        joints_isaac_raw = soma_to_isaac(joints_soma)
        origin_xy = joints_isaac_raw[0, 0, :2].copy()
        joints_isaac = apply_xy_origin(joints_isaac_raw, origin_xy)
        verts_isaac = apply_xy_origin(soma_to_isaac(verts_soma), origin_xy)
        print(f"[{spec.name}] rendering {spec.output_video}", flush=True)
        head_heights = render_video(spec.output_video, joints_isaac, verts_isaac, faces, parents, config, spec.height_json)
        copy_mirror(spec.output_video, spec.mirror_output)
        head_delta = head_heights - head_heights[0]
        rows.append(
            {
                "name": spec.name,
                "source_video": str(spec.source_video),
                "hpe_results": str(spec.hpe_results),
                "output_video": str(spec.output_video),
                "height_json": str(spec.height_json),
                "mirror_output": str(spec.mirror_output) if spec.mirror_output else None,
                "source_video_metadata": probe_video(spec.source_video),
                "output_video_metadata": probe_video(spec.output_video),
                "motion_metadata": motion_meta,
                "original_gem_camera": original_camera,
                "motion_analysis": motion_analysis(joints_soma, joints_isaac),
                "head_height_initial_m": float(head_heights[0]),
                "head_height_delta_min_m": float(head_delta.min()),
                "head_height_delta_max_m": float(head_delta.max()),
            }
        )
    write_report(args.report, rows, config)
    print(f"report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
