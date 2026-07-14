#!/usr/bin/env python3
"""Export a retargeted Unitree G1 CSV as a kinematic mesh USD replay.

This is for retarget *visual validation*: it uses Newton FK to place each real
G1 mesh link every frame, with no gravity or dynamics, so the robot cannot fall
because no balance controller is being simulated.
"""

from __future__ import annotations

import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import newton
import numpy as np
import trimesh
import warp as wp
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux
from scipy.spatial.transform import Rotation


def load_csv_as_joint_q(csv_path: Path):
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        raw = np.asarray([[float(value) for value in row] for row in reader], dtype=np.float32)

    joint_q = np.zeros((raw.shape[0], raw.shape[1]), dtype=np.float32)
    joint_q[:, 0:3] = raw[:, 1:4] * 0.01
    joint_q[:, 3:7] = Rotation.from_euler("xyz", raw[:, 4:7], degrees=True).as_quat().astype(np.float32)
    joint_q[:, 7:] = np.deg2rad(raw[:, 7:])
    return header, joint_q


def parse_bvh_motion(bvh_path: Path):
    names: list[str] = []
    parents: list[int] = []
    offsets: list[list[float]] = []
    channels: list[list[str]] = []
    stack: list[int] = []
    pending_joint: int | None = None
    in_end_site = False

    with bvh_path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    motion_index = None
    for line_index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if line == "MOTION":
            motion_index = line_index
            break
        if not line:
            continue
        if line.startswith("ROOT ") or line.startswith("JOINT "):
            names.append(line.split(maxsplit=1)[1])
            parents.append(stack[-1] if stack else -1)
            offsets.append([0.0, 0.0, 0.0])
            channels.append([])
            pending_joint = len(names) - 1
            in_end_site = False
        elif line.startswith("End Site"):
            pending_joint = None
            in_end_site = True
        elif line == "{":
            if pending_joint is not None:
                stack.append(pending_joint)
                pending_joint = None
        elif line == "}":
            if in_end_site:
                in_end_site = False
            elif stack:
                stack.pop()
        elif line.startswith("OFFSET ") and stack and not in_end_site:
            offsets[stack[-1]] = [float(value) * 0.01 for value in line.split()[1:4]]
        elif line.startswith("CHANNELS ") and stack and not in_end_site:
            parts = line.split()
            channels[stack[-1]] = parts[2 : 2 + int(parts[1])]

    if motion_index is None:
        raise ValueError(f"BVH has no MOTION section: {bvh_path}")

    frame_count = int(lines[motion_index + 1].split(":", maxsplit=1)[1])
    frame_time = float(lines[motion_index + 2].split(":", maxsplit=1)[1])
    values = np.asarray(
        [[float(value) for value in line.split()] for line in lines[motion_index + 3 : motion_index + 3 + frame_count]],
        dtype=np.float32,
    )
    return names, np.asarray(parents, dtype=np.int32), np.asarray(offsets, dtype=np.float32), channels, values, frame_time


def evaluate_bvh_joint_positions(bvh_path: Path):
    names, parents, offsets, channels, values, frame_time = parse_bvh_motion(bvh_path)
    channel_offsets = np.cumsum([0] + [len(joint_channels) for joint_channels in channels])
    positions = np.zeros((values.shape[0], len(names), 3), dtype=np.float32)

    for frame_index, frame_values in enumerate(values):
        world_rotations: list[Rotation] = [Rotation.identity()] * len(names)
        for joint_index, joint_channels in enumerate(channels):
            channel_values = frame_values[channel_offsets[joint_index] : channel_offsets[joint_index + 1]]
            local_position = offsets[joint_index].astype(np.float64).copy()
            rotation_axes = []
            rotation_values = []
            for channel_name, channel_value in zip(joint_channels, channel_values):
                if channel_name == "Xposition":
                    local_position[0] = float(channel_value) * 0.01
                elif channel_name == "Yposition":
                    local_position[1] = float(channel_value) * 0.01
                elif channel_name == "Zposition":
                    local_position[2] = float(channel_value) * 0.01
                elif channel_name.endswith("rotation"):
                    rotation_axes.append(channel_name[0])
                    rotation_values.append(float(channel_value))

            local_rotation = (
                Rotation.from_euler("".join(rotation_axes), rotation_values, degrees=True)
                if rotation_axes
                else Rotation.identity()
            )
            parent_index = parents[joint_index]
            if parent_index < 0:
                positions[frame_index, joint_index] = local_position.astype(np.float32)
                world_rotations[joint_index] = local_rotation
            else:
                parent_rotation = world_rotations[parent_index]
                positions[frame_index, joint_index] = (
                    positions[frame_index, parent_index].astype(np.float64) + parent_rotation.apply(local_position)
                ).astype(np.float32)
                world_rotations[joint_index] = parent_rotation * local_rotation

    return names, parents, positions, frame_time


def parse_mjcf_meshes(mjcf_path: Path):
    root = ET.parse(mjcf_path).getroot()
    mesh_files = {}
    for mesh in root.iter("mesh"):
        name = mesh.get("name")
        filename = mesh.get("file")
        if name and filename:
            mesh_files[name] = filename

    geom_infos = []
    for body in root.iter("body"):
        body_name = body.get("name")
        if not body_name:
            continue
        for geom in body.findall("geom"):
            if geom.get("type") != "mesh":
                continue
            mesh_name = geom.get("mesh")
            if mesh_name not in mesh_files:
                continue
            pos = np.fromstring(geom.get("pos", "0 0 0"), sep=" ", dtype=np.float64)
            quat_wxyz = np.fromstring(geom.get("quat", "1 0 0 0"), sep=" ", dtype=np.float64)
            rgba = np.fromstring(geom.get("rgba", "0.72 0.72 0.72 1.0"), sep=" ", dtype=np.float64)
            geom_infos.append((body_name, mesh_files[mesh_name], pos, quat_wxyz, rgba))
    return geom_infos


def load_visual_geometries(asset_root: Path):
    mjcf_path = asset_root / "mjcf" / "g1_29dof_rev_1_0.xml"
    meshes_dir = asset_root / "meshes"
    geometries = []
    seen = set()
    for body_name, mesh_file, local_pos, quat_wxyz, rgba in parse_mjcf_meshes(mjcf_path):
        key = (body_name, mesh_file, tuple(local_pos), tuple(quat_wxyz))
        if key in seen:
            continue
        seen.add(key)
        mesh_path = meshes_dir / mesh_file
        if not mesh_path.exists():
            continue
        mesh = trimesh.load(str(mesh_path), force="mesh")
        if mesh.vertices.size == 0 or mesh.faces.size == 0:
            continue
        w, x, y, z = quat_wxyz
        local_rot = Rotation.from_quat([x, y, z, w])
        geometries.append(
            {
                "body_name": body_name,
                "mesh_name": mesh_path.stem,
                "vertices": np.asarray(mesh.vertices, dtype=np.float32),
                "faces": np.asarray(mesh.faces, dtype=np.int32),
                "local_pos": local_pos.astype(np.float64),
                "local_rot": local_rot,
                "color": rgba[:3].astype(np.float32),
            }
        )
    return geometries, mjcf_path


def build_newton_model(mjcf_path: Path):
    builder = newton.ModelBuilder()
    builder.add_mjcf(mjcf_path)
    model = builder.finalize()
    state = model.state()
    body_name_to_index = {}
    for index, label in enumerate(model.body_label):
        body_name_to_index[label.split("/")[-1]] = index
    return model, state, body_name_to_index


def evaluate_body_transforms(model, state, joint_q: np.ndarray):
    frames = []
    qd = wp.zeros(model.joint_dof_count, dtype=wp.float32)
    for frame_q in joint_q:
        q = np.zeros(model.joint_coord_count, dtype=np.float32)
        q[: min(len(q), len(frame_q))] = frame_q[: min(len(q), len(frame_q))]
        newton.eval_fk(model, wp.array(q, dtype=wp.float32), qd, state)
        frames.append(state.body_q.numpy().copy())
    return np.asarray(frames, dtype=np.float32)


def compute_ground_offset(geometries, body_name_to_index, body_q_frame):
    min_z = np.inf
    for geom in geometries:
        body_index = body_name_to_index.get(geom["body_name"])
        if body_index is None:
            continue
        body_pos = body_q_frame[body_index, :3].astype(np.float64)
        body_rot = Rotation.from_quat(body_q_frame[body_index, 3:7])
        final_pos = body_pos + body_rot.apply(geom["local_pos"])
        final_rot = body_rot * geom["local_rot"]
        verts = final_rot.apply(geom["vertices"].astype(np.float64)) + final_pos
        min_z = min(min_z, float(verts[:, 2].min()))
    if not np.isfinite(min_z):
        return 0.0
    return -min_z + 0.015


def per_frame_ground_safety_shift(
    geometries, body_name_to_index, body_q, ground_scope: str, sole_clearance: float
):
    if ground_scope == "feet":
        selected = [
            geometry for geometry in geometries
            if geometry["body_name"] in {"left_ankle_roll_link", "right_ankle_roll_link"}
        ]
        if not selected:
            raise ValueError("feet ground scope found no ankle-roll visual meshes")
    else:
        selected = geometries

    shifts = np.zeros(len(body_q), dtype=np.float64)
    for frame, frame_body_q in enumerate(body_q):
        minimum = np.inf
        for geometry in selected:
            body_index = body_name_to_index.get(geometry["body_name"])
            if body_index is None:
                continue
            body_pos = frame_body_q[body_index, :3].astype(np.float64)
            body_rot = Rotation.from_quat(frame_body_q[body_index, 3:7])
            final_pos = body_pos + body_rot.apply(geometry["local_pos"])
            final_rot = body_rot * geometry["local_rot"]
            vertices = final_rot.apply(geometry["vertices"].astype(np.float64)) + final_pos
            minimum = min(minimum, float(vertices[:, 2].min()))
        if not np.isfinite(minimum):
            raise ValueError(f"no ground-scope geometry for frame {frame}")
        # The CSV postprocessor already aligns support soles. This is an
        # upward-only final guard against numeric penetration, not a second
        # grounding pass that would erase its contact decisions.
        shifts[frame] = max(0.0, float(sole_clearance) - minimum)
    return shifts


def add_scene(stage: Usd.Stage):
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.CreateSizeAttr(1.0)
    ground.AddScaleOp().Set(Gf.Vec3f(6.0, 6.0, 0.02))
    ground.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, -0.02))
    ground.GetDisplayColorAttr().Set([Gf.Vec3f(0.18, 0.18, 0.19)])

    key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    key.CreateIntensityAttr(900.0)
    key.AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 0.0, 30.0))
    dome = UsdLux.DomeLight.Define(stage, "/World/FillLight")
    dome.CreateIntensityAttr(250.0)

    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.AddTranslateOp().Set(Gf.Vec3f(1.8, -4.2, 1.45))
    camera.AddRotateXYZOp().Set(Gf.Vec3f(72.0, 0.0, 23.0))
    camera.CreateFocalLengthAttr(24.0)


def add_flattened_human_motion(stage: Usd.Stage, bvh_path: Path, fps: float, max_frames: int):
    names, parents, positions, frame_time = evaluate_bvh_joint_positions(bvh_path)
    frame_count = min(max_frames, positions.shape[0])
    bone_pairs = [(parent_index, joint_index) for joint_index, parent_index in enumerate(parents) if parent_index >= 0]
    ground_z = 0.025

    root = UsdGeom.Xform.Define(stage, "/World/HumanMotion_FlattenedGround")
    root.GetPrim().CreateAttribute("human:sourceBvh", Sdf.ValueTypeNames.String).Set(str(bvh_path))
    root.GetPrim().CreateAttribute("human:projection", Sdf.ValueTypeNames.String).Set(
        "xy_only_all_joint_z_set_to_ground"
    )
    root.GetPrim().CreateAttribute("human:heightAvailable", Sdf.ValueTypeNames.Bool).Set(False)
    root.GetPrim().CreateAttribute("human:sourceFrameTime", Sdf.ValueTypeNames.Float).Set(float(frame_time))
    root.GetPrim().CreateAttribute("human:replayFps", Sdf.ValueTypeNames.Float).Set(float(fps))

    joint_points = UsdGeom.Points.Define(stage, "/World/HumanMotion_FlattenedGround/Joints")
    joint_points_attr = joint_points.CreatePointsAttr()
    joint_points.CreateWidthsAttr([0.035] * len(names))
    joint_points.GetDisplayColorAttr().Set([Gf.Vec3f(0.15, 0.72, 0.95)])

    bone_curves = UsdGeom.BasisCurves.Define(stage, "/World/HumanMotion_FlattenedGround/Bones")
    bone_points_attr = bone_curves.CreatePointsAttr()
    bone_curves.CreateTypeAttr("linear")
    bone_curves.CreateCurveVertexCountsAttr([2] * len(bone_pairs))
    bone_curves.CreateWidthsAttr([0.012] * (len(bone_pairs) * 2))
    bone_curves.GetDisplayColorAttr().Set([Gf.Vec3f(0.98, 0.72, 0.18)])

    for frame_index in range(frame_count):
        time_code = Usd.TimeCode(frame_index)
        flat_positions = positions[frame_index].copy()
        flat_positions[:, 2] = ground_z

        joint_points_attr.Set(
            [Gf.Vec3f(float(point[0]), float(point[1]), float(point[2])) for point in flat_positions],
            time_code,
        )

        bone_points = []
        for parent_index, joint_index in bone_pairs:
            bone_points.append(flat_positions[parent_index])
            bone_points.append(flat_positions[joint_index])
        bone_points_attr.Set(
            [Gf.Vec3f(float(point[0]), float(point[1]), float(point[2])) for point in bone_points],
            time_code,
        )

    print(
        f"Added flattened human motion: {bvh_path} frames={frame_count} "
        f"joints={len(names)} bones={len(bone_pairs)} height_available=False"
    )


def export_usd(
    csv_path: Path,
    output_usd: Path,
    fps: float,
    human_bvh: Path | None = None,
    ground_aligned: bool = False,
    ground_scope: str = "feet",
    sole_clearance: float = 0.0,
):
    _, joint_q = load_csv_as_joint_q(csv_path)
    asset_root = newton.utils.download_asset("unitree_g1")
    geometries, mjcf_path = load_visual_geometries(asset_root)
    model, state, body_name_to_index = build_newton_model(mjcf_path)
    body_q = evaluate_body_transforms(model, state, joint_q)
    if ground_aligned:
        z_offsets = per_frame_ground_safety_shift(
            geometries, body_name_to_index, body_q, ground_scope, sole_clearance
        )
    else:
        z_offsets = np.full(len(body_q), compute_ground_offset(geometries, body_name_to_index, body_q[0]))

    output_usd.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_usd))
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(int(joint_q.shape[0] - 1))
    stage.SetTimeCodesPerSecond(fps)
    stage.SetFramesPerSecond(fps)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    add_scene(stage)
    if human_bvh is not None:
        add_flattened_human_motion(stage, human_bvh, fps, joint_q.shape[0])

    root = UsdGeom.Xform.Define(stage, "/World/G1_KinematicReplay")
    xform_ops = []
    for geom_index, geom in enumerate(geometries):
        geom_path = f"/World/G1_KinematicReplay/{geom['body_name']}_{geom_index:03d}_{geom['mesh_name']}"
        xform = UsdGeom.Xform.Define(stage, geom_path)
        xformable = UsdGeom.Xformable(xform.GetPrim())
        translate_op = xformable.AddTranslateOp()
        orient_op = xformable.AddOrientOp()
        xform_ops.append((geom, translate_op, orient_op))

        mesh_prim = UsdGeom.Mesh.Define(stage, f"{geom_path}/mesh")
        mesh_prim.CreatePointsAttr([Gf.Vec3f(*vertex.tolist()) for vertex in geom["vertices"]])
        face_vertex_counts = [3] * len(geom["faces"])
        mesh_prim.CreateFaceVertexCountsAttr(face_vertex_counts)
        mesh_prim.CreateFaceVertexIndicesAttr(geom["faces"].reshape(-1).astype(int).tolist())
        mesh_prim.CreateSubdivisionSchemeAttr("none")
        mesh_prim.GetDisplayColorAttr().Set([Gf.Vec3f(*geom["color"].tolist())])

    for frame_index in range(joint_q.shape[0]):
        time_code = Usd.TimeCode(frame_index)
        frame_body_q = body_q[frame_index]
        for geom, translate_op, orient_op in xform_ops:
            body_index = body_name_to_index.get(geom["body_name"])
            if body_index is None:
                continue
            body_pos = frame_body_q[body_index, :3].astype(np.float64)
            body_rot = Rotation.from_quat(frame_body_q[body_index, 3:7])
            final_pos = body_pos + body_rot.apply(geom["local_pos"])
            final_pos[2] += z_offsets[frame_index]
            final_rot = body_rot * geom["local_rot"]
            quat = final_rot.as_quat()
            translate_op.Set(Gf.Vec3f(float(final_pos[0]), float(final_pos[1]), float(final_pos[2])), time_code)
            orient_op.Set(
                Gf.Quatf(
                    float(quat[3]),
                    Gf.Vec3f(float(quat[0]), float(quat[1]), float(quat[2])),
                ),
                time_code,
            )

    root.GetPrim().CreateAttribute("retarget:sourceCsv", Sdf.ValueTypeNames.String).Set(str(csv_path))
    root.GetPrim().CreateAttribute("retarget:mode", Sdf.ValueTypeNames.String).Set("kinematic_newton_fk_no_gravity")
    root.GetPrim().CreateAttribute("retarget:groundAligned", Sdf.ValueTypeNames.Bool).Set(bool(ground_aligned))
    root.GetPrim().CreateAttribute("retarget:groundScope", Sdf.ValueTypeNames.String).Set(ground_scope)
    stage.GetRootLayer().Save()
    print(f"Saved kinematic G1 replay USD: {output_usd}")
    print(
        f"frames={joint_q.shape[0]} fps={fps} visual_geometries={len(geometries)} "
        f"ground_aligned={ground_aligned} z_shift=[{z_offsets.min():.4f},{z_offsets.max():.4f}]"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-usd", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--human-bvh",
        type=Path,
        default=None,
        help="Optional SOMA BVH to overlay as flattened human ground-plane motion.",
    )
    parser.add_argument("--ground-aligned", action="store_true")
    parser.add_argument("--ground-scope", choices=("feet", "all"), default="feet")
    parser.add_argument("--sole-clearance", type=float, default=0.0)
    args = parser.parse_args()
    export_usd(
        args.csv,
        args.output_usd,
        args.fps,
        args.human_bvh,
        ground_aligned=args.ground_aligned,
        ground_scope=args.ground_scope,
        sole_clearance=args.sole_clearance,
    )


if __name__ == "__main__":
    main()
