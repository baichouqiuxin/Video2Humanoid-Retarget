#!/usr/bin/env python3
"""Export GEM-X SOMA predictions to a BVH consumable by soma-retargeter."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _parse_soma_bvh_skeleton(bvh_path: Path):
    names: list[str] = []
    parents: list[int] = []
    stack: list[int] = []
    pending_joint: int | None = None
    in_end_site = False

    with bvh_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "MOTION":
                break
            if not line:
                continue
            if line.startswith("ROOT ") or line.startswith("JOINT "):
                name = line.split(maxsplit=1)[1]
                names.append(name)
                parents.append(stack[-1] if stack else -1)
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

    if len(names) != 78:
        raise ValueError(f"Expected 78 SOMA joints in {bvh_path}, found {len(names)}")
    return names, np.asarray(parents, dtype=np.int32)


def _build_offsets(gemx_root: Path, identity_coeffs, scale_params, parents: np.ndarray):
    sys.path.insert(0, str(gemx_root))
    sys.path.insert(0, str(gemx_root / "third_party" / "soma"))
    from gem.utils.soma_utils.soma_layer import SomaLayer

    original_cwd = Path.cwd()
    try:
        # GEM-X SomaLayer expects inputs/soma_assets relative to the project root.
        import os

        os.chdir(gemx_root)
        soma = SomaLayer(
            data_root="inputs/soma_assets",
            low_lod=True,
            device="cpu",
            identity_model_type="mhr",
            mode="warp",
        )
        identity = torch.as_tensor(_to_numpy(identity_coeffs), dtype=torch.float32)[:1]
        scale = torch.as_tensor(_to_numpy(scale_params), dtype=torch.float32)[:1]
        with torch.no_grad():
            positions_77 = soma.get_skeleton(identity, scale)[0].cpu().numpy()
    finally:
        import os

        os.chdir(original_cwd)

    positions = np.concatenate([np.zeros((1, 3), dtype=np.float32), positions_77], axis=0)
    rig_data = np.load(gemx_root / "inputs" / "soma_assets" / "SOMA_neutral.npz", allow_pickle=False)
    orient = Rotation.from_matrix(rig_data["t_pose_world"][..., :3, :3])

    offsets = np.zeros_like(positions, dtype=np.float32)
    rest_quats = np.zeros((len(parents), 4), dtype=np.float32)
    for joint_index, parent_index in enumerate(parents):
        if parent_index < 0:
            offsets[joint_index] = positions[joint_index]
            rest_rot = Rotation.identity()
        else:
            world_offset = positions[joint_index] - positions[parent_index]
            offsets[joint_index] = orient[parent_index].inv().apply(world_offset)
            rest_rot = orient[parent_index].inv() * orient[joint_index]
        rest_quats[joint_index] = rest_rot.as_quat().astype(np.float32)

    return offsets, rest_quats, orient


def _body_params_to_bvh_values(body_params: dict, parents: np.ndarray, offsets, rest_quats, orient):
    global_orient = _to_numpy(body_params["global_orient"]).astype(np.float32)
    body_pose = _to_numpy(body_params["body_pose"]).astype(np.float32)
    transl = _to_numpy(body_params["transl"]).astype(np.float32)

    num_frames = global_orient.shape[0]
    if body_pose.shape != (num_frames, 76 * 3):
        raise ValueError(f"Expected body_pose {(num_frames, 228)}, got {body_pose.shape}")

    all_rotvecs = np.concatenate([global_orient[:, None, :], body_pose.reshape(num_frames, 76, 3)], axis=1)
    quats = Rotation.from_rotvec(all_rotvecs.reshape(-1, 3)).as_quat().reshape(num_frames, 77, 4)

    local_positions = np.broadcast_to(offsets[None, :, :], (num_frames, len(parents), 3)).copy()
    local_positions[:, 1, :] = transl

    local_quats = np.broadcast_to(rest_quats[None, :, :], (num_frames, len(parents), 4)).copy()
    identity_quat = Rotation.identity().as_quat()
    for joint_index, parent_index in enumerate(parents):
        if parent_index < 0:
            local_quats[:, joint_index, :] = identity_quat
        else:
            body_rot = Rotation.from_quat(quats[:, joint_index - 1, :])
            local_quats[:, joint_index, :] = (orient[parent_index].inv() * body_rot * orient[joint_index]).as_quat()

    euler_zyx = Rotation.from_quat(local_quats.reshape(-1, 4)).as_euler("ZYX", degrees=True)
    euler_zyx = euler_zyx.reshape(num_frames, len(parents), 3).astype(np.float32)
    return local_positions, euler_zyx


def _children_from_parents(parents: np.ndarray):
    children = [[] for _ in range(len(parents))]
    for joint_index, parent_index in enumerate(parents):
        if parent_index >= 0:
            children[parent_index].append(joint_index)
    return children


def _write_bvh(output_path: Path, names, parents, offsets, positions, eulers_zyx, fps: float):
    children = _children_from_parents(parents)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def write_joint(handle, joint_index: int, indent: int):
        prefix = "  " * indent
        tag = "ROOT" if parents[joint_index] < 0 else "JOINT"
        handle.write(f"{prefix}{tag} {names[joint_index]}\n")
        handle.write(f"{prefix}{{\n")
        off = offsets[joint_index] * 100.0
        handle.write(f"{prefix}  OFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}\n")
        handle.write(f"{prefix}  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation\n")
        if children[joint_index]:
            for child_index in children[joint_index]:
                write_joint(handle, child_index, indent + 1)
        else:
            handle.write(f"{prefix}  End Site\n")
            handle.write(f"{prefix}  {{\n")
            handle.write(f"{prefix}    OFFSET 0.000000 0.000000 0.000000\n")
            handle.write(f"{prefix}  }}\n")
        handle.write(f"{prefix}}}\n")

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("HIERARCHY\n")
        write_joint(handle, 0, 0)
        handle.write("MOTION\n")
        handle.write(f"Frames: {positions.shape[0]}\n")
        handle.write(f"Frame Time: {1.0 / float(fps):.6f}\n")
        for frame_index in range(positions.shape[0]):
            values = []
            for joint_index in range(len(names)):
                pos_cm = positions[frame_index, joint_index] * 100.0
                rz, ry, rx = eulers_zyx[frame_index, joint_index]
                values.extend(
                    [
                        f"{pos_cm[0]:.6f}",
                        f"{pos_cm[1]:.6f}",
                        f"{pos_cm[2]:.6f}",
                        f"{rz:.6f}",
                        f"{ry:.6f}",
                        f"{rx:.6f}",
                    ]
                )
            handle.write(" ".join(values) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpe", required=True, type=Path)
    parser.add_argument("--output-bvh", required=True, type=Path)
    parser.add_argument("--fps", default=30.0, type=float)
    parser.add_argument("--gemx-root", default=Path(os.environ.get("GEMX_ROOT", "GEM-X")), type=Path)
    parser.add_argument(
        "--reference-bvh",
        default=Path(os.environ.get("SOMA_REFERENCE_BVH", "soma_zero_frame0.bvh")),
        type=Path,
    )
    args = parser.parse_args()

    pred = torch.load(args.hpe, map_location="cpu", weights_only=False)
    body_params = pred["body_params_global"]
    names, parents = _parse_soma_bvh_skeleton(args.reference_bvh)
    offsets, rest_quats, orient = _build_offsets(
        args.gemx_root,
        body_params["identity_coeffs"],
        body_params["scale_params"],
        parents,
    )
    positions, eulers_zyx = _body_params_to_bvh_values(body_params, parents, offsets, rest_quats, orient)
    _write_bvh(args.output_bvh, names, parents, offsets, positions, eulers_zyx, args.fps)
    print(f"Saved SOMA BVH: {args.output_bvh} frames={positions.shape[0]} fps={args.fps}")


if __name__ == "__main__":
    main()
