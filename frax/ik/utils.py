"""IK and motion utilities"""

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from frax.core.robot import Robot


def _parse_bvh(filepath: str | Path):
    """Bone names, parents, local positions (T, B, 3), and local rotations
    (T, B, 3, 3)"""
    names, parents, offsets, channels, stack = [], [], [], [], []
    lines = iter(Path(filepath).read_text().splitlines())
    for line in lines:
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] in ("ROOT", "JOINT"):
            parents.append(stack[-1] if stack else -1)
            names.append(tokens[1])
        elif tokens[0] == "End":  # Skip the End Site block
            next(lines), next(lines), next(lines)
        elif tokens[0] == "{":
            stack.append(len(names) - 1)
        elif tokens[0] == "}":
            stack.pop()
        elif tokens[0] == "OFFSET":
            offsets.append([float(x) for x in tokens[1:4]])
        elif tokens[0] == "CHANNELS":
            channels.append(tokens[2:])
        elif tokens[0] == "Frame" and tokens[1] == "Time:":
            break
    data = np.array([line.split() for line in lines if line.strip()], dtype=float)
    if np.isnan(data).any():
        raise ValueError(f"{filepath} has NaN frames")
    T, B = len(data), len(names)
    local_pos = np.broadcast_to(np.array(offsets), (T, B, 3)).copy()
    local_rot = np.zeros((T, B, 3, 3))
    col = 0
    for i, chans in enumerate(channels):
        values = data[:, col : col + len(chans)]
        col += len(chans)
        rot = [(c[0], k) for k, c in enumerate(chans) if c.endswith("rotation")]
        for k, c in enumerate(chans):
            if c.endswith("position"):
                local_pos[:, i, "XYZ".index(c[0])] = values[:, k]
        local_rot[:, i] = Rotation.from_euler(
            "".join(axis for axis, _ in rot),  # Intrinsic, in channel order
            values[:, [k for _, k in rot]],
            degrees=True,
        ).as_matrix()
    return names, np.array(parents), local_pos, local_rot


def load_bvh(
    filepath: str | Path, to_world: ArrayLike | None = None, scale: float = 0.01
) -> tuple[np.ndarray, list[str]]:
    """Global bone poses from a BVH file

    Args:
        filepath (str | Path): BVH file
        to_world (ArrayLike | None): Rotation from the BVH frame to the world,
            shape (3, 3). Defaults to identity
        scale (float): BVH length unit (m). Defaults to cm

    Returns:
        tuple[np.ndarray, list[str]]: Poses [position, WXYZ quaternion], shape
            (T, B, 7), and the B bone names

    Raises:
        ValueError: If any frame has NaNs
    """
    names, parents, local_pos, local_rot = _parse_bvh(filepath)
    T, B = local_pos.shape[:2]
    rot = np.zeros((T, B, 3, 3))
    pos = np.zeros((T, B, 3))
    for i, parent in enumerate(parents):  # Parents precede their children
        if parent < 0:
            rot[:, i], pos[:, i] = local_rot[:, i], local_pos[:, i]
        else:
            rot[:, i] = rot[:, parent] @ local_rot[:, i]
            pos[:, i] = pos[:, parent] + np.einsum(
                "tij,tj->ti", rot[:, parent], local_pos[:, i]
            )
    to_world = np.eye(3) if to_world is None else np.asarray(to_world, dtype=float)
    pos = scale * pos @ to_world.T
    quat = Rotation.from_matrix((to_world @ rot).reshape(-1, 3, 3))
    quat = quat.as_quat(scalar_first=True).reshape(T, B, 4)
    return np.concatenate([pos, quat], axis=-1), names


def smooth_poses(
    poses: ArrayLike, dt: float, sigma_pos: float = 0.0, sigma_rot: float = 0.0
) -> np.ndarray:
    """Zero-phase Gaussian smoothing of a clip of poses along time,
    for offline retargeting

    Args:
        poses (ArrayLike): Poses [position, WXYZ quaternion], shape (T, ..., 7)
        dt (float): Time between poses
        sigma_pos (float): Position kernel standard deviation (s). 0 to not smooth
        sigma_rot (float): Rotation kernel standard deviation (s). 0 to not smooth

    Returns:
        np.ndarray: Smoothed poses, shape (T, ..., 7)
    """
    poses = np.array(poses, dtype=float)
    if sigma_pos > 0:
        poses[..., :3] = gaussian_filter1d(
            poses[..., :3], sigma_pos / dt, axis=0, mode="nearest"
        )
    if sigma_rot > 0:
        quat = poses[..., 3:]
        # Sign-align, so componentwise filtering approximates the geodesic mean
        flips = np.sum(quat[1:] * quat[:-1], axis=-1) < 0
        signs = np.cumprod(np.where(flips, -1.0, 1.0), axis=0)
        quat[1:] *= signs[..., None]
        quat = gaussian_filter1d(quat, sigma_rot / dt, axis=0, mode="nearest")
        poses[..., 3:] = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    return poses


def resample_poses(poses: ArrayLike, fps_in: float, fps_out: float) -> np.ndarray:
    """Resample a clip of poses to a new frame rate

    Args:
        poses (ArrayLike): Poses [position, WXYZ quaternion], shape (T, ..., 7)
        fps_in (float): Frame rate of poses
        fps_out (float): Frame rate to resample to

    Returns:
        np.ndarray: Resampled poses, shape (T_out, ..., 7), with
            T_out = floor((T - 1) * fps_out / fps_in) + 1
    """
    poses = np.asarray(poses, dtype=float)
    T = len(poses)
    assert T >= 2, "Need at least two poses to resample"
    flat = poses.reshape(T, -1, 7)
    n = flat.shape[1]

    T_out = int(np.floor((T - 1) * fps_out / fps_in + 1e-9)) + 1
    frame_times = np.arange(T_out) * fps_in / fps_out  # In input frames
    i = np.minimum(frame_times.astype(int), T - 2)
    alpha = frame_times - i
    before, after = flat[i], flat[i + 1]

    a = alpha[:, None, None]
    pos = (1 - a) * before[..., :3] + a * after[..., :3]

    rot_before = Rotation.from_quat(before[..., 3:].reshape(-1, 4), scalar_first=True)
    rot_after = Rotation.from_quat(after[..., 3:].reshape(-1, 4), scalar_first=True)
    relative = (rot_before.inv() * rot_after).as_rotvec()
    a = np.repeat(alpha, n)[:, None]
    rot = rot_before * Rotation.from_rotvec(a * relative)
    quat = rot.as_quat(scalar_first=True).reshape(T_out, n, 4)

    out = np.concatenate([pos, quat], axis=-1)
    return out.reshape(T_out, *poses.shape[1:])


def save_motion(path: str | Path, robot: Robot, q: ArrayLike, fps: float) -> None:
    """Save a motion as an .npz file with fields q (float32), fps, and the
    actuated joint_names

    Args:
        path (str | Path): Output file (.npz)
        robot (Robot): Robot model
        q (ArrayLike): Configurations, shape (T, nq)
        fps (float): Frame rate
    """
    q = np.asarray(q)
    assert q.ndim == 2 and q.shape[1] == robot.nq, (
        f"q has shape {q.shape}, expected (T, {robot.nq})"
    )
    np.savez(
        path,
        q=q.astype(np.float32),
        fps=float(fps),
        joint_names=np.array(robot.joint_names[robot.nv_floating :]),
    )
