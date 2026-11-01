"""LAFAN1 -> G1 (examples/ik/lafan1): BVH loading, and retargeting of bone poses
made from G1 poses"""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from frax.ik import FloorConstraint, OnlineRetargeter

sys.path.append(str(Path(__file__).parents[2] / "examples/ik/lafan1"))
from lafan1_g1 import (
    FLOOR_OFFSET,
    G1_JOINT_NAMES,
    LAFAN1_BONES,
    LAFAN1_FPS,
    LEG_SCALE,
    PELVIS_OFFSET,
    ROT_OFFSETS,
    UPPER_SCALE,
    WEIGHT_POS,
    load_lafan1,
    make_retargeter,
)

LEG = """    JOINT {side}Foot
    {{
        OFFSET 0.0 -50.0 0.0
        CHANNELS 3 Zrotation Yrotation Xrotation
        JOINT {side}Toe
        {{
            OFFSET 0.0 0.0 10.0
            CHANNELS 3 Zrotation Yrotation Xrotation
            End Site
            {{
                OFFSET 0.0 0.0 5.0
            }}
        }}
    }}
"""
# Hips with two legs (foot, toe). Frame 0: root at 100 cm height. Frame 1: root
# yawed 90 deg about BVH's Y (up)
BVH = (
    "HIERARCHY\nROOT Hips\n{\n    OFFSET 0.0 0.0 0.0\n"
    + "    CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation\n"
    + LEG.format(side="Left")
    + LEG.format(side="Right")
    + "}\nMOTION\nFrames: 2\nFrame Time: 0.033333\n"
    + "".join(
        root + " 0 0 0" * 4 + "\n" for root in ("0 100 0 0 0 0", "0 100 0 0 90 0")
    )
)


def test_load_lafan1(tmp_path):
    """Y-up cm -> Z-up m, global poses by FK, and GMR's foot-mod bones"""
    path = tmp_path / "clip.bvh"
    path.write_text(BVH)
    poses, names = load_lafan1(path)
    assert names == [
        "Hips",
        "LeftFoot",
        "LeftToe",
        "RightFoot",
        "RightToe",
        "LeftFootMod",
        "RightFootMod",
    ]
    assert poses.shape == (2, 7, 7)
    np.testing.assert_allclose(poses[0, 0, :3], [0.0, 0.0, 1.0])  # Z-up, m
    np.testing.assert_allclose(poses[0, 1, :3], [0.0, 0.0, 0.5])
    # BVH +Z (forward) is world -Y
    np.testing.assert_allclose(poses[0, 2, :3], [0.0, -0.1, 0.5], atol=1e-12)
    # Yaw 90 deg about BVH's Y turns BVH +Z into BVH +X (world +X)
    np.testing.assert_allclose(poses[1, 2, :3], [0.1, 0.0, 0.5], atol=1e-12)
    np.testing.assert_allclose(poses[:, 5, :3], poses[:, 1, :3])  # Foot position,
    np.testing.assert_allclose(poses[:, 5, 3:], poses[:, 2, 3:])  # toe orientation


def _bone_poses(g1, qs):
    """Human bone poses (T, 14, 7) that preprocess to the G1 frame poses at qs
    (the tracked joint frames):
    preprocessing scaled, with the rotation offsets undone. The hip bones are
    put at the human root"""
    idxs = [g1.joint_name_to_index[name] for name in G1_JOINT_NAMES]
    tfs = np.stack([np.asarray(g1.joint_to_world_transforms(q))[idxs] for q in qs])
    pos = tfs[..., :3, 3] + FLOOR_OFFSET
    hips = pos[:, :1] - tfs[:, :1, :3, :3] @ PELVIS_OFFSET  # G1's hip center
    root = hips / LEG_SCALE
    human = root + (pos - hips) / UPPER_SCALE
    for bone in ("LeftUpLeg", "RightUpLeg"):
        human[:, LAFAN1_BONES.index(bone)] = root[:, 0]
    for bone in ("LeftFootMod", "RightFootMod"):
        i = LAFAN1_BONES.index(bone)
        human[:, i] = pos[:, i] / LEG_SCALE
    pos = human
    rmats = tfs[..., :3, :3] @ np.swapaxes(ROT_OFFSETS, -1, -2)
    quats = Rotation.from_matrix(rmats.reshape(-1, 3, 3)).as_quat(scalar_first=True)
    return np.concatenate([pos, quats.reshape(len(qs), -1, 4)], axis=-1)


@pytest.mark.parametrize("self_collision", [False, True])
def test_retargets_g1_motion(g1, g1_q0, self_collision):
    """Bone poses from a slow G1 motion are tracked, online as offline"""
    dt = 1.0 / LAFAN1_FPS
    robot = g1
    t = np.arange(10) * dt
    qs = np.tile(g1_q0, (len(t), 1))
    qs[:, 0] += 0.3 * t  # Walk forward
    elbows = [
        robot.joint_name_to_index[f"{side}_elbow_joint"] - robot.nv_floating + 7
        for side in ("left", "right")
    ]
    qs[:, elbows] += 0.3 * np.sin(2 * t)[:, None]
    frames = _bone_poses(robot, qs)

    retargeter = make_retargeter(robot, self_collision=self_collision)
    q_offline, infos, _ = retargeter.retarget_clip(frames, dt)
    assert np.all(infos.qp_converged)
    online = OnlineRetargeter(retargeter)
    online_q = np.stack([online.step(frame, dt) for frame in frames])
    np.testing.assert_allclose(q_offline, online_q, atol=1e-9)

    # The position-tracked frames follow the G1 motion (to 1 mm)
    idxs = [
        robot.joint_name_to_index[name]
        for name, w in zip(G1_JOINT_NAMES, WEIGHT_POS)
        if w > 0
    ]
    tracked = np.stack(
        [np.asarray(robot.joint_to_world_transforms(q))[idxs, :3, 3] for q in q_offline]
    )
    target = np.stack(
        [np.asarray(robot.joint_to_world_transforms(q))[idxs, :3, 3] for q in qs]
    )
    assert np.max(np.linalg.norm(tracked - target, axis=-1)) < 1e-3


def test_stays_above_floor(g1, g1_q0):
    """Bone poses from a G1 motion sunk below the floor are retargeted above it,
    online as offline, and the floor constraint is inactive when standing on it"""
    dt = 1.0 / LAFAN1_FPS
    floor = FloorConstraint(g1)
    assert np.min(floor.barrier(jnp.asarray(g1_q0))) > -1e-3
    qs = np.tile(g1_q0, (20, 1))
    qs[:, 2] -= np.linspace(0.0, 0.05, len(qs))  # Sink 5 cm
    frames = _bone_poses(g1, qs)

    retargeter = make_retargeter(g1)
    q_offline, _, _ = retargeter.retarget_clip(frames, dt)
    heights = np.stack([np.asarray(floor.barrier(q)) for q in q_offline])
    assert np.min(heights) > -2e-3
    online = OnlineRetargeter(retargeter)
    online_q = np.stack([online.step(frame, dt) for frame in frames])
    np.testing.assert_allclose(q_offline, online_q, atol=1e-9)
    # Without it, the robot follows the targets down (the soles start 2 cm up)
    sunk, _, _ = make_retargeter(g1, floor=False).retarget_clip(frames, dt)
    assert np.min(floor.barrier(sunk[-1])) < -0.025
