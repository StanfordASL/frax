"""BONES-SEED motion capture -> Unitree G1 retargeting: BVH loading, and the
retargeter

The task and tuning follow `examples/ik/lafan1/lafan1_g1.py`, retargeting SEED's
SOMA skeletons directly, rather than via NVIDIA's soma-retargeter
(https://github.com/NVIDIA/soma-retargeter), which made SEED's own G1 retargets.
The rotation offsets come from soma-retargeter's zero pose, where SOMA matches
G1's zero configuration

SEED's "uniform" clips share one SOMA skeleton for all actors, so the scales and
floor offset are constants
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy.spatial.transform import Rotation

from frax.ik import (
    DiffIKSolver,
    FloorConstraint,
    FrameTask,
    GlobalIKSolver,
    JointLimits,
    JointTask,
    Retargeter,
    SelfCollisionConstraint,
    load_bvh,
)
from frax.utils.rotation_utils import quat_wxyz_to_rmat, rmat_to_quat_wxyz

SEED_FPS = 120
# BVH (Y-up, facing +z) -> world (Z-up, facing +x)
BVH_TO_WORLD = Rotation.from_euler("z", 90, degrees=True).as_matrix() @ np.array(
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
)


def load_seed(filepath: str | Path) -> tuple[np.ndarray, list[str]]:
    """Global bone poses from a SEED (SOMA) BVH file (load_bvh). Raises
    ValueError if any frame has NaNs, as a few SEED clips do

    Returns:
        poses (np.ndarray): Poses [position, WXYZ quaternion], shape (T, B, 7)
        names (list[str]): The B bone names
    """
    return load_bvh(filepath, BVH_TO_WORLD)


# fmt: off
# (G1 joint whose child link is tracked, SOMA bone, position weight, rotation
# weight, rotation offset (WXYZ, applied in the bone frame)). The root comes
# first. Weights are LAFAN1's, but the chest's rotation is heavier (100 vs. 50), so
# arm error left by the shoulder limits doesn't leak into the waist
RETARGETING_TABLE = (
    ("yaw_joint",                "Hips",         50,  50, (0.4992, 0.4992, 0.5008, 0.5008)),    # pelvis
    ("left_hip_yaw_joint",       "LeftLeg",      0,   10, (-0.4544, -0.4544, 0.5418, 0.5418)),
    ("left_knee_joint",          "LeftShin",     0,   10, (-0.5, -0.5, 0.5, 0.5)),
    ("left_ankle_roll_joint",    "LeftFoot",     100, 10, (0.6953, 0.6953, -0.1284, -0.1284)),
    ("right_hip_yaw_joint",      "RightLeg",     0,   10, (0.5418, 0.5418, 0.4544, 0.4544)),
    ("right_knee_joint",         "RightShin",    0,   10, (0.5, 0.5, 0.5, 0.5)),
    ("right_ankle_roll_joint",   "RightFoot",    100, 10, (0.1284, 0.1284, 0.6953, 0.6953)),
    ("waist_pitch_joint",        "Chest",        0,   100, (0.4991, 0.4991, 0.5009, 0.5009)),    # torso_link
    ("left_shoulder_yaw_joint",  "LeftArm",      0,   50, (-0.4520, 0.5659, 0.5131, -0.4607)),
    ("left_elbow_joint",         "LeftForeArm",  0,   10, (-0.6450, 0.7636, -0.0177, -0.0228)),
    ("left_wrist_yaw_joint",     "LeftHand",     10,  10, (-0.6925, 0.7195, -0.0266, -0.0459)),
    ("right_shoulder_yaw_joint", "RightArm",     0,   50, (-0.5131, 0.4607, -0.4520, 0.5659)),
    ("right_elbow_joint",        "RightForeArm", 0,   10, (0.0177, 0.0228, -0.6450, 0.7636)),
    ("right_wrist_yaw_joint",    "RightHand",    10,  10, (0.0266, 0.0459, -0.6925, 0.7195)),
)
# Arm joint limits narrowed from G1's (rad), as in GMR's g1_mocap_29dof.xml (see
# lafan1_g1.py)
RETARGETING_JOINT_LIMITS = {
    "left_shoulder_pitch_joint":  (-3.089, 1.149),
    "left_shoulder_roll_joint":   (-0.6, 2.252),
    "left_shoulder_yaw_joint":    (-1.4, 2.0),
    "left_elbow_joint":           (-1.047, 1.7),
    "right_shoulder_pitch_joint": (-3.089, 1.149),
    "right_shoulder_roll_joint":  (-2.252, 0.6),
    "right_shoulder_yaw_joint":   (-2.0, 1.4),
    "right_elbow_joint":          (-1.047, 1.7),
}
# fmt: on
G1_JOINT_NAMES = tuple(row[0] for row in RETARGETING_TABLE)
SEED_BONES = tuple(row[1] for row in RETARGETING_TABLE)
WEIGHT_POS = np.array([row[2] for row in RETARGETING_TABLE], dtype=float)
WEIGHT_ROT = np.array([row[3] for row in RETARGETING_TABLE], dtype=float)
# Pitch and roll (deg) of SOMA's foot bone from a planted sole, removed so planted
# feet map to flat G1 soles
FOOT_TILT = (0.2, 2.3)
# Forward pitch (deg) of SOMA's pelvis and chest bones when standing upright,
# removed so upright maps to upright
TORSO_PITCH = {"Hips": -0.5, "Chest": 3.4}


def _leveled_rot_offsets() -> np.ndarray:
    """The table's rotation offsets (n, 3, 3), with TORSO_PITCH and FOOT_TILT
    removed"""
    offsets = Rotation.from_quat(
        [row[4] for row in RETARGETING_TABLE], scalar_first=True
    ).as_matrix()
    for bone, pitch in TORSO_PITCH.items():
        level = Rotation.from_euler("y", -pitch, degrees=True).as_matrix()
        offsets[SEED_BONES.index(bone)] = offsets[SEED_BONES.index(bone)] @ level
    for bone, sign in (("LeftFoot", 1.0), ("RightFoot", -1.0)):
        # In the G1 foot's frame: pitch (y) the toes down, roll (x) the outer edge up
        tilt = (FOOT_TILT[0], sign * FOOT_TILT[1])
        level = Rotation.from_euler("yx", tilt, degrees=True).as_matrix()
        offsets[SEED_BONES.index(bone)] = offsets[SEED_BONES.index(bone)] @ level
    return offsets


ROT_OFFSETS = _leveled_rot_offsets()
# G1's leg length: hip pitch -> knee -> ankle roll joints (m)
G1_LEG = 0.3409 + 0.3176
# G1's upper body reach: hip center -> shoulder roll joint, plus shoulder roll ->
# elbow -> wrist yaw joints (m)
G1_UPPER = 0.4162 + 0.1845 + 0.1843
# The SOMA skeleton's, likewise, from its rest offsets (m)
SOMA_LEG = 0.8553
SOMA_UPPER = 1.0973
LEG_SCALE = G1_LEG / SOMA_LEG
UPPER_SCALE = G1_UPPER / SOMA_UPPER
# G1's pelvis frame origin relative to the center of its hip joints, in the
# pelvis frame (m)
PELVIS_OFFSET = np.array([0.0, 0.0, 0.1027])
# G1's ankle roll joint height above its sole (m)
G1_ANKLE_HEIGHT = 0.035
# SOMA's ankle height over planted feet, the median over SEED's actors (m)
SOMA_ANKLE_HEIGHT = 0.0671
# All targets drop by this much (m), so planted feet meet the floor
FLOOR_OFFSET = LEG_SCALE * SOMA_ANKLE_HEIGHT - G1_ANKLE_HEIGHT
_HIPS = np.array([SEED_BONES.index(b) for b in ("LeftLeg", "RightLeg")])
_FEET = np.array([SEED_BONES.index(b) for b in ("LeftFoot", "RightFoot")])
SHOULDER_JOINTS = tuple(
    f"{side}_shoulder_{axis}_joint"
    for side in ("left", "right")
    for axis in ("pitch", "roll", "yaw")
)
WRIST_JOINTS = tuple(
    f"{side}_wrist_{axis}_joint"
    for side in ("left", "right")
    for axis in ("roll", "pitch", "yaw")
)
# Tracking gain (1/s): closes the error with a 1/60 s time constant. Below the
# frame rate (a full step per frame), for smoother motion at a small tracking cost
GAIN = 60.0
# Damping (see lafan1_g1.py), independent of the frame rate
DAMPING = 10.0
SHOULDER_DAMPING = 1000.0
WRIST_DAMPING = 100.0
SELF_COLLISION_ROWS = 64
SELF_COLLISION_GAIN = GAIN
# Shoulder roll (rad) the arms are calibrated at: human arms hanging straight
# down map to G1's rolled out by this much, clear of its hips
ARM_ABDUCTION = 0.2
# Weight of a posture task pulling the shoulders toward the calibration pose, so
# raised arms don't drift into G1's flipped shoulder solution (pitch near -180)
SHOULDER_POSTURE_WEIGHT = 16.0
# Orientation smoothing kernel standard deviation (s), for offline retargeting
SMOOTH_ROT = 0.05
# Global IK for the first frame (global_ik_init): starts, iterations, and
# cost tolerance (see GlobalIKSolver), for clips that start in poses the zero
# configuration converges badly from (e.g. hands behind the back). The tolerance
# keeps the default solution unless another is clearly better
GLOBAL_IK_SAMPLES = 32
GLOBAL_IK_ITERS = 100
GLOBAL_IK_COST_TOL = 5e4
# retarget.py --refine redoes by global IK the clips whose first frame has a
# link this far off target (deg)
GLOBAL_IK_FIRST_FRAME_ERROR = 45.0


def rot_offsets(robot, arm_abduction: float = ARM_ABDUCTION) -> np.ndarray:
    """ROT_OFFSETS, with the arms calibrated against G1 with its shoulders
    rolled out by arm_abduction (rad), instead of its zero configuration"""
    q0 = np.zeros(robot.nq)
    q0[3] = 1.0  # Identity root quaternion
    q = q0.copy()
    for side, sign in (("left", 1.0), ("right", -1.0)):
        joint = robot.joint_name_to_index[f"{side}_shoulder_roll_joint"]
        q[robot.velocity_to_configuration_index[joint]] = sign * arm_abduction
    frames = np.array([robot.joint_name_to_index[name] for name in G1_JOINT_NAMES])
    R0 = np.asarray(robot.joint_to_world_transforms(jnp.asarray(q0)))[frames, :3, :3]
    R = np.asarray(robot.joint_to_world_transforms(jnp.asarray(q)))[frames, :3, :3]
    return ROT_OFFSETS @ np.swapaxes(R0, -1, -2) @ R


def preprocess(frame: dict, rot_offsets: np.ndarray = ROT_OFFSETS) -> dict:
    """Human bone poses (n, 7), in RETARGETING_TABLE order -> scaled target
    positions (n, 3) and rotation matrices (n, 3, 3) for the tracked G1 links

    The hip center and feet scale about the origin by LEG_SCALE. The pelvis
    sits above the hip center by PELVIS_OFFSET, and the upper body scales about
    the hip center by UPPER_SCALE. All targets then drop by FLOOR_OFFSET. The
    default, uncalibrated ROT_OFFSETS give the human's own bone orientations
    (as evaluate.py scores against)"""
    pos = frame["poses"][:, :3]
    rmats = jax.vmap(quat_wxyz_to_rmat)(frame["poses"][:, 3:]) @ rot_offsets
    hips = jnp.mean(pos[_HIPS], axis=0)
    targets = LEG_SCALE * hips + UPPER_SCALE * (pos - hips)
    targets = targets.at[0].set(LEG_SCALE * hips + rmats[0] @ PELVIS_OFFSET)
    targets = targets.at[_FEET].set(LEG_SCALE * pos[_FEET])
    return {"frame_pos": targets.at[:, 2].add(-FLOOR_OFFSET), "frame_rmat": rmats}


def make_retargeter(
    robot,
    floor: bool = True,
    self_collision: bool = True,
    arm_abduction: float = ARM_ABDUCTION,
    weight_pos: np.ndarray = WEIGHT_POS,
    weight_rot: np.ndarray = WEIGHT_ROT,
    damping: float = DAMPING,
    shoulder_damping: float = SHOULDER_DAMPING,
    wrist_damping: float = WRIST_DAMPING,
    global_ik_init: bool = False,
    shoulder_posture_weight: float = SHOULDER_POSTURE_WEIGHT,
) -> Retargeter:
    """Input frames ({"poses": human bone poses (n, 7)}) -> G1 configuration

    Args:
        robot (Humanoid): G1 from `load_g1()`
        floor (bool): Keep the robot's collision spheres above the floor
        self_collision (bool): Self-collision avoidance (the G1's self-collision pairs)
        arm_abduction (float): Shoulder roll (rad) that the arms are calibrated
            at (see ARM_ABDUCTION)
        weight_pos, weight_rot (np.ndarray): Per-frame weights, in
            RETARGETING_TABLE order
        damping (float): Damping on the velocity
        shoulder_damping (float): Damping on the shoulder joints' velocities
        wrist_damping (float): Damping on the wrist joints' velocities
        global_ik_init (bool): Solve the first frame by global IK (see
            GLOBAL_IK_SAMPLES)
        shoulder_posture_weight (float): Weight pulling the shoulder joints
            toward the calibration pose (see SHOULDER_POSTURE_WEIGHT)
    """
    offsets = rot_offsets(robot, arm_abduction)
    shoulder_posture = jnp.array(
        [
            (1.0 if side == "left" else -1.0) * arm_abduction * (axis == "roll")
            for side in ("left", "right")
            for axis in ("pitch", "roll", "yaw")
        ]
    )

    def preprocess_frame(frame: dict) -> dict:
        return preprocess(frame, offsets) | {"shoulder_posture": shoulder_posture}

    def initial_q(frame: dict) -> Array:
        """The robot's default joint positions, with the root at the first target
        pelvis pose"""
        ref = preprocess_frame(frame)
        root_quat = rmat_to_quat_wxyz(ref["frame_rmat"][0])
        q = jnp.asarray(robot.default_configuration).at[:3].set(ref["frame_pos"][0])
        return q.at[3:7].set(root_quat)

    task = FrameTask(
        robot,
        [robot.joint_name_to_index[name] for name in G1_JOINT_NAMES],
        gain_pos=GAIN,
        gain_rot=GAIN,
        weight_pos=weight_pos,
        weight_rot=weight_rot,
    )
    posture_task = JointTask(
        robot,
        [robot.joint_name_to_index[name] for name in SHOULDER_JOINTS],
        "shoulder_posture",
        GAIN,
        shoulder_posture_weight,
    )
    q_min = np.array(robot.actuated_joint_lower_limits, dtype=float)
    q_max = np.array(robot.actuated_joint_upper_limits, dtype=float)
    for name, (lower, upper) in RETARGETING_JOINT_LIMITS.items():
        i = robot.joint_name_to_index[name] - robot.nv_floating
        q_min[i], q_max[i] = lower, upper
    limits = JointLimits(robot, q_min=q_min, q_max=q_max)
    constraints = [limits]
    if floor:
        constraints.append(FloorConstraint(robot, gain=GAIN))
    if self_collision:
        constraints.append(
            SelfCollisionConstraint(
                robot, max_rows=SELF_COLLISION_ROWS, gain=SELF_COLLISION_GAIN
            )
        )
    regularization = np.full(robot.nv, damping)
    for joints, value in (
        (SHOULDER_JOINTS, shoulder_damping),
        (WRIST_JOINTS, wrist_damping),
    ):
        regularization[[robot.joint_name_to_index[name] for name in joints]] = value
    solver = DiffIKSolver(
        robot,
        [task, posture_task],
        constraints,
        regularization=regularization,
    )
    if not global_ik_init:
        return Retargeter(solver, preprocess_frame, initial_q)
    # Converged to the Retargeter's tolerance (1e-3), then converged onto again
    global_ik = GlobalIKSolver(
        solver,
        num_samples=GLOBAL_IK_SAMPLES,
        max_iters=GLOBAL_IK_ITERS,
        tol=1e-3,
        cost_tol=GLOBAL_IK_COST_TOL,
    )
    return Retargeter(
        solver,
        preprocess_frame,
        initial_q,
        converge_max_iters=GLOBAL_IK_ITERS,
        global_ik_init=global_ik,
    )
