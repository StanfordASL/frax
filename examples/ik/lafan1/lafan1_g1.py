"""Retargeting configuration for LAFAN1 -> Unitree G1

Many elements and tuning parameters inspired by GMR:
https://github.com/YanjieZe/GMR

NOTE ON AGENTIC TUNING:
Coding agents like Claude and Codex are quite good at
tuning parameters and tasks, given a well-posed set of
metrics to hill-climb on, and a well-designed library.
The code below was primarily agent-written, but I will
try to clean this up to make it more understandable in
the future. In general, after a long back-and-forth
tuning process, this set of tuning parameters was found
to give good results on the metrics I cared about
(avoiding artifacts at the floor like penetration/skid/
floating, reducing jerkiness between timesteps, avoiding
local minima, avoiding self-collision, etc.)
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
    JointLimits,
    JointTask,
    Retargeter,
    SelfCollisionConstraint,
    load_bvh,
)
from frax.utils.rotation_utils import quat_wxyz_to_rmat, rmat_to_quat_wxyz

LAFAN1_FPS = 30
DATA_DIR = Path(__file__).parents[3] / "data/lafan1"
# Retargeted clips (retarget.py)
OUT_DIR = Path(__file__).parents[3] / "data/lafan1_frax"
Y_UP_TO_Z_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])


def load_lafan1(filepath: str | Path) -> tuple[np.ndarray, list[str]]:
    """Global bone poses from a LAFAN1 BVH file

    Like GMR, adds "LeftFootMod" / "RightFootMod" bones: the foot position with
    the toe orientation

    Returns:
        poses (np.ndarray): Poses [position, WXYZ quaternion], shape (T, B, 7)
        names (list[str]): The B bone names
    """
    poses, names = load_bvh(filepath, Y_UP_TO_Z_UP)
    idx = {name: i for i, name in enumerate(names)}
    feet = [
        np.concatenate([poses[:, idx[foot], :3], poses[:, idx[toe], 3:]], axis=-1)
        for foot, toe in (("LeftFoot", "LeftToe"), ("RightFoot", "RightToe"))
    ]
    poses = np.concatenate([poses, np.stack(feet, axis=1)], axis=1)
    return poses, names + ["LeftFootMod", "RightFootMod"]


# fmt: off
# (G1 joint whose child link is tracked, LAFAN1 bone, position weight, rotation
# weight, rotation offset (WXYZ, applied in the bone frame)). The root comes
# first. Weights and rotation offsets are GMR's, but the torso's rotation is
# heavier (75 vs. 50), so arm error doesn't leak into the waist
RETARGETING_TABLE = (
    ("yaw_joint",                "Hips",         50,  50, (0.5, 0.5, 0.5, 0.5)),    # pelvis
    ("left_hip_yaw_joint",       "LeftUpLeg",    0,   10, (-0.5, 0.5, 0.5, -0.5)),
    ("left_knee_joint",          "LeftLeg",      0,   10, (-0.5, 0.5, 0.5, -0.5)),
    ("left_ankle_roll_joint",    "LeftFootMod",  100, 10, (-0.70710678, 0.70710678, 0.0, 0.0)),
    ("right_hip_yaw_joint",      "RightUpLeg",   0,   10, (-0.5, 0.5, 0.5, -0.5)),
    ("right_knee_joint",         "RightLeg",     0,   10, (-0.5, 0.5, 0.5, -0.5)),
    ("right_ankle_roll_joint",   "RightFootMod", 100, 10, (-0.70710678, 0.70710678, 0.0, 0.0)),
    ("waist_pitch_joint",        "Spine2",       0,   75, (0.5, 0.5, 0.5, 0.5)),    # torso_link
    ("left_shoulder_yaw_joint",  "LeftArm",      0,   50, (0.5, 0.5, -0.5, -0.5)),
    ("left_elbow_joint",         "LeftForeArm",  0,   10, (0.70710678, 0.70710678, 0.0, 0.0)),
    ("left_wrist_yaw_joint",     "LeftHand",     10,  10, (0.70710678, 0.70710678, 0.0, 0.0)),
    ("right_shoulder_yaw_joint", "RightArm",     0,   50, (0.5, 0.5, -0.5, -0.5)),
    ("right_elbow_joint",        "RightForeArm", 0,   10, (0.70710678, 0.70710678, 0.0, 0.0)),
    ("right_wrist_yaw_joint",    "RightHand",    10,  10, (0.70710678, 0.70710678, 0.0, 0.0)),
)
# Arm joint limits narrowed from G1's (rad), as in GMR's g1_mocap_29dof.xml, so
# the arms can't flip into G1's second shoulder solution (pitch and yaw ~150 deg)
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
LAFAN1_BONES = tuple(row[1] for row in RETARGETING_TABLE)
WEIGHT_POS = np.array([row[2] for row in RETARGETING_TABLE], dtype=float)
WEIGHT_ROT = np.array([row[3] for row in RETARGETING_TABLE], dtype=float)
ROT_OFFSETS = Rotation.from_quat(
    [row[4] for row in RETARGETING_TABLE], scalar_first=True
).as_matrix()
# G1's leg length: hip pitch -> knee -> ankle roll joints (m)
G1_LEG = 0.3409 + 0.3176
# G1's upper body reach: hip center -> shoulder roll joint, plus shoulder roll ->
# elbow -> wrist yaw joints (m)
G1_UPPER = 0.4162 + 0.1845 + 0.1843
# LAFAN1's, likewise (m). Hip center -> shoulder is the 90th percentile over
# LAFAN1, since its rest pose isn't upright
LAFAN1_LEG = 0.4350 + 0.4237
LAFAN1_UPPER = 0.5375 + 0.3300 + 0.2520
LEG_SCALE = G1_LEG / LAFAN1_LEG
UPPER_SCALE = G1_UPPER / LAFAN1_UPPER
# G1's pelvis frame origin relative to the center of its hip joints, in the
# pelvis frame (m)
PELVIS_OFFSET = np.array([0.0, 0.0, 0.1027])
# All targets drop by the scaled LAFAN1 ankle height (0.077 m) minus G1's
# (0.035 m), so the feet meet the floor
FLOOR_OFFSET = np.array([0.0, 0.0, LEG_SCALE * 0.077 - 0.035])
_HIPS = np.array([LAFAN1_BONES.index(b) for b in ("LeftUpLeg", "RightUpLeg")])
_FEET = np.array([LAFAN1_BONES.index(b) for b in ("LeftFootMod", "RightFootMod")])
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
LEG_JOINTS = tuple(
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee")
)
# Tracking gain (1/s): closes the error with a 1/30 s time constant. At 30 fps
# that is a full Gauss-Newton step per frame, the most gain * dt <= 1 allows
GAIN = 30.0
# Damping on the velocity, against large alternating steps near singularities
DAMPING = 10.0
# G1's shoulder is singular with the arm out to the side, where pitch and yaw
# turn about the same axis. Damping >= (weight / 2)^2 = 625 for the upper arm
# keeps the shoulder from stepping further than the arm's error
SHOULDER_DAMPING = 1000.0
# Otherwise the wrists do most of the self-collision avoiding, then snap back
WRIST_DAMPING = 100.0
# Otherwise the hips and knees are the cheapest joints to move, so the legs yield
# to the arms in self-collisions (e.g. a hand sweeping past a knee), then snap
# back in a step too large for the linearized floor constraint
LEG_DAMPING = 100.0
# Shoulder roll (rad) the arms are calibrated at (as in seed_g1.py): the arms
# map to G1's rolled out by this much more than the human's, clear of its hips
ARM_ABDUCTION = 0.2
# Weight of a posture task pulling the shoulders toward SHOULDER_POSTURE, so
# raised arms don't drift into G1's flipped shoulder solution (as in
# seed_g1.py). 0 to not use one
SHOULDER_POSTURE_WEIGHT = 16.0
# Shoulder (pitch, roll, yaw) posture of the left arm (rad), mirrored for the
# right: about the median over LAFAN1 without the arm calibration, arms hanging
# by the sides, rolled out by ARM_ABDUCTION
SHOULDER_POSTURE = (-0.2, 0.4 + ARM_ABDUCTION, -0.55)
# Self-collision rows kept per step (the closest pairs), above the most pairs
# near contact at once over LAFAN1
SELF_COLLISION_ROWS = 64
# Self-collision CBF gain (1/s): lower gains drag the limbs, higher ones overshoot
SELF_COLLISION_GAIN = GAIN
# Orientation smoothing kernel standard deviation (s), for offline retargeting
SMOOTH_ROT = 0.05


def rot_offsets(robot, arm_abduction: float) -> np.ndarray:
    """ROT_OFFSETS, with the arms calibrated against G1 with its shoulders
    rolled out by arm_abduction (rad), instead of its zero configuration (as in
    seed_g1.py)"""
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


def preprocess(human_poses: Array, rot_offsets: np.ndarray = ROT_OFFSETS) -> dict:
    """Human bone poses (n, 7), in RETARGETING_TABLE order -> scaled target
    positions (n, 3) and rotation matrices (n, 3, 3) for the tracked G1 links

    The hip center and feet scale about the origin by LEG_SCALE. The pelvis
    sits above the hip center by PELVIS_OFFSET, and the upper body scales about
    the hip center by UPPER_SCALE. All targets then drop by FLOOR_OFFSET"""
    pos = human_poses[:, :3]
    rmats = jax.vmap(quat_wxyz_to_rmat)(human_poses[:, 3:]) @ rot_offsets
    hips = jnp.mean(pos[_HIPS], axis=0)
    targets = LEG_SCALE * hips + UPPER_SCALE * (pos - hips)
    targets = targets.at[0].set(LEG_SCALE * hips + rmats[0] @ PELVIS_OFFSET)
    targets = targets.at[_FEET].set(LEG_SCALE * pos[_FEET])
    # FrameTask's default reference keys
    return {"frame_pos": targets - FLOOR_OFFSET, "frame_rmat": rmats}


def make_retargeter(
    robot,
    floor: bool = True,
    self_collision: bool = False,
    weight_pos: np.ndarray = WEIGHT_POS,
    weight_rot: np.ndarray = WEIGHT_ROT,
    damping: float = DAMPING,
    shoulder_damping: float = SHOULDER_DAMPING,
    wrist_damping: float = WRIST_DAMPING,
    leg_damping: float = LEG_DAMPING,
    shoulder_posture_weight: float = SHOULDER_POSTURE_WEIGHT,
    shoulder_posture: tuple[float, float, float] = SHOULDER_POSTURE,
    arm_abduction: float = ARM_ABDUCTION,
) -> Retargeter:
    """Human bone poses (n, 7) -> G1 configuration

    Args:
        robot (Humanoid): G1 from `load_g1()`
        floor (bool): Keep the robot's collision spheres above the floor
        self_collision (bool): Self-collision avoidance (the G1's self-collision pairs)
        weight_pos, weight_rot (np.ndarray): Per-frame weights, in
            RETARGETING_TABLE order. Default to GMR's
        damping (float): Damping on the velocity
        shoulder_damping (float): Damping on the shoulder joints' velocities
        wrist_damping (float): Damping on the wrist joints' velocities
        leg_damping (float): Damping on the hip and knee joints' velocities
        shoulder_posture_weight (float): Weight pulling the shoulder joints
            toward shoulder_posture (see SHOULDER_POSTURE_WEIGHT)
        shoulder_posture (tuple[float, float, float]): Left shoulder (pitch,
            roll, yaw) posture (rad), mirrored for the right
        arm_abduction (float): Shoulder roll (rad) that the arms are calibrated
            at (see ARM_ABDUCTION). 0 for the uncalibrated ROT_OFFSETS
    """
    offsets = rot_offsets(robot, arm_abduction) if arm_abduction else ROT_OFFSETS
    pitch, roll, yaw = shoulder_posture
    posture = jnp.array([pitch, roll, yaw, pitch, -roll, -yaw])

    def preprocess_frame(human_poses: Array) -> dict:
        return preprocess(human_poses, offsets) | {"shoulder_posture": posture}

    def initial_q(human_poses: Array) -> Array:
        """The robot's default joint positions, with the root at the first target
        pelvis pose"""
        ref = preprocess(human_poses)
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
        (LEG_JOINTS, leg_damping),
    ):
        regularization[[robot.joint_name_to_index[name] for name in joints]] = value
    tasks = [task]
    if shoulder_posture_weight > 0:
        tasks.append(
            JointTask(
                robot,
                [robot.joint_name_to_index[name] for name in SHOULDER_JOINTS],
                "shoulder_posture",
                GAIN,
                shoulder_posture_weight,
            )
        )
    solver = DiffIKSolver(robot, tasks, constraints, regularization=regularization)
    return Retargeter(solver, preprocess_frame, initial_q)
