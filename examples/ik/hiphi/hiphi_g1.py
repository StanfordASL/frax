"""Retargeting configuration for HiPHI -> Unitree G1

This tuning configuration is adapted from the BONES-SEED
frax example, and again, inspired by GMR
(https://github.com/YanjieZe/GMR) and the SOMA retargeter
(https://github.com/NVIDIA/soma-retargeter)

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

Differences from BONES-SEED, summarized by Claude:

- HiPHI's BVH bones are world-aligned in its T-pose rest pose, so
  the rotation offsets are computed (rot_offsets) by matching a
  neutral human pose (arms down, elbows bent forward) to G1's zero
  configuration, rather than tabulated
- HiPHI keeps each actor's own proportions (132 actors), so the
  position targets scale per clip from the actor's skeleton
  (skeleton_scales), rather than by one constant for all clips
- HiPHI's forearm bones twist with the forearm (and in some clips,
  with mocap artifacts), which G1 has at its wrists, so the forearm
  targets keep only their direction (untwisted)
- A few actors' bones are turned about their own length, which the
  skeleton's symmetric offsets hide in the joint positions, but not
  in the orientations (unflip, applied as the clips load)
- A few clips have two spine bones rather than five (chest)
- The floor's height varies between clips, so each clip's planted
  ankle height is measured (ground()). Planted feet level out, as
  the foot bones' tilt varies by a few degrees between actors, and
  move out to G1's (relatively wider) hip width (preprocess)
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy.ndimage import gaussian_filter1d
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
)
from frax.utils.rotation_utils import quat_wxyz_to_rmat, rmat_to_quat_wxyz

# Nominal: the BVHs' frame times are 1 / 90.0009 s (a few 1 / 90.009 s)
HIPHI_FPS = 90.0
# BVH (Y-up, facing +z) -> world (Z-up, facing +x)
BVH_TO_WORLD = Rotation.from_euler("z", 90, degrees=True).as_matrix() @ np.array(
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
)
# fmt: off
# (G1 joint whose child link is tracked, HiPHI bone, position weight, rotation
# weight, the bone's neutral pose: its world rotation (from the T-pose's, as
# extrinsic XYZ Euler angles, deg) when G1's link is at its zero configuration's).
# The root comes first. Weights are SEED's, but for the hands' orientations:
# HiPHI's hands often twist and bend past a human wrist's range (and G1's), and
# tracking them loosely keeps G1's wrists from chasing those into their limits
RETARGETING_TABLE = (
    ("yaw_joint",                "Hips",         50,  50,  (0, 0, 0)),     # pelvis
    ("left_hip_yaw_joint",       "LeftUpLeg",    0,   10,  (0, 0, 0)),
    ("left_knee_joint",          "LeftLeg",      0,   10,  (0, 0, 0)),
    ("left_ankle_roll_joint",    "LeftFoot",     100, 10,  (0, 0, 0)),
    ("right_hip_yaw_joint",      "RightUpLeg",   0,   10,  (0, 0, 0)),
    ("right_knee_joint",         "RightLeg",     0,   10,  (0, 0, 0)),
    ("right_ankle_roll_joint",   "RightFoot",    100, 10,  (0, 0, 0)),
    ("waist_pitch_joint",        "Chest",        0,   100, (0, 0, 0)),     # torso_link
    # Arms down, by the sides
    ("left_shoulder_yaw_joint",  "LeftArm",      0,   50,  (-90, 0, 0)),
    # ... with the elbows bent forward 90 deg
    ("left_elbow_joint",         "LeftForeArm",  0,   10,  (-90, -90, 0)),
    ("left_wrist_yaw_joint",     "LeftHand",     10,  4,   (-90, -90, 0)),
    ("right_shoulder_yaw_joint", "RightArm",     0,   50,  (90, 0, 0)),
    ("right_elbow_joint",        "RightForeArm", 0,   10,  (90, -90, 0)),
    ("right_wrist_yaw_joint",    "RightHand",    10,  4,   (90, -90, 0)),
)
# Arm joint limits narrowed from G1's (rad), as in GMR's g1_mocap_29dof.xml (see
# ../lafan1/lafan1_g1.py)
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
# HiPHI's spine and neck bones: Spine to Spine4 and Neck, Neck1, Head in most
# clips, but only Spine, Spine1 and Neck, Head in a few (see chest())
SPINE = ("Spine", "Spine1", "Spine2", "Spine3", "Spine4")
NECK = ("Neck", "Neck1", "Head")
HIPHI_BONES = tuple(row[1] for row in RETARGETING_TABLE)
WEIGHT_POS = np.array([row[2] for row in RETARGETING_TABLE], dtype=float)
WEIGHT_ROT = np.array([row[3] for row in RETARGETING_TABLE], dtype=float)
# Pitch and roll (deg) of HiPHI's foot bone from a planted sole, removed so
# planted feet map to flat G1 soles (medians over the planted frames of a
# 2-per-Frame-LU sample of clips)
FOOT_TILT = (1.3, 1.2)
# Forward pitch (deg) of HiPHI's pelvis and chest bones when standing upright
# (medians likewise, over frames standing still), removed so upright maps to
# upright. The chest bone leans back, along the upper back's curve
TORSO_PITCH = {"Hips": 1.2, "Chest": -12.4}
# G1's leg length: hip pitch -> knee -> ankle roll joints (m)
G1_LEG = 0.3409 + 0.3176
# G1's upper body reach: hip center -> shoulder roll joint, plus shoulder roll ->
# elbow -> wrist yaw joints (m)
G1_UPPER = 0.4162 + 0.1845 + 0.1843
# G1's pelvis frame origin relative to the center of its hip joints, in the
# pelvis frame (m)
PELVIS_OFFSET = np.array([0.0, 0.0, 0.1027])
# G1's ankle roll joint height above its sole (m)
G1_ANKLE_HEIGHT = 0.035
# HiPHI's ankle height over planted feet (m), the median over clips likewise.
# It varies (+-1.3 cm) between clips (with the floor's height in the capture,
# and the foot markers), but not with the actors' leg lengths, so it's
# measured per clip (see ground()), with this as the fallback
HIPHI_ANKLE_HEIGHT = 0.071
# A clip's ankle height is measured over its planted frames: the ankle slower
# than this (m/s), the foot within PLANTED_TILT (deg) of level (not e.g. on its
# side, lying down), and the ankle less than PLANTED_HEIGHT (m) above the 5th
# percentile height of the level feet, if it has at least MIN_PLANTED of them
# (s), and kept within ANKLE_HEIGHT_RANGE (m)
PLANTED_SPEED = 0.2
PLANTED_TILT = 30.0
PLANTED_HEIGHT = 0.03
MIN_PLANTED = 2.0
ANKLE_HEIGHT_RANGE = (0.04, 0.14)
# Foot contact, for leveling planted feet (see preprocess): full while the ankle
# is within the first of these above the clip's ankle height (m), and slower
# than the first of these (m/s), fading out by the second of each, then smoothed
# over CONTACT_SMOOTH (s)
CONTACT_HEIGHT = (0.01, 0.03)
CONTACT_SPEED = (0.2, 0.4)
# ... and the foot within the first of these of level (deg), fading out by the
# second, so feet planted steeply (e.g. on the heels, sitting with the legs out,
# or the toes, kneeling) keep their tilt
CONTACT_TILT = (20.0, 35.0)
CONTACT_SMOOTH = 0.05
# Distance of G1's ankles from its centerline in its zero configuration, where
# its legs hang straight (m). Its hips are wider than the scaled actors', so the
# feet move out by the difference, to keep G1's legs as straight as the human's
G1_ANKLE_SPREAD = 0.1185
_HIPS = np.array([HIPHI_BONES.index(b) for b in ("LeftUpLeg", "RightUpLeg")])
_FEET = np.array([HIPHI_BONES.index(b) for b in ("LeftFoot", "RightFoot")])
_ARMS = np.array([HIPHI_BONES.index(b) for b in ("LeftArm", "RightArm")])
_FOREARMS = np.array([HIPHI_BONES.index(b) for b in ("LeftForeArm", "RightForeArm")])
_SIDES = np.array([1.0, -1.0])  # Left, right
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
# Tracking gain (1/s): closes the error with a 1/45 s time constant. Below SEED's
# (60), and the frame rate, for smoother motion at a small tracking cost
GAIN = 45.0
# Damping (see ../lafan1/lafan1_g1.py), independent of the frame rate
DAMPING = 10.0
# Shoulders damped more than SEED's (1000), which halves the snaps (mostly at
# the shoulders) of HiPHI's hardest clips, for ~0.1 deg more orientation error
SHOULDER_DAMPING = 3000.0
WRIST_DAMPING = 100.0
SELF_COLLISION_ROWS = 64
SELF_COLLISION_GAIN = GAIN
# Shoulder roll (rad) the arms are calibrated at: human arms hanging straight
# down map to G1's rolled out by this much, clear of its hips
ARM_ABDUCTION = 0.2
# Weight of a posture task pulling the shoulders toward the calibration pose, so
# raised arms don't drift into G1's flipped shoulder solution (pitch near -180)
SHOULDER_POSTURE_WEIGHT = 16.0
# Weight of a (light) posture task pulling the wrists toward zero, the
# calibration pose. Where a hand target twists past wrist roll's range, the
# wrists can wind into a corner of their limits and stay there, far from the
# target, as the orientation error's gradient vanishes near 180 deg: this helps
# them back out, at no cost to tracking elsewhere
WRIST_POSTURE_WEIGHT = 1.0
# Orientation smoothing kernel standard deviation (s), for offline retargeting
SMOOTH_ROT = 0.05
# Global IK for the first frame (global_ik_init, used by retarget.py --refine):
# starts, iterations, and cost tolerance (see GlobalIKSolver)
GLOBAL_IK_SAMPLES = 32
GLOBAL_IK_ITERS = 100
GLOBAL_IK_COST_TOL = 5e4
# retarget.py --refine redoes by global IK the saved clips whose first frame has
# a link this far off target (deg)
GLOBAL_IK_FIRST_FRAME_ERROR = 45.0


def chest(names: list[str]) -> str:
    """The spine bone the shoulders hang off. Most HiPHI clips have five spine
    bones (Spine to Spine4), but a few have two (Spine, Spine1)"""
    return [b for b in SPINE if b in names][-1]


def body_parents(names: list[str]) -> dict:
    """Parents of a HiPHI skeleton's body bones (the BVH also has the fingers)"""
    spine = [b for b in SPINE if b in names]
    parents = dict(zip(spine, ["Hips"] + spine[:-1]))
    neck = [b for b in NECK if b in names]
    parents |= dict(zip(neck, spine[-1:] + neck[:-1]))
    for s in ("Left", "Right"):
        parents |= {
            f"{s}Shoulder": spine[-1],
            f"{s}Arm": f"{s}Shoulder",
            f"{s}ForeArm": f"{s}Arm",
            f"{s}Hand": f"{s}ForeArm",
            f"{s}UpLeg": "Hips",
            f"{s}Leg": f"{s}UpLeg",
            f"{s}Foot": f"{s}Leg",
            f"{s}ToeBase": f"{s}Foot",
        }
    return parents


def bone_indices(names: list[str], bones=None) -> list[int]:
    """Indices of bones (default: HIPHI_BONES) in a skeleton's names, with
    "Chest" for chest(names)"""
    bones = HIPHI_BONES if bones is None else bones
    alias = {"Chest": chest(names)}
    return [names.index(alias.get(b, b)) for b in bones]


def rest_offsets(poses: np.ndarray, names: list[str]) -> dict:
    """Each body bone's offset from its parent (m), in the rest pose (in the
    world frame, facing +x), from a clip's global poses (T, B, 7) (median over frames, as the
    BVH's position channels vary a little)"""
    idx = {name: i for i, name in enumerate(names)}
    offsets = {}
    parents = body_parents(names)
    for bone, parent in parents.items():
        rot = Rotation.from_quat(poses[:, idx[parent], 3:], scalar_first=True)
        world = rot * Rotation.from_matrix(BVH_TO_WORLD).inv()  # BVH rest -> world
        rel = poses[:, idx[bone], :3] - poses[:, idx[parent], :3]
        offsets[bone] = np.median(world.inv().apply(rel), axis=0)
    return offsets


def rest_position(offsets: dict, parents: dict, bone: str) -> np.ndarray:
    """A bone's position relative to the hips, in the rest pose"""
    pos = np.zeros(3)
    while bone != "Hips":
        pos, bone = pos + offsets[bone], parents[bone]
    return pos


# Some HiPHI clips' skeletons (all the clips of actors A116 to A132 but A127:
# 346, or 1%, mostly vaults, sashays, swings, kicks and falls) have bones
# turned about their own length: (bone, the child whose offset is the bone's
# length, the turn (rad), bones turned with it). The leg bones are turned half
# way, the shoulders (clavicles) a quarter (outward), with the arms beyond them
FLIPPED = tuple(
    (f"{s}{bone}", f"{s}{child}", turn, tuple(f"{s}{b}" for b in also))
    for s, sign in (("Left", 1.0), ("Right", -1.0))
    for bone, child, turn, also in (
        ("UpLeg", "Leg", np.pi, ()),
        ("Leg", "Foot", np.pi, ()),
        ("Foot", "ToeBase", np.pi, ()),
        ("Shoulder", "Arm", sign * np.pi / 2, ("Arm", "ForeArm")),
    )
)


def _twist(rel: Rotation, axis: np.ndarray) -> np.ndarray:
    """Signed angles (rad) of rotations' twist about axis"""
    q = rel.as_quat()  # XYZW, with w >= 0
    q = q * np.where(q[:, 3:] < 0, -1.0, 1.0)
    return 2 * np.arctan2(q[:, :3] @ axis, q[:, 3])


def is_flipped(poses: np.ndarray, names: list[str]) -> bool:
    """Whether a clip's skeleton is the flipped one (see FLIPPED): its shoulders
    (clavicles) are turned a quarter turn about their length, relative to the
    chest, in most of its frames, which no real shoulder can do. Over the whole
    dataset this is all or nothing: within 22.5 deg of the turn in all of a
    clip's frames, or none"""
    idx = {name: i for i, name in enumerate(names)}
    offsets = rest_offsets(poses, names)
    to_world = Rotation.from_matrix(BVH_TO_WORLD)
    chest = Rotation.from_quat(
        poses[:, idx[body_parents(names)["LeftShoulder"]], 3:], scalar_first=True
    )
    for bone, child, turn, _ in FLIPPED:
        if not bone.endswith("Shoulder"):
            continue
        axis = offsets[child] / np.linalg.norm(offsets[child])  # In the rest frame
        rot = Rotation.from_quat(poses[:, idx[bone], 3:], scalar_first=True)
        # Both from the rest pose, in the world frame
        twist = _twist((chest * to_world.inv()).inv() * rot * to_world.inv(), axis)
        if np.mean(np.abs(twist - turn) < np.pi / 8) <= 0.5:
            return False
    return True


def unflip(poses: np.ndarray, names: list[str]) -> np.ndarray:
    """poses (T, B, 7), with a flipped skeleton's bones (see FLIPPED and
    is_flipped) turned back, in all of the clip's frames. Other clips' poses
    are returned unchanged

    The flipped skeleton's offsets are symmetric about the turned bones' axes,
    so its joints' positions are the same either way, but not its orientations:
    the thighs and shins point backward, the feet are upside down (then pitched
    to keep the toes down), and the arms are turned a quarter turn (undone at
    the hands)"""
    if not is_flipped(poses, names):
        return poses
    poses = poses.copy()
    idx = {name: i for i, name in enumerate(names)}
    offsets = rest_offsets(poses, names)
    to_world = Rotation.from_matrix(BVH_TO_WORLD)
    for bone, child, turn, also in FLIPPED:
        axis = offsets[child] / np.linalg.norm(offsets[child])  # In the rest frame
        back = Rotation.from_rotvec(-turn * axis)
        for b in (bone,) + also:
            rot = Rotation.from_quat(poses[:, idx[b], 3:], scalar_first=True)
            fixed = rot * to_world.inv() * back * to_world
            poses[:, idx[b], 3:] = fixed.as_quat(scalar_first=True)
    return poses


def hip_spread(poses: np.ndarray, names: list[str]) -> float:
    """Distance (m) of the actor's hip joints from their center, in the rest pose"""
    offsets = rest_offsets(poses, names)
    return float(
        np.mean([np.linalg.norm(offsets[f"{s}UpLeg"]) for s in ("Left", "Right")])
    )


def ground(poses: np.ndarray, names: list[str]) -> tuple[float, np.ndarray]:
    """A clip's ankle height (m) over its planted feet, and each foot's contact
    with the floor (T, 2): 1 planted, 0 not (see CONTACT_HEIGHT)"""
    fps = HIPHI_FPS
    ankles = np.stack(
        [poses[:, names.index(f"{s}Foot"), :3] for s in ("Left", "Right")], 1
    )
    smooth = gaussian_filter1d(ankles, CONTACT_SMOOTH * fps, axis=0, mode="nearest")
    speed = np.linalg.norm(np.gradient(smooth, axis=0), axis=-1) * fps  # (T, 2)
    z = ankles[..., 2]
    # Feet within PLANTED_TILT of level: their up axes (BVH +y), in the world
    up = np.array([0.0, 1.0, 0.0])
    tilt = np.degrees(
        np.arccos(
            np.clip(
                np.stack(
                    [
                        Rotation.from_quat(
                            poses[:, names.index(f"{s}Foot"), 3:], scalar_first=True
                        ).apply(up)[:, 2]
                        for s in ("Left", "Right")
                    ],
                    1,
                ),
                -1.0,
                1.0,
            )
        )
    )  # (T, 2)
    level = tilt < PLANTED_TILT
    low = np.percentile(z[level], 5) if level.any() else -np.inf
    planted = (speed < PLANTED_SPEED) & level & (z < low + PLANTED_HEIGHT)
    if planted.sum() >= MIN_PLANTED * fps:
        height = float(np.clip(np.median(z[planted]), *ANKLE_HEIGHT_RANGE))
    else:
        height = HIPHI_ANKLE_HEIGHT

    def fade(x, full_below):
        lo, hi = full_below
        return np.clip((hi - x) / (hi - lo), 0.0, 1.0)

    contact = (
        fade(z - height, CONTACT_HEIGHT)
        * fade(speed, CONTACT_SPEED)
        * fade(tilt, CONTACT_TILT)
    )
    contact = gaussian_filter1d(contact, CONTACT_SMOOTH * fps, axis=0, mode="nearest")
    return height, contact


def skeleton_scales(poses: np.ndarray, names: list[str]) -> tuple[float, float]:
    """The clip's (leg, upper body) scales: G1's lengths over the actor's (see
    G1_LEG and G1_UPPER), from the actor's rest pose"""
    offsets, parents = rest_offsets(poses, names), body_parents(names)
    norm = {bone: np.linalg.norm(offset) for bone, offset in offsets.items()}
    leg = np.mean([norm[f"{s}Leg"] + norm[f"{s}Foot"] for s in ("Left", "Right")])
    hips = np.mean(
        [rest_position(offsets, parents, f"{s}UpLeg") for s in ("Left", "Right")],
        axis=0,
    )
    upper = np.mean(
        [
            np.linalg.norm(rest_position(offsets, parents, f"{s}Arm") - hips)
            + norm[f"{s}ForeArm"]
            + norm[f"{s}Hand"]
            for s in ("Left", "Right")
        ]
    )
    return float(G1_LEG / leg), float(G1_UPPER / upper)


def neutral_rotations() -> np.ndarray:
    """The tracked bones' world rotations (n, 3, 3) in the neutral pose (see
    RETARGETING_TABLE), standing in the rest pose's place, facing +x"""
    euler = np.array([row[4] for row in RETARGETING_TABLE], dtype=float)
    return Rotation.from_euler("xyz", euler, degrees=True).as_matrix() @ BVH_TO_WORLD


def rot_offsets(robot, arm_abduction: float = ARM_ABDUCTION) -> np.ndarray:
    """Rotation offsets (n, 3, 3), applied in the bone frame, from each human
    bone to its tracked G1 link: the neutral pose maps to G1's zero
    configuration with its shoulders rolled out by arm_abduction (rad), less
    TORSO_PITCH and FOOT_TILT"""
    q = np.zeros(robot.nq)
    q[3] = 1.0  # Identity root quaternion
    for side, sign in (("left", 1.0), ("right", -1.0)):
        joint = robot.joint_name_to_index[f"{side}_shoulder_roll_joint"]
        q[robot.velocity_to_configuration_index[joint]] = sign * arm_abduction
    frames = np.array([robot.joint_name_to_index[name] for name in G1_JOINT_NAMES])
    R = np.asarray(robot.joint_to_world_transforms(jnp.asarray(q)))[frames, :3, :3]
    offsets = np.swapaxes(neutral_rotations(), -1, -2) @ R
    for bone, pitch in TORSO_PITCH.items():
        level = Rotation.from_euler("y", -pitch, degrees=True).as_matrix()
        offsets[HIPHI_BONES.index(bone)] = offsets[HIPHI_BONES.index(bone)] @ level
    for bone, sign in (("LeftFoot", 1.0), ("RightFoot", -1.0)):
        # In the G1 foot's frame: pitch (y) the toes down, roll (x) the outer edge up
        tilt = (FOOT_TILT[0], sign * FOOT_TILT[1])
        level = Rotation.from_euler("yx", tilt, degrees=True).as_matrix()
        offsets[HIPHI_BONES.index(bone)] = offsets[HIPHI_BONES.index(bone)] @ level
    return offsets


def untwisted(parent: Array, child: Array) -> Array:
    """child (3, 3) without its twist about its x axis relative to parent: the
    rotation from parent with the same x axis, by the shortest path

    HiPHI's forearm bones twist about their length (pronation, and in some
    clips, mocap artifacts past 90 deg), which G1's elbow links can't follow,
    since G1 has that twist at its wrist roll joints. So the forearm targets
    keep only the forearm's direction, and the wrists take the twist from the
    hand targets (whose orientation already includes it)"""
    rel = parent.T @ child
    x = rel[:, 0]  # The child's x axis, in the parent's frame
    # Rotation taking the parent's x axis onto x: axis e_x cross x, angle acos(x_0)
    axis = jnp.array([0.0, -x[2], x[1]])
    s, c = jnp.linalg.norm(axis), x[0]
    k = axis / jnp.maximum(s, 1e-12)
    K = jnp.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    swing = jnp.eye(3) + s * K + (1 - c) * K @ K
    return parent @ swing


def leveled(rmat: Array, weight: Array) -> Array:
    """rmat (3, 3) of a frame with x forward and z up, turned toward level
    (keeping its heading) by weight in [0, 1]"""
    yaw = jnp.arctan2(rmat[1, 0], rmat[0, 0])
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    level = jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    q, q_level = rmat_to_quat_wxyz(rmat), rmat_to_quat_wxyz(level)
    q_level = jnp.where(jnp.dot(q, q_level) < 0, -q_level, q_level)
    q = (1 - weight) * q + weight * q_level  # Small angles, so nlerp is enough
    return quat_wxyz_to_rmat(q / jnp.linalg.norm(q))


def floor_offset(frame: dict) -> Array:
    """How far (m) preprocess drops the scaled targets, so the clip's planted
    ankles sit at G1's ankle height"""
    return frame["leg_scale"] * frame["ankle_height"] - G1_ANKLE_HEIGHT


def preprocess(frame: dict, rot_offsets: np.ndarray) -> dict:
    """Human bone poses (n, 7), in RETARGETING_TABLE order, and the clip's
    measurements (see dataset.load_clip) -> scaled target positions (n, 3) and
    rotation matrices (n, 3, 3) for the tracked G1 links

    The hip center and feet scale about the origin by the leg scale. The pelvis
    sits above the hip center by PELVIS_OFFSET, and the upper body scales about
    the hip center by the upper body scale. The feet move out to G1's hip width
    (along their own lateral axes, so planted feet stay put), and level out
    while planted (frame["contact"]). All targets then drop, so planted feet
    meet the floor"""
    leg, upper = frame["leg_scale"], frame["upper_scale"]
    pos = frame["poses"][:, :3]
    rmats = jax.vmap(quat_wxyz_to_rmat)(frame["poses"][:, 3:]) @ rot_offsets
    rmats = rmats.at[_FOREARMS].set(jax.vmap(untwisted)(rmats[_ARMS], rmats[_FOREARMS]))
    feet = jax.vmap(leveled)(rmats[_FEET], frame["contact"])
    rmats = rmats.at[_FEET].set(feet)
    hips = jnp.mean(pos[_HIPS], axis=0)
    targets = leg * hips + upper * (pos - hips)
    targets = targets.at[0].set(leg * hips + rmats[0] @ PELVIS_OFFSET)
    # The feet's level lateral axes
    lateral = jax.vmap(lambda R: leveled(R, 1.0)[:, 1])(feet)
    spread = G1_ANKLE_SPREAD - leg * frame["hip_spread"]
    targets = targets.at[_FEET].set(
        leg * pos[_FEET] + (_SIDES * spread)[:, None] * lateral
    )
    return {
        "frame_pos": targets.at[:, 2].add(-floor_offset(frame)),
        "frame_rmat": rmats,
    }


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
    wrist_posture_weight: float = WRIST_POSTURE_WEIGHT,
) -> Retargeter:
    """Input frames ({"poses": human bone poses (n, 7), "leg_scale",
    "upper_scale"}, see dataset.load_clip) -> G1 configuration

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
        wrist_posture_weight (float): Weight pulling the wrist joints toward
            zero (see WRIST_POSTURE_WEIGHT)
    """
    offsets = rot_offsets(robot, arm_abduction)
    # Shoulders, then wrists (zero)
    posture = jnp.array(
        [
            (1.0 if side == "left" else -1.0) * arm_abduction * (axis == "roll")
            for side in ("left", "right")
            for axis in ("pitch", "roll", "yaw")
        ]
        + [0.0] * len(WRIST_JOINTS)
    )

    def preprocess_frame(frame: dict) -> dict:
        return preprocess(frame, offsets) | {"posture": posture}

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
        [robot.joint_name_to_index[n] for n in SHOULDER_JOINTS + WRIST_JOINTS],
        "posture",
        GAIN,
        [shoulder_posture_weight] * len(SHOULDER_JOINTS)
        + [wrist_posture_weight] * len(WRIST_JOINTS),
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
