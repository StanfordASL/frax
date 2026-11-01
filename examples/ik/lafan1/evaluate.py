"""Find artifacts in a retargeted LAFAN1 dataset (from retarget.py): snaps,
where the robot's joints move much faster than the human's, and lost or
skating feet. Prints dataset totals, per clip metrics with --clips, and the
worst snaps (clip, frame, joint group) to look at in view.py

A snap is a joint group (e.g. a shoulder's three joints) moving much faster
than the human's bones across it (e.g. upper arm relative to torso), with the
human's speed maxed over +-LAG frames so lag isn't a snap

Metrics (all independent of the position targets' scaling):
- snaps: events where some group moves SNAP rad/s faster than the human
  (jerks: JERK rad/s)
- off_target: % of frames with some tracked link > OFF_TARGET deg from its
  target orientation (wrong IK branches, links pinned at joint limits)
- rot_err: mean orientation error of the tracked links (deg)
- skate: mean horizontal foot speed (cm/s) while the human's foot is planted
- contact_z_std: standard deviation (cm) of the foot height while planted
- floor: % frames with some part of the robot (its collision spheres)
  > FLOOR_TOL below the floor
- floor_depth: deepest point below the floor (cm)
- rms_acc: RMS joint acceleration (rad/s^2)
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false "
    "--xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED"
)
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from lafan1_g1 import (
    DATA_DIR,
    G1_JOINT_NAMES,
    LAFAN1_BONES,
    LAFAN1_FPS,
    OUT_DIR,
    SMOOTH_ROT,
    load_lafan1,
    preprocess,
)
from scipy.ndimage import label, maximum_filter1d

from frax import load_g1
from frax.ik import FloorConstraint, smooth_poses

SNAP = 10.0  # rad/s
JERK = 5.0  # rad/s
LAG = 3  # frames
OFF_TARGET = 45.0  # deg
# Planted: the human ankle is slower than this (m/s), and near its lowest height
CONTACT_SPEED = 0.2
CONTACT_HEIGHT = 0.03  # m above the clip's 5th percentile ankle height
FLOOR_TOL = 0.01  # m
BUCKET = 1024  # Clips are evaluated padded to multiples of this many frames
# Joint group: (parent bone, child bone, G1 joints)
_LEG = ("hip_pitch", "hip_roll", "hip_yaw")
_ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw")
_WRIST = ("wrist_roll", "wrist_pitch", "wrist_yaw")
GROUPS = {"waist": ("Hips", "Spine2", ("waist_yaw", "waist_roll", "waist_pitch"))}
for side, s in (("left", "Left"), ("right", "Right")):
    GROUPS |= {
        f"{side}_hip": ("Hips", f"{s}UpLeg", _LEG),
        f"{side}_knee": (f"{s}UpLeg", f"{s}Leg", ("knee",)),
        f"{side}_ankle": (f"{s}Leg", f"{s}FootMod", ("ankle_pitch", "ankle_roll")),
        f"{side}_shoulder": ("Spine2", f"{s}Arm", _ARM),
        f"{side}_elbow": (f"{s}Arm", f"{s}ForeArm", ("elbow",)),
        f"{side}_wrist": (f"{s}ForeArm", f"{s}Hand", _WRIST),
    }


def _angle(Ra, Rb) -> np.ndarray:
    """Angles (rad) between rotation matrices (..., 3, 3): atan2 of the relative
    rotation's sine (from its skew part) and cosine (from its trace)"""
    rel = np.swapaxes(Ra, -1, -2) @ Rb
    skew = rel - np.swapaxes(rel, -1, -2)
    sin = 0.5 * np.linalg.norm(skew[..., [2, 0, 1], [1, 2, 0]], axis=-1)
    cos = 0.5 * (np.trace(rel, axis1=-2, axis2=-1) - 1.0)
    return np.arctan2(sin, cos)


def _bucketed(f, x):
    """f over a clip's frames (leading axis), with the frames padded to a
    multiple of BUCKET (holding the last), so f compiles for a few lengths"""
    n = len(x)
    x = np.concatenate([x, np.repeat(x[-1:], -n % BUCKET, 0)])
    return jax.tree.map(lambda a: np.asarray(a)[:n], f(x))


class Evaluator:
    def __init__(self, robot, sigma_rot: float = SMOOTH_ROT):
        self.sigma_rot = sigma_rot
        frames = [robot.joint_name_to_index[name] for name in G1_JOINT_NAMES]
        self.fk = jax.jit(
            jax.vmap(lambda q: robot.joint_to_world_transforms(q)[np.array(frames)])
        )
        self.preprocess = jax.jit(jax.vmap(preprocess))
        floor = FloorConstraint(robot)
        self.floor = jax.jit(jax.vmap(lambda q: jnp.min(floor.barrier(q))))
        bone = {b: i for i, b in enumerate(LAFAN1_BONES)}
        self.groups = {
            g: (
                bone[parent],
                bone[child],
                [
                    robot.joint_name_to_index[f"{g.split('_')[0]}_{j}_joint"]
                    if g != "waist"
                    else robot.joint_name_to_index[f"{j}_joint"]
                    for j in joints
                ],
            )
            for g, (parent, child, joints) in GROUPS.items()
        }
        self.nv_floating = robot.nv_floating
        self.feet = [
            ("LeftFoot", bone["LeftFootMod"]),
            ("RightFoot", bone["RightFootMod"]),
        ]

    def clip(self, human: tuple, q: np.ndarray) -> dict:
        """Metrics of a clip's configurations q, given its human poses and bone
        names (load_lafan1)"""
        poses, names = human
        frames = poses[:, [names.index(b) for b in LAFAN1_BONES]]
        if self.sigma_rot > 0:
            frames = smooth_poses(frames, 1.0 / LAFAN1_FPS, sigma_rot=self.sigma_rot)
        target = _bucketed(self.preprocess, frames)["frame_rmats"]
        tfs = _bucketed(self.fk, q)
        rot_err = np.degrees(_angle(tfs[..., :3, :3], target))

        # Joint velocities. q = [root position, root WXYZ quaternion, joints]
        dq = np.diff(q[:, 7:], axis=0) * LAFAN1_FPS
        excess = []
        for parent, child, joints in self.groups.values():
            human = _angle(
                target[:-1, parent].swapaxes(-1, -2) @ target[:-1, child],
                target[1:, parent].swapaxes(-1, -2) @ target[1:, child],
            )
            human = maximum_filter1d(human * LAFAN1_FPS, 2 * LAG + 1)
            robot = np.linalg.norm(dq[:, np.array(joints) - self.nv_floating], axis=1)
            excess.append(robot - human)
        excess = np.stack(excess, axis=1)  # (T - 1, groups)

        skate, height = [], []
        for bone, i in self.feet:
            ankle = poses[:, names.index(bone), :3]
            speed = np.linalg.norm(np.diff(ankle, axis=0), axis=1) * LAFAN1_FPS
            low = np.percentile(ankle[:, 2], 5) + CONTACT_HEIGHT
            planted = (speed < CONTACT_SPEED) & (ankle[1:, 2] < low)
            foot = tfs[:, i, :3, 3]
            foot_speed = np.linalg.norm(np.diff(foot[:, :2], axis=0), axis=1)
            skate.append(foot_speed[planted] * LAFAN1_FPS)
            height.append(foot[1:, 2][planted])
        acc = np.diff(dq, axis=0) * LAFAN1_FPS
        floor = _bucketed(self.floor, q)
        return {
            "snaps": label(excess.max(axis=1) > SNAP)[1],
            "jerks": label(excess.max(axis=1) > JERK)[1],
            "off_target": float(np.mean(rot_err.max(axis=1) > OFF_TARGET) * 100),
            "rot_err": float(rot_err.mean()),
            "skate": float(np.concatenate(skate).mean() * 100),
            "contact_z_std": float(np.concatenate(height).std() * 100),
            "floor": float(np.mean(floor < -FLOOR_TOL) * 100),
            "floor_depth": float(max(-floor.min(), 0.0) * 100),
            "rms_acc": float(np.sqrt(np.mean(acc**2))),
            "excess": excess,
        }


def worst_snaps(excess: np.ndarray, n: int) -> list:
    """Largest snaps of a clip: (excess speed, frame, group)"""
    events = []
    labels, count = label(excess.max(axis=1) > SNAP)
    for k in range(1, count + 1):
        frames = np.flatnonzero(labels == k)
        f, g = np.unravel_index(
            np.argmax(excess[frames]), (len(frames), excess.shape[1])
        )
        events.append((float(excess[frames[f], g]), int(frames[f]), list(GROUPS)[g]))
    return sorted(events, reverse=True)[:n]


METRICS = (
    "snaps",
    "jerks",
    "off_target",
    "rot_err",
    "skate",
    "contact_z_std",
    "floor",
    "floor_depth",
    "rms_acc",
)


def main(args):
    paths = sorted(Path(args.frax_dir).glob("*.npz"))
    assert paths, f"No .npz files in {args.frax_dir}"
    bvhs = [Path(args.data_dir) / f"{path.stem}.bvh" for path in paths]
    # Spawn, not fork: forking after JAX has started its threads is unsafe
    with ProcessPoolExecutor(mp_context=multiprocessing.get_context("spawn")) as pool:
        humans = list(pool.map(load_lafan1, bvhs))
    evaluator = Evaluator(load_g1(), args.sigma_rot)
    results, events = {}, []
    for path, human in zip(paths, humans, strict=True):
        m = evaluator.clip(human, np.load(path)["q"])
        results[path.stem] = m
        events += [
            (e, path.stem, f, g) for e, f, g in worst_snaps(m["excess"], args.worst)
        ]

    def row(m):
        return " ".join(
            f"{k}={m[k]:.2f}" if isinstance(m[k], float) else f"{k}={m[k]}"
            for k in METRICS
        )

    if args.clips:
        for clip, m in sorted(results.items(), key=lambda x: -x[1]["snaps"]):
            print(f"{clip:28s} {row(m)}")
    total = {
        k: (sum if k in ("snaps", "jerks") else np.mean)(
            [m[k] for m in results.values()]
        )
        for k in METRICS
    }
    print(f"{Path(args.frax_dir).name}: {row(total)}")
    groups = np.concatenate([m["excess"] for m in results.values()])
    print(
        "Snaps per group: "
        + ", ".join(
            f"{g}={label(groups[:, i] > SNAP)[1]}" for i, g in enumerate(GROUPS)
        )
    )
    print(f"Worst {args.worst} snaps (excess rad/s, clip, frame, group):")
    for e, clip, f, g in sorted(events, reverse=True)[: args.worst]:
        print(f"  {e:5.1f}  {clip}  {f}  {g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR, help="LAFAN1 BVH files")
    parser.add_argument(
        "--frax_dir", default=OUT_DIR, help="Retargeted clips (.npz), to score"
    )
    parser.add_argument(
        "--sigma_rot",
        type=float,
        default=SMOOTH_ROT,
        help="Orientation smoothing the clips were retargeted with (0: online)",
    )
    parser.add_argument("--clips", action="store_true", help="Print per clip metrics")
    parser.add_argument("--worst", type=int, default=10, help="Worst snaps to list")
    main(parser.parse_args())
