"""Score retargeted BONES-SEED clips (from retarget.py), against SEED's own G1
retargets (NVIDIA's soma-retargeter) on the same clips, in parallel across
processes. Prints dataset totals, totals per package (--packages), per clip
metrics (--clips), and the clips with the worst snaps (clip, frame, joint group)
to look at in view.py. --save keeps the per clip metrics (a CSV), to report
again (--load) or pick hard cases from (hard_cases.py)

The metrics are examples/ik/lafan1/evaluate.py's, against the human's own
(unsmoothed) bone orientations, so smoothing counts as error. Also:
- self_coll: % frames with some pair of G1's self-collision links
  > SELF_COLLISION_TOL into each other
- self_coll_depth: deepest such penetration (cm), mean over clips
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
import csv
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from dataset import (
    DATA_DIR,
    OUT_DIR,
    bvh_path,
    load_clip,
    load_g1_csv,
    read_clip_list,
    read_metadata,
)
from scipy.ndimage import label, maximum_filter1d
from scipy.spatial.transform import Rotation
from seed_g1 import G1_JOINT_NAMES, SEED_BONES, SEED_FPS, preprocess

from frax import load_g1

SNAP = 10.0  # rad/s
JERK = 5.0  # rad/s
LAG = round(0.1 * SEED_FPS)  # frames
OFF_TARGET = 45.0  # deg
# Planted: the human ankle is slower than this (m/s), and near its lowest height
CONTACT_SPEED = 0.2
CONTACT_HEIGHT = 0.03  # m above the clip's 5th percentile ankle height
FLOOR_TOL = 0.01  # m
SELF_COLLISION_TOL = 0.01  # m
BUCKET = 1024  # Clips are evaluated padded to multiples of this many frames
# Joint group: (parent bone, child bone, G1 joints)
_LEG = ("hip_pitch", "hip_roll", "hip_yaw")
_ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw")
_WRIST = ("wrist_roll", "wrist_pitch", "wrist_yaw")
GROUPS = {"waist": ("Hips", "Chest", ("waist_yaw", "waist_roll", "waist_pitch"))}
for side, s in (("left", "Left"), ("right", "Right")):
    GROUPS |= {
        f"{side}_hip": ("Hips", f"{s}Leg", _LEG),
        f"{side}_knee": (f"{s}Leg", f"{s}Shin", ("knee",)),
        f"{side}_ankle": (f"{s}Shin", f"{s}Foot", ("ankle_pitch", "ankle_roll")),
        f"{side}_shoulder": ("Chest", f"{s}Arm", _ARM),
        f"{side}_elbow": (f"{s}Arm", f"{s}ForeArm", ("elbow",)),
        f"{side}_wrist": (f"{s}ForeArm", f"{s}Hand", _WRIST),
    }
METRICS = (
    "snaps",
    "jerks",
    "off_target",
    "rot_err",
    "skate",
    "contact_z_std",
    "floor",
    "floor_depth",
    "self_coll",
    "self_coll_depth",
    "rms_acc",
)
SOURCES = ("frax", "nvidia")


def _angle(Ra, Rb) -> np.ndarray:
    """Angles (rad) between rotation matrices (..., 3, 3)"""
    rel = np.swapaxes(Ra, -1, -2) @ Rb
    return (
        Rotation.from_matrix(rel.reshape(-1, 3, 3)).magnitude().reshape(rel.shape[:-2])
    )


def _bucketed(f, x):
    """f over a clip's frames (leading axis), with the frames padded to a
    multiple of BUCKET (holding the last), so f compiles for a few lengths"""
    n = len(jax.tree.leaves(x)[0])
    pad = -n % BUCKET
    x = jax.tree.map(lambda a: np.concatenate([a, np.repeat(a[-1:], pad, 0)]), x)
    return jax.tree.map(lambda a: np.asarray(a)[:n], f(x))


class Evaluator:
    def __init__(self, robot):
        frames = [robot.joint_name_to_index[name] for name in G1_JOINT_NAMES]
        self.fk = jax.jit(
            jax.vmap(lambda q: robot.joint_to_world_transforms(q)[np.array(frames)])
        )
        self.preprocess = jax.jit(jax.vmap(preprocess))

        def floor_clearance(q):
            pos, radii = robot.link_collision_data(q)
            return jnp.min(pos[:, 2] - radii)

        self.floor = jax.jit(jax.vmap(floor_clearance))
        self.self_coll = jax.jit(
            jax.vmap(lambda q: jnp.min(robot.self_collision_distances(q)))
        )
        bone = {b: i for i, b in enumerate(SEED_BONES)}
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
        self.feet = [bone["LeftFoot"], bone["RightFoot"]]

    def clip(self, human: dict, q: np.ndarray) -> dict:
        q = np.asarray(q, dtype=float)
        target = _bucketed(self.preprocess, human)["frame_rmats"]
        tfs = _bucketed(self.fk, q)
        rot_err = np.degrees(_angle(tfs[..., :3, :3], target))

        # Joint velocities. q = [root position, root WXYZ quaternion, joints]
        dq = np.diff(q[:, 7:], axis=0) * SEED_FPS
        excess = []
        for parent, child, joints in self.groups.values():
            speed = _angle(
                target[:-1, parent].swapaxes(-1, -2) @ target[:-1, child],
                target[1:, parent].swapaxes(-1, -2) @ target[1:, child],
            )
            speed = maximum_filter1d(speed * SEED_FPS, 2 * LAG + 1)
            robot = np.linalg.norm(dq[:, np.array(joints) - self.nv_floating], axis=1)
            excess.append(robot - speed)
        excess = np.stack(excess, axis=1)  # (T - 1, groups)

        skate, height = [], []
        for i in self.feet:
            ankle = human["poses"][:, i, :3]
            speed = np.linalg.norm(np.diff(ankle, axis=0), axis=1) * SEED_FPS
            low = np.percentile(ankle[:, 2], 5) + CONTACT_HEIGHT
            planted = (speed < CONTACT_SPEED) & (ankle[1:, 2] < low)
            foot = tfs[:, i, :3, 3]
            foot_speed = np.linalg.norm(np.diff(foot[:, :2], axis=0), axis=1)
            skate.append(foot_speed[planted] * SEED_FPS)
            height.append(foot[1:, 2][planted])
        skate, height = np.concatenate(skate), np.concatenate(height)
        acc = np.diff(dq, axis=0) * SEED_FPS
        floor = _bucketed(self.floor, q)
        self_coll = _bucketed(self.self_coll, q)
        return {
            "snaps": label(excess.max(axis=1) > SNAP)[1],
            "jerks": label(excess.max(axis=1) > JERK)[1],
            "off_target": float(np.mean(rot_err.max(axis=1) > OFF_TARGET) * 100),
            "rot_err": float(rot_err.mean()),
            "skate": float(skate.mean() * 100) if len(skate) else np.nan,
            "contact_z_std": float(height.std() * 100) if len(height) else np.nan,
            "floor": float(np.mean(floor < -FLOOR_TOL) * 100),
            "floor_depth": float(max(-floor.min(), 0.0) * 100),
            "self_coll": float(np.mean(self_coll < -SELF_COLLISION_TOL) * 100),
            "self_coll_depth": float(max(-self_coll.min(), 0.0) * 100),
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


_evaluator = None  # Each worker process's own


def score_clip(bvh: Path, sigma_rot: float, qs: dict) -> dict | None:
    """Score each source's retarget of one clip

    Args:
        bvh (Path): Human clip
        sigma_rot (float): Orientation smoothing for the human clip (s)
        qs (dict): Path to each source's retarget, as {source: path}

    Returns:
        dict | None: Metrics for each source, as {source: metrics}. None if the
            human clip has NaN frames
    """
    global _evaluator
    if _evaluator is None:
        _evaluator = Evaluator(load_g1())
    human = load_clip(bvh, sigma_rot)
    if human is None:
        return None
    out = {}
    for source, path in qs.items():
        q = np.load(path)["q"] if source == "frax" else load_g1_csv(path)
        m = _evaluator.clip(human, q)
        excess = m.pop("excess")
        e, f, g = (worst_snaps(excess, 1) or [(0.0, -1, "")])[0]
        m |= {"frames": len(q), "worst_snap": e, "worst_frame": f, "worst_group": g}
        m |= {g: label(excess[:, i] > SNAP)[1] for i, g in enumerate(GROUPS)}
        out[source] = m
    return out


def save_metrics(path: Path, results: dict):
    """Per clip metrics, one row per (source, clip)"""
    rows = [
        {"source": s, "clip": c} | m
        for s, clips in results.items()
        for c, m in clips.items()
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_metrics(path: Path) -> dict:
    """Load metrics written by save_metrics, as {source: {clip: metrics}}"""
    results = defaultdict(dict)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            source, clip = r.pop("source"), r.pop("clip")
            results[source][clip] = {
                k: v
                if k == "worst_group"
                else (int(v) if v.lstrip("-").isdigit() else float(v))
                for k, v in r.items()
            }
    return dict(results)


def total(results: list) -> dict:
    """Dataset totals: snap counts summed, the rest averaged over clips"""
    return {
        k: (sum if k in ("snaps", "jerks") else np.nanmean)([m[k] for m in results])
        for k in METRICS
    }


def row(m: dict) -> str:
    return " ".join(
        f"{k}={m[k]:.2f}" if isinstance(m[k], float) else f"{k}={m[k]}" for k in METRICS
    )


def evaluate(args, meta: dict, names: list) -> dict:
    """Score every clip over worker processes, as {source: {clip: metrics}}"""
    data_dir = Path(args.data_dir)
    paths = {
        "frax": lambda n: Path(args.frax_dir) / f"{n}.npz",
        "nvidia": lambda n: data_dir / meta[n]["move_g1_path"],
    }
    results = {s: {} for s in args.sources}
    with ProcessPoolExecutor(mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [
            pool.submit(
                score_clip,
                bvh_path(meta[n], data_dir),
                args.sigma_rot,
                {s: paths[s](n) for s in args.sources},
            )
            for n in names
        ]
        for k, (name, future) in enumerate(zip(names, futures)):
            for s, m in (future.result() or {}).items():
                results[s][name] = m
            if (k + 1) % 5000 == 0:
                print(f"Scored {k + 1}/{len(names)} clips", flush=True)
    return results


def main(args):
    data_dir = Path(args.data_dir)
    meta = {r["filename"]: r for r in read_metadata(data_dir)}
    if args.load:
        results = load_metrics(Path(args.load))
        names = sorted(results[args.sources[0]])
    else:
        names = sorted(p.stem for p in Path(args.frax_dir).glob("*.npz"))
        assert names, f"No .npz files in {args.frax_dir}"
    if args.clip_list:
        listed = set(read_clip_list(args.clip_list))
        names = [n for n in names if n in listed]
    if args.num_clips is not None:
        names = names[: args.num_clips]
    if args.load:
        results = {s: {n: results[s][n] for n in names} for s in args.sources}
    else:
        results = evaluate(args, meta, names)
    if args.save:
        save_metrics(Path(args.save), results)

    names = list(results[args.sources[0]])
    print(f"{len(names)} clips")
    for s in args.sources:
        print(f"{s:7s} {row(total(list(results[s].values())))}")
    if args.packages:
        packages = defaultdict(list)
        for name in names:
            packages[meta[name]["package"]].append(name)
        for package, clips in sorted(packages.items(), key=lambda x: -len(x[1])):
            print(f"{package} ({len(clips)} clips):")
            for s in args.sources:
                print(f"  {s:7s} {row(total([results[s][c] for c in clips]))}")
    for s in args.sources:
        if args.clips:
            print(f"{s} per clip:")
            for clip, m in sorted(results[s].items(), key=lambda x: -x[1]["snaps"]):
                print(f"  {clip:48s} {row(m)}")
        print(
            f"{s} snaps per group: "
            + ", ".join(f"{g}={sum(m[g] for m in results[s].values())}" for g in GROUPS)
        )
        print(f"{s} worst {args.worst} snaps (excess rad/s, clip, frame, group):")
        worst = sorted(results[s].items(), key=lambda x: -x[1]["worst_snap"])
        for clip, m in worst[: args.worst]:
            print(
                f"  {m['worst_snap']:5.1f}  {clip}  {m['worst_frame']}  {m['worst_group']}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR, help="SEED dataset root")
    parser.add_argument(
        "--frax_dir", default=OUT_DIR, help="Retargeted clips (.npz), to score"
    )
    parser.add_argument(
        "--sources", nargs="+", default=SOURCES, choices=SOURCES, help="To score"
    )
    parser.add_argument(
        "--num_clips", type=int, default=None, help="Only the first n clips"
    )
    parser.add_argument(
        "--sigma_rot",
        type=float,
        default=0.0,
        help="Smoothing of the human orientations scored against (0: none)",
    )
    parser.add_argument(
        "--packages", action="store_true", help="Print totals per package"
    )
    parser.add_argument("--clips", action="store_true", help="Print per clip metrics")
    parser.add_argument(
        "--worst", type=int, default=10, help="Clips with the worst snaps to list"
    )
    parser.add_argument(
        "--clip_list", help="Only the clips in this file (hard_cases.py)"
    )
    parser.add_argument("--save", help="Save the per clip metrics to this CSV")
    parser.add_argument(
        "--load", help="Report the per clip metrics in this CSV (--save), not rescore"
    )
    main(parser.parse_args())
