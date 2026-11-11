"""HiPHI dataset processing utils

See: https://huggingface.co/datasets/noitomrobotics/HiPHI

To fetch the data, download it (with git lfs, or `hf download`), extract
its .tar.zst archives in place (about 860 GB, into the dataset's
data/{frame}/{lu}/{motion_id} tree), and (optionally) link it to
data/hiphi in the frax folder
"""

import argparse
import csv
import itertools
import json
import re
from collections import defaultdict, deque
from collections.abc import Iterator
from concurrent.futures import Future
from pathlib import Path

import numpy as np
from hiphi_g1 import (
    BVH_TO_WORLD,
    HIPHI_BONES,
    HIPHI_FPS,
    SMOOTH_ROT,
    bone_indices,
    ground,
    hip_spread,
    skeleton_scales,
    unflip,
)
from scipy.spatial.transform import Rotation

from frax.ik import load_bvh, smooth_poses

DATA_DIR = Path(__file__).parents[3] / "data/hiphi"
OUT_DIR = Path(__file__).parents[3] / "data/hiphi_frax"
METADATA = "metadata/hiphi_metadata.csv"
BVH = "motion_actor.bvh"
MIRROR = "__mirror"


def load_hiphi(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Global bone poses from a HiPHI BVH. Raises ValueError if any frame has
    NaNs

    Returns:
        poses (np.ndarray): Poses [position, WXYZ quaternion], shape (T, B, 7)
        names (list[str]): The B bone names
    """
    return load_bvh(path, BVH_TO_WORLD)


def read_metadata(data_dir: str | Path = DATA_DIR) -> list[dict]:
    """hiphi_metadata.csv's rows"""
    with open(Path(data_dir) / METADATA, newline="") as f:
        return list(csv.DictReader(f))


def read_clip_list(path: str | Path) -> list[str]:
    """Motion IDs, the first word of each line (hard_cases.py's lists),
    skipping blank lines and # comments"""
    lines = Path(path).read_text().splitlines()
    return [line.split()[0] for line in lines if line.strip() and line[0] != "#"]


def name_matches(name: str, match: str) -> bool:
    """Whether a clip name matches the regex match (case-insensitive)"""
    return re.search(match, name, re.IGNORECASE) is not None


def select_clips(
    rows: list[dict],
    n: int | None = None,
    rng_seed: int = 0,
    match: str = "",
    mirrors: bool = True,
    clip_list: str | Path | None = None,
    per_frame_lu: int | None = None,
) -> list[dict]:
    """Rows whose motion ID matches the regex match (and is in clip_list, if
    given), without the __mirror clips unless mirrors, then per_frame_lu of
    each Frame-LU at random (a subset covering every kind of motion), then n
    of them at random"""
    rows = [r for r in rows if name_matches(r["motion_id"], match)]
    if not mirrors:
        rows = [r for r in rows if r["mirrored"] != "true"]
    if clip_list is not None:
        listed = set(read_clip_list(clip_list))
        rows = [r for r in rows if r["motion_id"] in listed]
    if per_frame_lu is not None:
        rng = np.random.default_rng(rng_seed)
        labels = defaultdict(list)
        for i, r in enumerate(rows):
            labels[r["frame_lu"]].append(i)
        keep = set()
        for _, idx in sorted(labels.items()):
            keep |= set(rng.permutation(idx)[:per_frame_lu].tolist())
        rows = [r for i, r in enumerate(rows) if i in keep]
    if n is not None and n < len(rows):
        idx = np.random.default_rng(rng_seed).choice(len(rows), n, replace=False)
        rows = [rows[i] for i in sorted(idx)]
    return rows


def package_dir(row: dict, data_dir: str | Path = DATA_DIR) -> Path:
    """A clip's package (its BVH, metadata.json, and object tracks)"""
    return Path(data_dir) / "data" / row["frame"] / row["lu"] / row["motion_id"]


def bvh_path(row: dict, data_dir: str | Path = DATA_DIR) -> Path:
    """A clip's BVH"""
    return package_dir(row, data_dir) / BVH


def map_clips(
    pool, fn, rows: list[dict], data_dir: str | Path = DATA_DIR, ahead: int = 64
) -> Iterator[tuple[dict, Future]]:
    """(row, future of fn(BVH path, row)) for each row, in order, fn running on
    pool (a process pool). Keeps up to ahead clips submitted past the one last
    yielded, so they load while the caller works"""
    pending = deque()
    for r in rows:
        pending.append((r, pool.submit(fn, bvh_path(r, data_dir), r)))
        if len(pending) > ahead:
            yield pending.popleft()
    yield from pending


def chunks(items: Iterator, size: int) -> Iterator[list]:
    """Lists of up to size consecutive items"""
    while chunk := list(itertools.islice(items, size)):
        yield chunk


def load_clip(path, row: dict | None = None, sigma_rot: float = SMOOTH_ROT):
    """A clip's retargeting input (see hiphi_g1.preprocess) from its BVH: the tracked bones' poses, the feet's contact, and the
    clip's measurements (skeleton scales, hip spread, ankle height), per frame.
    None if the clip has NaN frames"""
    try:
        poses, names = load_hiphi(path)
    except ValueError:
        return None
    poses = unflip(poses, names)
    leg_scale, upper_scale = skeleton_scales(poses, names)
    ankle_height, contact = ground(poses, names)
    frames = poses[:, bone_indices(names, HIPHI_BONES)]
    if sigma_rot > 0:
        frames = smooth_poses(frames, 1.0 / HIPHI_FPS, sigma_rot=sigma_rot)
    T = len(frames)
    measured = {
        "leg_scale": leg_scale,
        "upper_scale": upper_scale,
        "hip_spread": hip_spread(poses, names),
        "ankle_height": ankle_height,
    }
    return {"poses": frames, "contact": contact} | {
        k: np.full(T, v) for k, v in measured.items()
    }


def load_objects(row: dict, data_dir: str | Path = DATA_DIR) -> list[dict]:
    """An HOI package's tracked objects: {object_id, mesh (path),
    poses (T, 7): [position, WXYZ quaternion] in the world frame}"""
    pkg = package_dir(row, data_dir)
    meta = json.loads((pkg / "metadata.json").read_text())
    objects = []
    for obj in meta["objects"]:
        track = np.loadtxt(pkg / obj["trajectory_path"], delimiter=",", skiprows=1)
        pos = track[:, 2:5] @ BVH_TO_WORLD.T  # Already in m
        rot = Rotation.from_matrix(BVH_TO_WORLD) * Rotation.from_quat(track[:, 5:9])
        objects.append(
            {
                "object_id": obj["object_id"],
                "mesh": Path(data_dir) / obj["mesh_path"],
                "poses": np.concatenate([pos, rot.as_quat(scalar_first=True)], 1),
            }
        )
    return objects


def add_selection_args(parser):
    """Arguments choosing clips (see select_clips), shared by the scripts"""
    parser.add_argument("--data-dir", default=DATA_DIR, help="HiPHI dataset root")
    parser.add_argument(
        "--match", default="", help="Only clips whose motion ID matches this regex"
    )
    parser.add_argument(
        "--clip-list", help="Only the clips in this file (e.g. hard_cases.py's)"
    )
    parser.add_argument(
        "--mirrors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the __mirror clips",
    )
    parser.add_argument(
        "--per-frame-lu",
        type=int,
        default=None,
        help="Random subset of this many clips per Frame-LU",
    )
    parser.add_argument(
        "--num-clips", type=int, default=None, help="Random subset size (default: all)"
    )
    parser.add_argument(
        "--rng-seed", type=int, default=0, help="Random seed for the subsets"
    )


def selected_clips(args) -> list[dict]:
    """The rows add_selection_args' arguments choose"""
    return select_clips(
        read_metadata(args.data_dir),
        args.num_clips,
        args.rng_seed,
        args.match,
        args.mirrors,
        args.clip_list,
        args.per_frame_lu,
    )
