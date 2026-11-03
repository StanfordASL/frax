"""BONES-SEED dataset processing utils

See: https://huggingface.co/datasets/bones-studio/seed

To fetch the data, you'll need to clone it (with git lfs),
extract soma_uniform.tar.gz, and (optionally) link it to
data/seed in the frax folder

If you want to compare against the SOMA retargeter's
performance, extract g1.tar.gz as well
"""

import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from seed_g1 import SEED_BONES, SEED_FPS, load_seed

from frax.ik import smooth_poses

DATA_DIR = Path(__file__).parents[3] / "data/seed"
OUT_DIR = Path(__file__).parents[3] / "data/seed_frax"
METADATA = "metadata/seed_metadata_v004.csv"


def read_metadata(data_dir: str | Path = DATA_DIR) -> list[dict]:
    with open(Path(data_dir) / METADATA, newline="") as f:
        return list(csv.DictReader(f))


def select_clips(
    rows: list[dict],
    n: int | None = None,
    seed: int = 0,
    match: str = "",
) -> list[dict]:
    rows = [
        r
        for r in rows
        if match in r["filename"] or match in r["package"] or match in r["category"]
    ]
    if n is not None and n < len(rows):
        idx = np.random.default_rng(seed).choice(len(rows), n, replace=False)
        rows = [rows[i] for i in sorted(idx)]
    return rows


def bvh_path(row: dict, data_dir: str | Path) -> Path:
    return Path(data_dir) / row["move_soma_uniform_path"]


def load_clip(path: str | Path, sigma_rot: float) -> dict | None:
    try:
        poses, names = load_seed(path)
    except ValueError:
        return None
    frames = poses[:, [names.index(bone) for bone in SEED_BONES]]
    if sigma_rot > 0:
        frames = smooth_poses(frames, 1.0 / SEED_FPS, sigma_rot=sigma_rot)
    return {"poses": frames}


def load_g1_csv(path: str | Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", skiprows=1)[:, 1:]
    turn = Rotation.from_euler("z", 90, degrees=True)
    rot = turn * Rotation.from_euler("xyz", data[:, 3:6], degrees=True)
    pos = turn.apply(data[:, :3] / 100.0)
    quat = rot.as_quat(scalar_first=True)
    return np.concatenate([pos, quat, np.radians(data[:, 6:])], axis=1)
