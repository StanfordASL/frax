"""Script to retarget the full LAFAN1 dataset to the Unitree G1

(at >100,000 FPS on a i9-14900KF CPU!)
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false "
    "--xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED"
)
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import jax

jax.config.update("jax_num_cpu_devices", os.cpu_count())  # One device per core

import argparse
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from lafan1_g1 import (
    DATA_DIR,
    LAFAN1_BONES,
    LAFAN1_FPS,
    OUT_DIR,
    SMOOTH_ROT,
    load_lafan1,
    make_retargeter,
)

from frax import load_g1
from frax.ik import (
    OfflineRetargeter,
    save_motion,
    smooth_poses,
)


def load_clip(path: Path, sigma_rot: float = SMOOTH_ROT) -> np.ndarray:
    """LAFAN1_BONES poses from one BVH file, smoothed. Offline, so the bone
    orientations can be smoothed without lag"""
    poses, names = load_lafan1(path)
    frames = poses[:, [names.index(bone) for bone in LAFAN1_BONES]]
    return smooth_poses(frames, 1.0 / LAFAN1_FPS, sigma_rot=sigma_rot)


def main(args):
    dt = 1.0 / LAFAN1_FPS
    paths = sorted(Path(args.data_dir).glob("*.bvh"))
    if not paths:
        raise SystemExit(
            f"No LAFAN1 BVH files in {args.data_dir}: "
            + "run python examples/ik/lafan1/download.py"
        )
    start = time.perf_counter()
    # Spawn, not fork: forking after JAX has started its threads is unsafe
    with ProcessPoolExecutor(mp_context=multiprocessing.get_context("spawn")) as pool:
        clips = list(pool.map(load_clip, paths, [args.sigma_rot] * len(paths)))
    print(
        f"Loaded {len(clips)} clips ({sum(map(len, clips))} frames) "
        + f"in {time.perf_counter() - start:.1f} s"
    )
    robot = load_g1()
    retargeter = make_retargeter(robot, self_collision=args.self_collision)
    retarget = OfflineRetargeter(retargeter, dt)
    start = time.perf_counter()
    qs = retarget(clips)
    print(
        f"Retargeted {len(clips)} clips on "
        + f"{min(len(clips), len(retarget.devices))} devices in "
        + f"{time.perf_counter() - start:.1f} s"
    )
    print(
        f"Compilation time (in parallel with the solve): {retarget.compile_time:.1f} s"
    )

    if args.serial:
        start = time.perf_counter()
        for clip in clips:
            np.asarray(retargeter.retarget_clip(clip, dt)[0])
        print(
            f"One clip at a time on one device: {time.perf_counter() - start:.1f} s "
            + "(including compilation)"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, q in zip(paths, qs):
        save_motion(out_dir / f"{path.stem}.npz", robot, q, LAFAN1_FPS)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR, help="LAFAN1 BVH files")
    parser.add_argument(
        "--serial", action="store_true", help="Also time one clip at a time"
    )
    parser.add_argument("--out-dir", default=OUT_DIR, help="Retargeted clips (.npz)")
    parser.add_argument(
        "--self-collision", action="store_true", help="Self-collision avoidance"
    )
    parser.add_argument(
        "--sigma-rot",
        type=float,
        default=SMOOTH_ROT,
        help="Orientation smoothing (s). 0: none, as online",
    )
    main(parser.parse_args())
