"""Retarget BONES-SEED clips to the Unitree G1 (`seed_g1.py`), with the clips
spread across CPU cores, and save each clip's configurations (play them back
with view.py)

SEED is large, so clips load in chunks, in worker processes, while the previous
chunk retargets. Saved clips are skipped, so an interrupted run resumes

--refine then redoes the saved clips whose first frame is off target, solving
it by global IK (slower, so only where needed: retarget, then refine)

Data: see dataset.py
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

# One device per core. Set before JAX first runs
jax.config.update("jax_num_cpu_devices", os.cpu_count())

import argparse
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from dataset import (
    DATA_DIR,
    OUT_DIR,
    bvh_path,
    load_clip,
    read_metadata,
    select_clips,
)
from seed_g1 import (
    G1_JOINT_NAMES,
    GLOBAL_IK_FIRST_FRAME_ERROR,
    SEED_FPS,
    SMOOTH_ROT,
    make_retargeter,
    preprocess,
)

from frax import load_g1
from frax.ik import OfflineRetargeter, save_motion

CHUNK = 1024  # Clips loaded at once


def first_frame_error(robot):
    """Returns a function of (clip, configurations) giving the worst tracked link's
    orientation error (deg) in the first frame"""
    links = np.array([robot.joint_name_to_index[name] for name in G1_JOINT_NAMES])

    @jax.jit
    def error(frame, q):
        rmats = robot.joint_to_world_transforms(q)[links, :3, :3]
        rel = jnp.swapaxes(rmats, -1, -2) @ preprocess(frame)["frame_rmat"]
        cos = (jnp.trace(rel, axis1=-2, axis2=-1) - 1) / 2
        return jnp.degrees(jnp.max(jnp.arccos(jnp.clip(cos, -1.0, 1.0))))

    return lambda clip, q: float(error({k: v[0] for k, v in clip.items()}, q[0]))


def main(args):
    data_dir = Path(args.data_dir)
    verb = "refine" if args.refine else "retarget"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = select_clips(read_metadata(data_dir), args.num_clips, args.seed, args.match)

    def saved(r):
        return out_dir / f"{r['filename']}.npz"

    # Refining checks the saved clips; otherwise, they're skipped
    todo = [r for r in rows if saved(r).exists() == args.refine]
    # Longest first, so each chunk's groups have similar lengths
    todo.sort(key=lambda r: -int(r["move_duration_frames"]))
    print(f"{len(rows)} clips selected, {len(todo)} to {verb}")

    # Spawn, not fork: forking after JAX has started its threads is unsafe
    pool = ProcessPoolExecutor(mp_context=multiprocessing.get_context("spawn"))

    def submit(chunk):
        return [
            pool.submit(load_clip, bvh_path(r, data_dir), args.sigma_rot) for r in chunk
        ]

    robot = load_g1()
    retargeter = make_retargeter(
        robot,
        self_collision=not args.no_self_collision,
        global_ik_init=args.global_ik_init or args.refine,
    )
    error = first_frame_error(robot)
    retarget = OfflineRetargeter(retargeter, 1.0 / SEED_FPS)
    chunks = [todo[i : i + CHUNK] for i in range(0, len(todo), CHUNK)]
    loading = submit(chunks[0]) if chunks else []
    load_time = solve_time = frames = refined = 0
    skipped = []
    for k, chunk in enumerate(chunks):
        t0 = time.perf_counter()
        clips = [f.result() for f in loading]
        loading = submit(chunks[k + 1]) if k + 1 < len(chunks) else []
        skipped += [r["filename"] for r, c in zip(chunk, clips) if c is None]
        chunk = [r for r, c in zip(chunk, clips) if c is not None]
        clips = [c for c in clips if c is not None]
        if args.refine:
            off = [
                error(c, np.load(saved(r))["q"]) > GLOBAL_IK_FIRST_FRAME_ERROR
                for r, c in zip(chunk, clips)
            ]
            chunk = [r for r, o in zip(chunk, off) if o]
            clips = [c for c, o in zip(clips, off) if o]
            refined += len(clips)
        t1 = time.perf_counter()
        qs = retarget(clips)
        for r, q in zip(chunk, qs):
            save_motion(saved(r), robot, q, SEED_FPS)
        t2 = time.perf_counter()
        load_time += t1 - t0
        solve_time += t2 - t1
        frames += sum(len(q) for q in qs)
        print(
            f"Chunk {k + 1}/{len(chunks)}: {frames} frames, "
            + f"{frames / solve_time:.0f} frames/s (waited {load_time:.0f} s "
            + "on loading)"
            + (f", {refined} clips refined" if args.refine else "")
        )
    pool.shutdown()
    print(
        f"Compilation time (in parallel with the solve): {retarget.compile_time:.0f} s"
    )
    if skipped:
        (out_dir / "skipped.txt").write_text("\n".join(skipped) + "\n")
        print(f"Skipped {len(skipped)} clips with NaN frames (skipped.txt)")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR, help="SEED dataset root")
    parser.add_argument("--out_dir", default=OUT_DIR, help="Retargeted clips (.npz)")
    parser.add_argument(
        "--num_clips", type=int, default=None, help="Random clips (default: all)"
    )
    parser.add_argument("--seed", type=int, default=0, help="Clip sampling seed")
    parser.add_argument(
        "--match", default="", help="Only clips whose name/package/category has this"
    )
    parser.add_argument(
        "--global_ik_init",
        action="store_true",
        help="Solve each clip's first frame by global IK",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Redo the saved clips whose first frame is off target, by global IK",
    )
    parser.add_argument(
        "--no_self_collision", action="store_true", help="No self-collision avoidance"
    )
    parser.add_argument(
        "--sigma_rot",
        type=float,
        default=SMOOTH_ROT,
        help="Orientation smoothing (s). 0: none, as online",
    )
    main(parser.parse_args())
