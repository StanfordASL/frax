"""Script to retarget the full BONES-SEED dataset to the Unitree G1

Note: BONES-SEED is much larger than LAFAN1, so clips load in
chunks while the previous chunk is processing.

Clips already in --results-dir are skipped, so an interrupted run
picks up where it left off. Pass --overwrite to redo them (e.g.
after changing the retargeting settings).

This script also adds a `--refine` option where any clips where
the first frame is off-target are re-solved on initialization by
the frax.ik global IK solver. This is much slower than solving one
differential IK problem, so this is done only when needed, after
a first pass from just differential IK. This is useful for a few
BONES-SEED clips to avoid local minima
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
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = select_clips(
        read_metadata(data_dir), args.num_clips, args.rng_seed, args.match
    )

    def saved(r):
        return results_dir / f"{r['filename']}.npz"

    if args.refine:
        # Refining checks the saved clips
        todo = [r for r in rows if saved(r).exists()]
    elif args.overwrite:
        todo = rows
    else:
        todo = [r for r in rows if not saved(r).exists()]
        if len(todo) < len(rows):
            print(
                f"Skipping {len(rows) - len(todo)} clips already in {results_dir} "
                + "(--overwrite to redo them)"
            )
    # Longest first, so each chunk's groups have similar lengths
    todo.sort(key=lambda r: -float(r["move_duration_frames"]))
    print(f"{len(rows)} clips selected, {len(todo)} to {verb}")

    # Spawn, not fork: forking after JAX has started its threads is unsafe
    pool = ProcessPoolExecutor(mp_context=multiprocessing.get_context("spawn"))

    def submit(chunk):
        return [pool.submit(load_clip, bvh_path(r, data_dir)) for r in chunk]

    robot = load_g1()
    retargeter = make_retargeter(
        robot,
        self_collision=args.self_collision_avoidance,
        global_ik_init=args.refine,
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
        (results_dir / "skipped.txt").write_text("\n".join(skipped) + "\n")
        print(f"Skipped {len(skipped)} clips with NaN frames (skipped.txt)")
    print(f"Saved to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR, help="SEED dataset root")
    parser.add_argument(
        "--results-dir", default=OUT_DIR, help="Where to save retargeted clips (.npz)"
    )
    parser.add_argument(
        "--match", default="", help="Only clips whose name matches this regex"
    )
    parser.add_argument(
        "--num-clips", type=int, default=None, help="Random subset size (default: all)"
    )
    parser.add_argument(
        "--rng-seed", type=int, default=0, help="Random seed for --num-clips"
    )
    parser.add_argument(
        "--self-collision-avoidance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Avoid self-collisions",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Redo clips already in --results-dir (default: skip them)",
    )
    mode.add_argument(
        "--refine",
        action="store_true",
        help="Redo the saved clips whose first frame is off target, by global IK",
    )
    main(parser.parse_args())
