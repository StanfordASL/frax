"""Script to retarget the HiPHI dataset to the Unitree G1

As in the bones seed retargeting,
- Pass --refine to re-solve any inaccurate first frames with global IK
- Pass --overwrite to re-retarget any clips. By default, if a run is
  interrupted, a new run will begin where it left off
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

jax.config.update("jax_num_cpu_devices", os.cpu_count())

import argparse
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from dataset import (
    OUT_DIR,
    add_selection_args,
    chunks,
    load_clip,
    map_clips,
    selected_clips,
)
from hiphi_g1 import (
    G1_JOINT_NAMES,
    GLOBAL_IK_FIRST_FRAME_ERROR,
    HIPHI_FPS,
    make_retargeter,
    preprocess,
    rot_offsets,
)

from frax import load_g1
from frax.ik import OfflineRetargeter, save_motion

CHUNK = 1024  # Clips retargeted at once


def first_frame_error(robot):
    """Returns a function of (clip, configurations) giving the worst tracked link's
    orientation error (deg) in the first frame"""
    links = np.array([robot.joint_name_to_index[name] for name in G1_JOINT_NAMES])
    offsets = rot_offsets(robot)

    @jax.jit
    def error(frame, q):
        rmats = robot.joint_to_world_transforms(q)[links, :3, :3]
        rel = jnp.swapaxes(rmats, -1, -2) @ preprocess(frame, offsets)["frame_rmat"]
        cos = (jnp.trace(rel, axis1=-2, axis2=-1) - 1) / 2
        return jnp.degrees(jnp.max(jnp.arccos(jnp.clip(cos, -1.0, 1.0))))

    return lambda clip, q: float(error({k: v[0] for k, v in clip.items()}, q[0]))


def main(args):
    verb = "refine" if args.refine else "retarget"
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = selected_clips(args)

    def saved(r):
        return results_dir / f"{r['motion_id']}.npz"

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
    print(f"{len(rows)} clips selected, {len(todo)} to {verb}")

    robot = load_g1()
    retargeter = make_retargeter(
        robot,
        self_collision=args.self_collision_avoidance,
        global_ik_init=args.refine,
    )
    error = first_frame_error(robot)
    retarget = OfflineRetargeter(retargeter, 1.0 / HIPHI_FPS)
    solve_time = frames = refined = 0
    skipped = []
    # Spawn, not fork: forking after JAX has started its threads is unsafe
    with ProcessPoolExecutor(mp_context=multiprocessing.get_context("spawn")) as pool:
        # The next chunk loads while this one solves
        loading = map_clips(pool, load_clip, todo, args.data_dir, ahead=CHUNK)
        start = time.perf_counter()
        for k, chunk in enumerate(chunks(loading, CHUNK)):
            clips = [(r, f.result()) for r, f in chunk]
            skipped += [r["motion_id"] for r, c in clips if c is None]
            clips = [(r, c) for r, c in clips if c is not None]
            if args.refine:
                clips = [
                    (r, c)
                    for r, c in clips
                    if error(c, np.load(saved(r))["q"]) > GLOBAL_IK_FIRST_FRAME_ERROR
                ]
                refined += len(clips)
            t1 = time.perf_counter()
            qs = retarget([c for _, c in clips])
            for (r, _), q in zip(clips, qs):
                save_motion(saved(r), robot, q, HIPHI_FPS)
            solve_time += time.perf_counter() - t1
            frames += sum(len(q) for q in qs)
            done = min((k + 1) * CHUNK, len(todo))
            print(
                f"Chunk {k + 1}/{-(-len(todo) // CHUNK)}: {done} clips, "
                + f"{frames} frames, {frames / solve_time:.0f} frames/s solving, "
                + f"{time.perf_counter() - start:.0f} s total"
                + (f", {refined} clips refined" if args.refine else ""),
                flush=True,
            )
    print(
        f"Compilation time (in parallel with the solve): {retarget.compile_time:.0f} s"
    )
    if skipped:
        (results_dir / "skipped.txt").write_text("\n".join(skipped) + "\n")
        print(f"Skipped {len(skipped)} clips with NaN frames (skipped.txt)")
    print(f"Saved to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_selection_args(parser)
    parser.add_argument(
        "--results-dir", default=OUT_DIR, help="Where to save retargeted clips (.npz)"
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
