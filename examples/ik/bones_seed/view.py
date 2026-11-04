"""Play the retargeted BONES-SEED dataset in a viser viewer

This script also visualizes the result of the SOMA retargeter,
as well as the original human skeleton, for comparison
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
import time
from itertools import pairwise
from pathlib import Path

import numpy as np
import viser
from dataset import (
    DATA_DIR,
    OUT_DIR,
    bvh_path,
    load_g1_csv,
    load_seed,
    name_matches,
    read_metadata,
)
from seed_g1 import SEED_FPS
from viser.extras import ViserUrdf

from frax.assets import G1_ASSETS_DIR

G1_URDF = G1_ASSETS_DIR / "g1_29dof_rev_1_0.urdf"
# Sideways offsets (m) of SEED's G1 retarget and the human from frax's
SPACING = 1.0
# Human skeleton, as chains of SOMA bones
CHAINS = [
    ("Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "HeadEnd")
] + [
    chain
    for s in ("Left", "Right")
    for chain in (
        ("Chest", f"{s}Shoulder", f"{s}Arm", f"{s}ForeArm", f"{s}Hand"),
        ("Hips", f"{s}Leg", f"{s}Shin", f"{s}Foot", f"{s}ToeBase", f"{s}ToeEnd"),
    )
]


def main(args):
    data_dir = Path(args.data_dir)
    meta = {r["filename"]: r for r in read_metadata(data_dir)}
    paths = [
        p
        for p in sorted(Path(args.results_dir).glob("*.npz"))
        if name_matches(p.stem, args.match)
    ]
    assert paths, f"No .npz files matching {args.match!r} in {args.results_dir}"
    clips = {path.stem: path for path in paths[: args.max_clips]}
    dt = 1.0 / SEED_FPS

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/grid", width=8.0, height=8.0)
    robots = {}
    for name, y in (("frax", 0.0), ("seed", SPACING)):
        root = server.scene.add_frame(f"/{name}", show_axes=False)
        robots[name] = (root, ViserUrdf(server, G1_URDF, root_node_name=f"/{name}"), y)
    human = server.scene.add_line_segments(
        "/human", np.zeros((1, 2, 3)), (230, 120, 40), thickness=0.02
    )

    clip_dropdown = server.gui.add_dropdown("Clip", tuple(clips))
    description = server.gui.add_markdown("")
    show_seed = server.gui.add_checkbox("Show SEED's G1", True)
    show_human = server.gui.add_checkbox("Show human", True)
    playing = server.gui.add_checkbox("Playing", True)
    speed = server.gui.add_slider(
        "Speed", min=0.1, max=2.0, step=0.1, initial_value=1.0
    )
    frame_slider = server.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)

    clip = None  # (qs, segments) of the current clip

    def load_clip(_=None):
        nonlocal clip
        name = clip_dropdown.value
        row = meta[name]
        # q = [root position, root WXYZ quaternion, joints], per frame
        qs = {
            "frax": np.load(clips[name])["q"],
            "seed": load_g1_csv(data_dir / row["move_g1_path"]),
        }
        poses, names = load_seed(bvh_path(row, data_dir))
        idx = [(names.index(a), names.index(b)) for c in CHAINS for a, b in pairwise(c)]
        segments = poses[:, np.array(idx), :3] + np.array([0.0, -SPACING, 0.0])
        # One assignment, since this runs on viser's thread while the loop plays
        clip = (qs, segments)
        description.content = (
            f"**{row['package']} / {row['category']}**: "
            + row["content_natural_desc_4"]
        )
        frame_slider.max = len(qs["frax"]) - 1
        frame_slider.value = 0

    clip_dropdown.on_update(load_clip)
    load_clip()

    try:
        while True:
            start = time.perf_counter()
            qs, segments = clip
            i = min(int(frame_slider.value), len(qs["frax"]) - 1)
            with server.atomic():
                for name, (root, urdf, y) in robots.items():
                    q = qs[name][min(i, len(qs[name]) - 1)]
                    root.position = q[:3] + np.array([0.0, y, 0.0])
                    root.wxyz = q[3:7]
                    urdf.update_cfg(q[7:])
                robots["seed"][0].visible = show_seed.value
                human.points = segments[min(i, len(segments) - 1)]
                human.visible = show_human.value
            if playing.value:
                frame_slider.value = (i + 1) % len(qs["frax"])
            time.sleep(max(0.0, dt / speed.value - (time.perf_counter() - start)))
    except KeyboardInterrupt:
        pass
    server.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR, help="SEED dataset root")
    parser.add_argument(
        "--results-dir", default=OUT_DIR, help="Retargeted clips from frax.ik"
    )
    parser.add_argument(
        "--match", default="", help="Only clips whose name matches this regex"
    )
    parser.add_argument(
        "--max-clips", type=int, default=1000, help="Max clips listed in the dropdown"
    )
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    main(parser.parse_args())
