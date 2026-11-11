"""Play the retargeted HiPHI dataset in a viser viewer

This script also shows the original human skeleton, beside the robot, and
for human-object interaction clips, the tracked objects: at the human's
scale beside the human, and scaled like the robot's targets (by the clip's
leg scale) beside the robot

This script also visualizes the original human skeleton, and for any
clips that are interacting with an object, it also shows the objects
(re-scaled accordingly as well)
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import threading
import time
from pathlib import Path

import numpy as np
import trimesh
import viser
from dataset import (
    OUT_DIR,
    add_selection_args,
    bvh_path,
    load_clip,
    load_hiphi,
    load_objects,
    selected_clips,
)
from hiphi_g1 import HIPHI_ANKLE_HEIGHT, HIPHI_FPS, body_parents, floor_offset
from viser.extras import ViserUrdf

from frax.assets import G1_ASSETS_DIR

G1_URDF = G1_ASSETS_DIR / "g1_29dof_rev_1_0.urdf"
# Sideways offset (m) of the human from the robot
SPACING = 1.0
OBJECT_COLOR = (90, 140, 200)


def load_mesh(path: Path, cache: dict) -> tuple[np.ndarray, np.ndarray]:
    """An OBJ's vertices (m) and faces"""
    if path not in cache:
        mesh = trimesh.load(path, force="mesh", process=False)
        cache[path] = (np.asarray(mesh.vertices) * 0.01, np.asarray(mesh.faces))
    return cache[path]


def main(args):
    rows = {r["motion_id"]: r for r in selected_clips(args)}
    paths = [p for p in sorted(Path(args.results_dir).glob("*.npz")) if p.stem in rows]
    assert paths, f"No selected clips' .npz files in {args.results_dir}"
    clips = {path.stem: path for path in paths[: args.max_clips]}
    dt = 1.0 / HIPHI_FPS
    meshes = {}

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/grid", width=8.0, height=8.0)
    root = server.scene.add_frame("/frax", show_axes=False)
    urdf = ViserUrdf(server, G1_URDF, root_node_name="/frax")
    human = server.scene.add_line_segments(
        "/human", np.zeros((1, 2, 3)), (230, 120, 40), thickness=0.02
    )

    clip_dropdown = server.gui.add_dropdown("Clip", tuple(clips))
    description = server.gui.add_markdown("")
    show_human = server.gui.add_checkbox("Show human", True)
    show_objects = server.gui.add_checkbox("Show objects", True)
    playing = server.gui.add_checkbox("Playing", True)
    speed = server.gui.add_slider(
        "Speed", min=0.1, max=2.0, step=0.1, initial_value=1.0
    )
    frame_slider = server.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)

    clip = None  # (q, segments, [(handle, poses)]) of the current clip
    # show_clip runs on viser's thread while the loop plays: the lock keeps the
    # loop off object handles as they're removed and replaced
    lock = threading.Lock()

    def show_clip(_=None):
        nonlocal clip
        name = clip_dropdown.value
        row = rows[name]
        # q = [root position, root WXYZ quaternion, joints], per frame
        q = np.load(clips[name])["q"]
        bvh = bvh_path(row, args.data_dir)
        poses, names = load_hiphi(bvh)
        # The human's clip measurements, as the retargeter saw them
        measured = {k: v[0] for k, v in load_clip(bvh, row).items()}
        leg_scale = measured["leg_scale"]
        # The human and their objects drop by the clip's floor height (as
        # measured from its planted ankles), and the robot's objects scale and
        # drop like its targets
        human_drop = measured["ankle_height"] - HIPHI_ANKLE_HEIGHT
        idx = {b: i for i, b in enumerate(names)}
        bones = [(idx[p], idx[b]) for b, p in body_parents(names).items()]
        segments = poses[:, np.array(bones), :3]
        segments = segments + np.array([0.0, -SPACING, -human_drop])
        tracks = [
            (load_mesh(obj["mesh"], meshes), obj["poses"])
            for obj in load_objects(row, args.data_dir)
        ]
        with lock:
            # The old meshes go first, as the new ones reuse their names
            if clip is not None:
                for handle, _ in clip[2]:
                    handle.remove()
            objects = []
            for k, ((vertices, faces), obj_poses) in enumerate(tracks):
                for where, scale, offset in (
                    ("human", 1.0, [0.0, -SPACING, -human_drop]),
                    ("frax", leg_scale, [0.0, 0.0, -floor_offset(measured)]),
                ):
                    handle = server.scene.add_mesh_simple(
                        f"/objects/{where}_{k}",
                        vertices * scale,
                        faces,
                        color=OBJECT_COLOR,
                    )
                    placed = obj_poses.copy()
                    placed[:, :3] = placed[:, :3] * scale + offset
                    objects.append((handle, placed))
            clip = (q, segments, objects)
        description.content = (
            f"**{row['frame_lu']}** ({row['actor_id']}): " + row["text_annotation"]
        )
        frame_slider.max = len(q) - 1
        frame_slider.value = 0

    clip_dropdown.on_update(show_clip)
    show_clip()

    try:
        while True:
            start = time.perf_counter()
            with lock, server.atomic():
                q, segments, objects = clip
                i = min(int(frame_slider.value), len(q) - 1)
                root.position = q[i, :3]
                root.wxyz = q[i, 3:7]
                urdf.update_cfg(q[i, 7:])
                human.points = segments[min(i, len(segments) - 1)]
                human.visible = show_human.value
                for handle, poses in objects:
                    pose = poses[min(i, len(poses) - 1)]
                    handle.position, handle.wxyz = pose[:3], pose[3:]
                    handle.visible = show_objects.value
            if playing.value:
                frame_slider.value = (i + 1) % len(q)
            time.sleep(max(0.0, dt / speed.value - (time.perf_counter() - start)))
    except KeyboardInterrupt:
        pass
    server.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_selection_args(parser)
    parser.add_argument(
        "--results-dir", default=OUT_DIR, help="Retargeted clips from frax.ik"
    )
    parser.add_argument(
        "--max-clips", type=int, default=1000, help="Max clips listed in the dropdown"
    )
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    main(parser.parse_args())
