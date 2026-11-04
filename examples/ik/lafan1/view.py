"""Play the retargeted LAFAN1 dataset in a viser viewer"""

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
from pathlib import Path

import numpy as np
import viser
from lafan1_g1 import LAFAN1_FPS, OUT_DIR
from viser.extras import ViserUrdf

from frax.assets import G1_ASSETS_DIR

G1_URDF = G1_ASSETS_DIR / "g1_29dof_rev_1_0.urdf"


def main(args):
    paths = sorted(Path(args.frax_dir).glob("*.npz"))
    assert paths, f"No .npz files in {args.frax_dir}"
    clips = {path.stem: path for path in paths}
    dt = 1.0 / LAFAN1_FPS

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/grid", width=6.0, height=6.0)
    root = server.scene.add_frame("/robot", show_axes=False)
    urdf_viz = ViserUrdf(server, G1_URDF, root_node_name="/robot")

    clip_dropdown = server.gui.add_dropdown("Clip", tuple(clips))
    playing = server.gui.add_checkbox("Playing", True)
    speed = server.gui.add_slider(
        "Speed", min=0.1, max=2.0, step=0.1, initial_value=1.0
    )
    frame_slider = server.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)

    qs = None  # The current clip's q = [root position, root WXYZ quaternion, joints]

    def load_clip(_=None):
        nonlocal qs
        qs = np.load(clips[clip_dropdown.value])["q"]
        frame_slider.max = len(qs) - 1
        frame_slider.value = 0

    clip_dropdown.on_update(load_clip)
    load_clip()

    try:
        while True:
            start = time.perf_counter()
            i = min(int(frame_slider.value), len(qs) - 1)
            q = qs[i]
            with server.atomic():
                root.position = q[:3]
                root.wxyz = q[3:7]
                urdf_viz.update_cfg(q[7:])
            if playing.value:
                frame_slider.value = (i + 1) % len(qs)
            time.sleep(max(0.0, dt / speed.value - (time.perf_counter() - start)))
    except KeyboardInterrupt:
        pass
    server.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frax-dir", default=OUT_DIR, help="Retargeted clips (.npz)")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    main(parser.parse_args())
