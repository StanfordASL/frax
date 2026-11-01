"""Example: Franka Panda tracking a sinusoidal end-effector motion
as best it can while avoiding a spherical obstacle

The spherical obstacle can be resized and repositioned interactively
in the viser viewer
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

import jax
import jax.numpy as jnp
import numpy as np
import viser
from viser.extras import ViserUrdf

from frax import load_panda
from frax.assets import FRANKA_ASSETS_DIR
from frax.ik import (
    DiffIKSolver,
    EETask,
    JointLimits,
    JointTask,
    ObstacleConstraint,
    SphereObstacles,
)

PANDA_URDF = FRANKA_ASSETS_DIR / "panda.urdf"
DT = 0.01
OBSTACLE_POS = np.array([0.2, 0.2, 0.55])
OBSTACLE_RADIUS = 0.1


def build_solver(robot):
    tasks = [
        EETask(robot, gain_pos=10.0, gain_rot=5.0),
        JointTask(robot, np.arange(7), "q_home", gain=1.0, weight=0.1),
    ]
    obstacle = SphereObstacles(np.zeros((1, 3)), np.ones(1), np.ones(1, dtype=bool))
    constraints = [
        JointLimits(robot),
        ObstacleConstraint(robot, obstacle, ref_key="obstacle", gain=10.0),
    ]
    return DiffIKSolver(robot, tasks, constraints)


def main(port: int):
    robot = load_panda()
    solver = build_solver(robot)
    step = jax.jit(solver.step)
    ee_pos = jax.jit(lambda q: robot.ee_transform(q)[:3, 3])
    sphere_pos = jax.jit(robot.link_collision_positions)
    sphere_radii = np.asarray(robot.flat_collision_radii)

    server = viser.ViserServer(port=port)
    server.scene.add_grid("/grid", width=2.0, height=2.0)
    urdf_viz = ViserUrdf(server, PANDA_URDF, root_node_name="/robot")
    q_home = np.asarray(robot.default_configuration)
    urdf_viz.update_cfg(q_home)

    home = np.asarray(robot.ee_transform(q_home))
    target = server.scene.add_frame(
        "/target", axes_length=0.1, axes_radius=0.005, position=home[:3, 3]
    )
    obstacle = server.scene.add_transform_controls(
        "/obstacle", scale=0.2, disable_rotations=True, position=OBSTACLE_POS
    )
    radius = server.gui.add_slider(
        "Obstacle radius", min=0.02, max=0.2, step=0.01, initial_value=OBSTACLE_RADIUS
    )
    avoid = server.gui.add_checkbox("Avoid obstacle", True)
    show_spheres = server.gui.add_checkbox("Show collision spheres", False)
    status = server.gui.add_markdown("")

    def draw_obstacle(_=None):
        server.scene.add_icosphere(
            "/obstacle/sphere", radius=radius.value, color=(220, 40, 40), opacity=0.6
        )

    radius.on_update(draw_obstacle)
    draw_obstacle()
    spheres = [
        server.scene.add_icosphere(
            f"/spheres/{i}", radius=r, color=(40, 120, 220), opacity=0.4
        )
        for i, r in enumerate(sphere_radii)
    ]

    state = solver.init(q_home)
    t = 0.0
    try:
        while True:
            start = time.perf_counter()
            target.position = home[:3, 3] + [0.0, 0.25 * np.sin(t), 0.0]
            ref = {
                "ee_pos": np.asarray(target.position),
                "ee_rmat": home[:3, :3],
                "q_home": q_home,
                "obstacle": SphereObstacles(
                    jnp.asarray(obstacle.position)[None],
                    jnp.array([radius.value]),
                    jnp.array([avoid.value]),
                ),
            }
            state, _, _ = step(state, ref, DT)
            q = np.asarray(state.q)
            positions = np.asarray(sphere_pos(q))
            clearance = np.linalg.norm(positions - obstacle.position, axis=1)
            clearance -= sphere_radii + radius.value

            with server.atomic():
                urdf_viz.update_cfg(q)
                for sphere, pos in zip(spheres, positions, strict=True):
                    sphere.position = pos
                    sphere.visible = show_spheres.value
            ee_error = np.linalg.norm(ee_pos(q) - ref["ee_pos"])
            status.content = (
                f"Position error: {1000 * ee_error:.0f} mm  \n"
                f"Obstacle clearance: {1000 * clearance.min():.0f} mm"
            )
            t += DT
            time.sleep(max(0.0, DT - (time.perf_counter() - start)))
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    args = parser.parse_args()
    main(args.port)
