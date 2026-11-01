"""Example: Interactive whole-body IK for the Unitree A2 quadruped"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false "
    "--xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED"
)
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse

import jax
import numpy as np
import viser
from viser.extras import ViserUrdf

from frax import load_a2
from frax.assets import A2_ASSETS_DIR
from frax.ik import (
    DiffIKSolver,
    FloorConstraint,
    FrameTask,
    JointLimits,
    JointTask,
)
from frax.utils.rotation_utils import quat_wxyz_to_rmat

A2_URDF = A2_ASSETS_DIR / "a2.urdf"
LEGS = ("FL", "FR", "RL", "RR")
DT = 0.01
GAIN = 20.0
DAMPING = 3e-3


def build_solver(robot):
    # The last floating base joint's frame is the base link's
    base = robot.joint_name_to_index["yaw_joint"]
    calves = [robot.joint_name_to_index[f"{leg}_calf_joint"] for leg in LEGS]
    actuated = np.arange(robot.nv_floating, robot.nv)
    tasks = [
        # Base position + orientation tracking
        FrameTask(
            robot,
            [base],
            gain_pos=GAIN,
            gain_rot=GAIN,
            pos_key="base_pos",
            rot_key="base_rmat",
        ),
        # Foot centers, position only
        FrameTask(
            robot,
            calves,
            gain_pos=GAIN,
            weight_pos=100.0,
            pos_key="foot_pos",
            rot_key=None,
            offsets=robot.front_left_foot_offset,
        ),
        # Weak posture task
        JointTask(robot, actuated, "q_default", gain=1.0, weight=0.1),
    ]
    constraints = [
        JointLimits(robot),
        FloorConstraint(robot, gain=GAIN),
    ]
    return DiffIKSolver(robot, tasks, constraints, damping=DAMPING)


def main(port: int):
    robot = load_a2()
    solver = build_solver(robot)
    step = jax.jit(solver.step)

    server = viser.ViserServer(port=port)
    server.scene.add_grid("/grid", width=3.0, height=3.0)
    root = server.scene.add_frame("/robot", show_axes=False)
    urdf_viz = ViserUrdf(server, A2_URDF, root_node_name="/robot")

    q0 = np.asarray(robot.default_configuration)
    foot_transforms = (
        robot.front_left_foot_transform,
        robot.front_right_foot_transform,
        robot.hind_left_foot_transform,
        robot.hind_right_foot_transform,
    )
    foot_home = [np.asarray(tf(q0))[:3, 3] for tf in foot_transforms]
    base_target = server.scene.add_transform_controls(
        "/base_target", scale=0.3, position=q0[:3], wxyz=q0[3:7]
    )
    foot_targets = [
        server.scene.add_transform_controls(
            f"/{leg}_target", scale=0.12, disable_rotations=True, position=pos
        )
        for leg, pos in zip(LEGS, foot_home, strict=True)
    ]
    reset = server.gui.add_button("Reset")

    state = solver.init(q0)

    @reset.on_click
    def _(_):
        nonlocal state
        state = solver.init(q0)
        base_target.position, base_target.wxyz = q0[:3], q0[3:7]
        for target, pos in zip(foot_targets, foot_home, strict=True):
            target.position = pos

    try:
        while True:
            ref = {
                "base_pos": np.asarray(base_target.position)[None],
                "base_rmat": np.asarray(quat_wxyz_to_rmat(base_target.wxyz))[None],
                "foot_pos": np.array([target.position for target in foot_targets]),
                "q_default": q0[robot.nq_floating :],
            }
            state, _, _ = step(state, ref, DT)
            q = np.asarray(state.q)
            with server.atomic():
                root.position, root.wxyz = q[:3], q[3:7]
                urdf_viz.update_cfg(q[robot.nq_floating :])
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    args = parser.parse_args()
    main(args.port)
