"""Example: Interactive end-effector control and IK with the Franka Panda

(this is the best place to get started with frax.ik)
"""

import os

# These environment variables tend to give the best performance
# for JAX on CPU -- we use these in all examples
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

from frax import load_panda
from frax.assets import FRANKA_ASSETS_DIR
from frax.ik import DiffIKSolver, EETask, JointLimits, JointTask
from frax.utils.rotation_utils import quat_wxyz_to_rmat, rmat_to_quat_wxyz

PANDA_URDF = FRANKA_ASSETS_DIR / "panda.urdf"

DT = 0.01
DAMPING = 1e-3


def main(port: int):
    # Load the FRAX robot model
    robot = load_panda()

    # Define tasks (QP objective terms)
    tasks = [
        # The end-effector pose, with frax's analytic Jacobian. Reads the
        # reference keys "ee_pos" and "ee_rmat"
        EETask(robot, gain_pos=10.0, gain_rot=5.0),
        # 7 joints for a 6-DOF pose leave one free motion (the elbow): a weak
        # posture task picks it
        JointTask(robot, np.arange(7), "q_home", gain=1.0, weight=0.1),
    ]

    # Define constraints (QP equality and inequality constraints)
    # We use ElastiQP as our solver, so all constraints are l1-relaxed
    # by default, with a penalty on violation. The l1-relaxed structure
    # means that constraints are (essentially) only ever violated when
    # strictly necessary
    constraints = [JointLimits(robot)]
    # Build the solver. Check out the solver class's docs for
    # info on other tuning parameters such as damping, regularization,
    # tolerances, etc.
    solver = DiffIKSolver(robot, tasks, constraints, damping=DAMPING)

    # The step function in the solver is the main function you'll call,
    # whether on its own or part of a larger function. If it's the
    # top-level function you're calling, make sure to JIT!
    step = jax.jit(solver.step)

    # Set up visualization
    server = viser.ViserServer(port=port)
    server.scene.add_grid("/grid", width=2.0, height=2.0)
    urdf_viz = ViserUrdf(server, PANDA_URDF, root_node_name="/robot")
    q_home = np.asarray(robot.default_configuration)
    urdf_viz.update_cfg(q_home)

    state = solver.init(q_home)
    home = np.asarray(robot.ee_transform(q_home))
    home_pos, home_wxyz = home[:3, 3], np.asarray(rmat_to_quat_wxyz(home[:3, :3]))
    target = server.scene.add_transform_controls(
        "/target", scale=0.2, position=home_pos, wxyz=home_wxyz
    )
    server.scene.add_icosphere(
        "/target/sphere", radius=0.02, color=(220, 40, 40), opacity=0.6
    )
    reset = server.gui.add_button("Reset")

    @reset.on_click
    def _(_):
        nonlocal state
        state = solver.init(q_home)
        target.position, target.wxyz = home_pos, home_wxyz

    try:
        while True:
            # Any new information that you need for your tasks/constraints
            # gets passed in through the ref dict. This should generally
            # just contain values that are updated at each timestep; any
            # static info should be built into the solver
            ref = {
                "ee_pos": np.asarray(target.position),
                "ee_rmat": np.asarray(quat_wxyz_to_rmat(np.asarray(target.wxyz))),
                "q_home": q_home,
            }
            # Call the jitted step function to solve the problem!
            state, _, _ = step(state, ref, DT)
            q = np.asarray(state.q)
            urdf_viz.update_cfg(q)

    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    args = parser.parse_args()
    main(args.port)
