"""Differential inverse kinematics demo with FRAX kinematics + dynamics"""

import argparse

import numpy as np
import jax
import jax.numpy as jnp

from frax import load_panda, load_iiwa
from frax.utils.rotation_utils import orientation_error_3D
from example_utils import ManipulatorEnv, SinusoidalTaskTrajectory

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

MANUAL_MODE_MSG = """
====================================
Starting demo in MANUAL mode

To control the end-effector position:
- Double-click on the target
- Right-click-and-drag to perturb it

NOTE: This demo does not include safety constraints
(for example, collision/singularity avoidance).
For a full handling of these, see the OSCBF
demo in this repo, as well as:
https://github.com/StanfordASL/oscbf

To switch to TRAJECTORY mode, run the script
with the --mode trajectory flag
====================================
"""

TRAJ_MODE_MSG = """
====================================
Starting demo in TRAJECTORY mode

This will show the controller tracking
a sinusoidal task-space position trajectory
while maintaining a fixed EE orientation

To switch to MANUAL mode, run the script
with the --mode manual flag
====================================
"""


def main(robot_name, demo_mode):
    assert demo_mode in {"trajectory", "manual"}
    assert robot_name in {"panda", "iiwa"}

    # TODO: move this somewhere else
    # Also, update these to match the FK result at q init
    if robot_name == "panda":
        robot = load_panda()
        init_rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    else:  # iiwa
        robot = load_iiwa()
        init_rot = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

    if demo_mode == "trajectory":
        print(TRAJ_MODE_MSG)
        traj = SinusoidalTaskTrajectory(
            init_pos=(0.4, 0, 0.4),
            init_rot=init_rot,
            amplitude=(0, 0.25, 0),
            angular_freq=(0, 2, 0),
            phase=(0, 0, 0),
        )
    else:
        print(MANUAL_MODE_MSG)
        traj = None

    env = ManipulatorEnv(
        robot=robot_name, control_mode="velocity", traj=traj, real_time=True
    )

    # Define gains
    kp_pos = 5.0 * np.ones(3)
    kp_rot = 2.0 * np.ones(3)
    kp_task = np.concatenate([kp_pos, kp_rot])
    kp_joint = 1.0 * np.ones(robot.num_joints)

    # Define nullspace posture task
    is_redundant = True  # 6DOF task, 7DOF robot
    des_q = env.q_init

    @jax.jit
    def differential_ik(q, z_ee_des):
        # Extract state info
        des_pos = z_ee_des[:3]
        des_rot = jnp.reshape(z_ee_des[3:12], (3, 3))
        des_vel = z_ee_des[12:15]
        des_omega = z_ee_des[15:18]

        # FRAX Kinematics (+ inverse mass matrix for dynamic consistency)
        M_inv, J, ee_tmat = robot.dynamically_consistent_velocity_control_matrices(q)
        pos = ee_tmat[:3, 3]
        rot = ee_tmat[:3, :3]

        # Errors
        pos_error = pos - des_pos
        rot_error = orientation_error_3D(rot, des_rot)
        task_p_error = jnp.concatenate([pos_error, rot_error])

        # Jacobian inversion
        if M_inv is None:
            J_hash = jnp.linalg.pinv(J)  # "J pseudo"
        else:
            task_inertia_inv = J @ M_inv @ J.T
            task_inertia = jnp.linalg.inv(task_inertia_inv)
            J_hash = M_inv @ J.T @ task_inertia  # "J bar"

        # Compute task velocities
        task_vel = jnp.concatenate([des_vel, des_omega]) - kp_task * task_p_error

        # Map to joint velocities
        v = J_hash @ task_vel

        if is_redundant:
            # Nullspace projection
            N = jnp.eye(robot.num_joints) - J_hash @ J
            # Add nullspace joint task
            q_error = q - des_q
            secondary_joint_vel = -kp_joint * q_error
            v += N @ secondary_joint_vel

        # Clamp to velocity limits
        return jnp.clip(
            v,
            -1.0 * jnp.asarray(robot.joint_max_velocities),
            jnp.asarray(robot.joint_max_velocities),
        )

    try:
        while env.viewer.is_running():
            z = env.get_joint_state()
            z_ee_des = env.get_desired_ee_state()
            q = z[: robot.num_joints]
            u = differential_ik(q, z_ee_des)
            env.apply_control(u)
            env.step()
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=["panda", "iiwa"], default="panda")
    parser.add_argument(
        "--mode", choices=["trajectory", "manual"], default="trajectory"
    )
    args = parser.parse_args()
    main(args.robot, args.mode)
