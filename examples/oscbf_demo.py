"""Operational space control demo with FRAX kinematics + dynamics"""

import os

os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["JAX_ENABLE_X64"] = "True"
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse

import numpy as np
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from frax.core.manipulator import Manipulator
from frax import load_panda, load_iiwa
from frax.utils.rotation_utils import orientation_error_3D
from example_utils import ManipulatorEnv
from cbf_utils import OSCBFTorqueConfig
from cbfpy import CBF


STARTUP_MSG = """
====================================
Initializing OSCBF demo.

Please wait a few seconds to fully
load and JIT-compile the robot and
controller
====================================
"""

INFO_MSG = """
====================================
Demo ready.

To control the end-effector position:
- Double-click on the target
- Right-click-and-drag to perturb it

This demo enforces the following constraints:
- Singularity avoidance
- Joint limit avoidance
- Collision avoidance with the floor
- Collision avoidance with an external obstacle
- Self-collision avoidance
====================================
"""


@jax.tree_util.register_static
class DemoOSCBFConfig(OSCBFTorqueConfig):
    def __init__(
        self,
        robot: Manipulator,
        collision_positions: ArrayLike,
        collision_radii: ArrayLike,
    ):
        self.q_min = robot.joint_lower_limits
        self.q_max = robot.joint_upper_limits
        self.singularity_tol = 1e-3
        self.collision_positions = np.atleast_2d(collision_positions)
        self.collision_radii = np.ravel(collision_radii)
        assert len(collision_positions) == len(collision_radii)
        self.num_collision_bodies = len(collision_positions)
        super().__init__(robot)

    def h_2(self, z, **kwargs):
        # Extract values
        q = z[: self.num_joints]
        q_min = jnp.asarray(self.q_min)
        q_max = jnp.asarray(self.q_max)

        # Joint Limit Avoidance
        h_joint_limits = jnp.concatenate([q_max - q, q - q_min])

        # Singularity Avoidance
        sigmas = jax.lax.linalg.svd(self.robot.ee_jacobian(q), compute_uv=False)
        h_singularity = jnp.array([jnp.prod(sigmas) - self.singularity_tol])

        # Collision Avoidance
        robot_collision_positions, robot_collision_radii = (
            self.robot.link_collision_data(q)
        )
        center_deltas = (
            robot_collision_positions[:, None, :] - self.collision_positions[None, :, :]
        ).reshape(-1, 3)
        radii_sums = (
            robot_collision_radii[:, None] + self.collision_radii[None, :]
        ).reshape(-1)
        h_collision = jnp.linalg.norm(center_deltas, axis=1) - radii_sums

        # Floor avoidance
        floor_height = 0.01  # Slight offset to be able to see it a bit better
        h_floor = robot_collision_positions[:, 2] - robot_collision_radii - floor_height

        # Self-collision avoidance
        # Note: using already computed positions of the spheres
        h_self_collision = self.robot._self_collision_distances_from_link_data(
            robot_collision_positions, robot_collision_radii
        )

        return jnp.concatenate(
            [
                h_joint_limits,
                h_singularity,
                h_collision,
                h_self_collision,
                h_floor,
            ]
        )

    def alpha(self, h):
        return 10.0 * h

    def alpha_2(self, h_2):
        return 10.0 * h_2


def main(robot_name):
    assert robot_name in {"panda", "iiwa"}

    print(STARTUP_MSG)
    env = ManipulatorEnv(
        robot=robot_name, control_mode="torque", real_time=True, load_obstacle=True
    )

    if robot_name == "panda":
        robot = load_panda()
    else:  # iiwa
        robot = load_iiwa()

    # Define gains
    kp_pos = 50.0 * np.ones(3)
    kp_rot = 20.0 * np.ones(3)
    kd_pos = 20.0 * np.ones(3)
    kd_rot = 10.0 * np.ones(3)
    kp_task = np.concatenate([kp_pos, kp_rot])
    kd_task = np.concatenate([kd_pos, kd_rot])
    kp_joint = 10.0 * np.ones(robot.num_joints)
    kd_joint = 5.0 * np.ones(robot.num_joints)

    # Define nullspace posture task
    is_redundant = True  # 6DOF task, 7DOF robot
    des_q = env.q_init
    des_qdot = np.zeros(robot.num_joints)

    # Define acceleration terms for EE task
    des_accel = np.zeros(3)
    des_alpha = np.zeros(3)

    # Define CBF
    collision_pos = np.array([[0.5, 0.5, 0.5]])
    collision_radii = np.array([0.3])
    cbf_config = DemoOSCBFConfig(robot, collision_pos, collision_radii)
    cbf = CBF.from_config(cbf_config)

    @jax.jit
    def operational_space_control(z, z_ee_des):
        # Extract state info
        q = z[: robot.num_joints]
        qdot = z[robot.num_joints :]
        des_pos = z_ee_des[:3]
        des_rot = jnp.reshape(z_ee_des[3:12], (3, 3))
        des_vel = z_ee_des[12:15]
        des_omega = z_ee_des[15:18]

        # FRAX Kinematics + Dynamics
        M, M_inv, g, c, J, ee_tmat = robot.torque_control_matrices(q, qdot)
        pos = ee_tmat[:3, 3]
        rot = ee_tmat[:3, :3]

        # Compute twist
        twist = J @ qdot
        vel = twist[:3]
        omega = twist[3:]

        # Errors
        pos_error = pos - des_pos
        vel_error = vel - des_vel
        rot_error = orientation_error_3D(rot, des_rot)
        omega_error = omega - des_omega
        task_p_error = jnp.concatenate([pos_error, rot_error])
        task_d_error = jnp.concatenate([vel_error, omega_error])

        # Operational space matrices
        task_inertia_inv = J @ M_inv @ J.T
        task_inertia = jnp.linalg.inv(task_inertia_inv)
        J_bar = M_inv @ J.T @ task_inertia

        # Compute operational space task torques
        task_accel = (
            jnp.concatenate([des_accel, des_alpha])
            - kp_task * task_p_error
            - kd_task * task_d_error
        )
        task_wrench = task_inertia @ task_accel
        tau = J.T @ task_wrench

        # Add compensation for nonlinear effects
        tau += g + c

        if is_redundant:
            # Nullspace projection
            NT = jnp.eye(robot.num_joints) - J.T @ J_bar.T
            # Add nullspace joint task
            q_error = q - des_q
            qdot_error = qdot - des_qdot
            joint_accel = -kp_joint * q_error - kd_joint * qdot_error
            secondary_joint_torques = M @ joint_accel
            tau += NT @ secondary_joint_torques

        # Enforce CBFs
        return cbf.safety_filter(z, tau)

    # Initial call for JIT
    _ = operational_space_control(np.zeros(14), np.zeros(18))

    print(INFO_MSG)

    try:
        while env.viewer.is_running():
            z = env.get_joint_state()
            z_ee_des = env.get_desired_ee_state()
            u = operational_space_control(z, z_ee_des)
            env.apply_control(u)
            env.step()
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=["panda", "iiwa"], default="panda")
    args = parser.parse_args()
    main(args.robot)
