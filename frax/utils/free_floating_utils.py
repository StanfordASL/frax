"""Conversion utilties between various ways of specifying the 6DOF state
of the free-floating root, and its 6DOF twist

Note: in the future, we'll be working with imu data, which will be in body frame.
In this case, remember that omega_world = R @ omega_body

Note the different quaternion conventions used by different packages.
- Mujoco: WXYZ
- Pinocchio: XYZW
"""


import jax.numpy as jnp
from jax import Array

from frax.utils.rotation_utils import (
    intrinsic_euler_xyz_to_quat_wxyz,
    quat_wxyz_to_intrinsic_euler_xyz,
    rotate_vector_by_quat_wxyz,
    xyzw_to_wxyz,
)


def virtual_joints_to_pose_and_twist(
    q: Array, qd: Array
) -> tuple[Array, Array, Array, Array]:
    """Convert the state of 6 virtual joints (3 prismatic, 3 revolute)
    to the world-frame pose and twist of the free-floating root

    Args:
        q (Array): Virtual joint positions, shape (6,)
        qd (Array): Virtual joint velocities, shape (6,)

    Returns:
        pos (Array): Position in world frame, shape (3,)
        quat_wxyz (Array): WXYZ quaternion in world frame, shape (4,)
        vel (Array): Linear velocity in world frame, shape (3,)
        omega (Array): Angular velocity in world frame, shape (3,)
    """
    q = jnp.asarray(q)
    qd = jnp.asarray(qd)

    pos = q[:3]
    euler = q[3:]
    vel = qd[:3]
    edot = qd[3:]  # dr/dt, dp/dt, dy/dt

    quat_wxyz = intrinsic_euler_xyz_to_quat_wxyz(euler)

    r, p, y = euler

    cr, sr = jnp.cos(r), jnp.sin(r)
    cp, sp = jnp.cos(p), jnp.sin(p)

    # Mapping matrix M such that omega = M * edot
    M = jnp.array([[1.0, 0.0, sp], [0.0, cr, -sr * cp], [0.0, sr, cr * cp]])

    omega = M @ edot

    return pos, quat_wxyz, vel, omega


def pose_and_twist_to_virtual_joints(
    pos: Array, quat_wxyz: Array, vel: Array, omega: Array
) -> tuple[Array, Array]:
    """Convert the world-frame state of the free-floating root (pose and twist)
    to the joint state of 6 virtual joints (3 prismatic, 3 revolute)

    Args:
        pos (Array): Position in world frame, shape (3,)
        quat_wxyz (Array): WXYZ quaternion in world frame, shape (4,)
        vel (Array): Linear velocity in world frame, shape (3,)
        omega (Array): Angular velocity in world frame, shape (3,)

    Returns:
        q (Array): Virtual joint positions, shape (6,)
        qd (Array): Virtual joint velocities, shape (6,)
    """
    pos = jnp.asarray(pos)
    vel = jnp.asarray(vel)
    omega = jnp.asarray(omega)

    euler = quat_wxyz_to_intrinsic_euler_xyz(quat_wxyz)
    r, p, y = euler

    q = jnp.concatenate([pos, euler])

    cr, sr = jnp.cos(r), jnp.sin(r)
    cp, sp = jnp.cos(p), jnp.sin(p)

    M = jnp.array([[1.0, 0.0, sp], [0.0, cr, -sr * cp], [0.0, sr, cr * cp]])

    edot = jnp.linalg.solve(M, omega)
    qd = jnp.concatenate([vel, edot])

    return q, qd


##########
# Conversions from mujoco and pinocchio's qpos and qvel
# Note that this assumes that either the qpos/qvel passed in is only
# associated with the ff joint, or that the ff joint is the first joint
##########


# TODO: make numpy versions of these -- likely, these will not be used under jit


def mujoco_qpos_qvel_to_pose_and_twist(
    qpos: Array, qvel: Array
) -> tuple[Array, Array, Array, Array]:
    """Convert MuJoCo's representation of a free-floating joint's generalized
    coordinates and velocities to a world-frame pose and twist

    Args:
        qpos (Array): MuJoCo's generalized joint coordinates for the free-floating
            joint, shape (7,) (position and WXYZ quaternion in world frame)
        qvel (Array): MuJoCo's generalized joint velocities for the free-floating
            joint, shape (6,) (linear and angular velocities. Linear = world, angular = body)

    Returns:
        pos (Array): Position in world frame, shape (3,)
        quat_wxyz (Array): WXYZ quaternion in world frame, shape (4,)
        vel (Array): Linear velocity in world frame, shape (3,)
        omega (Array): Angular velocity in world frame, shape (3,)
    """
    qpos = jnp.asarray(qpos)
    qvel = jnp.asarray(qvel)
    pos_world = qpos[:3]
    quat_wxyz_world = qpos[3:7]
    vel_world = qvel[:3]
    omega_body = qvel[3:6]
    omega_world = rotate_vector_by_quat_wxyz(quat_wxyz_world, omega_body)
    return pos_world, quat_wxyz_world, vel_world, omega_world


def pinocchio_qpos_qvel_to_pose_and_twist(
    qpos: Array, qvel: Array
) -> tuple[Array, Array, Array, Array]:
    """Convert pinocchio's representation of a free-floating joint's generalized
    coordinates and velocities to a world-frame pose and twist

    Args:
        qpos (Array): Pinocchio's generalized joint coordinates for the free-floating
            joint, shape (7,) (position and XYZW quaternion in world frame)
        qvel (Array): Pinocchio's generalized joint velocities for the free-floating
            joint, shape (6,) (linear and angular velocities in body frame)

    Returns:
        pos (Array): Position in world frame, shape (3,)
        quat_wxyz (Array): WXYZ quaternion in world frame, shape (4,)
        vel (Array): Linear velocity in world frame, shape (3,)
        omega (Array): Angular velocity in world frame, shape (3,)
    """
    qpos = jnp.asarray(qpos)
    qvel = jnp.asarray(qvel)
    pos_world = qpos[:3]
    quat_xyzw_world = qpos[3:7]
    quat_wxyz_world = xyzw_to_wxyz(quat_xyzw_world)
    vel_body = qvel[:3]
    omega_body = qvel[3:6]
    vel_world = rotate_vector_by_quat_wxyz(quat_wxyz_world, vel_body)
    omega_world = rotate_vector_by_quat_wxyz(quat_wxyz_world, omega_body)
    return pos_world, quat_wxyz_world, vel_world, omega_world
