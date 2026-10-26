"""Utility functions for rigid body dynamics algorithms

Note: we use the [linear, angular] convention, which differs
from Featherstone's textbook which uses [angular, linear]. But,
[linear, angular] is very common in other libraries like Pinocchio
and MuJoCo.
"""

import jax
import jax.numpy as jnp
from jax import Array

from frax.utils.linalg_utils import skew


def get_spatial_joint_axes(
    joint_transforms: Array,
    joint_axes_local: Array,
    revolute_mask: Array,
) -> Array:
    """Computes the spatial joint axes, expressed in the root frame.

    Args:
        joint_transforms (Array): Transformation matrices for every joint, shape (nv, 4, 4)
        joint_axes_local (Array): Local joint axes, shape (nv, 3)
        revolute_mask (Array): Mask for revolute joints, shape (nv,)

    Returns:
        Array: Spatial joint axes, shape (nv, 6)
    """
    joint_axes_rot = jnp.einsum(
        "qij,qj->qi", joint_transforms[:, :3, :3], joint_axes_local
    )
    joint_pos = joint_transforms[:, :3, 3]

    # For revolute joints, spatial axis is [pos x axis; axis]
    s_rev = jnp.concatenate(
        [jnp.cross(joint_pos, joint_axes_rot), joint_axes_rot], axis=1
    )
    # For prismatic joints, spatial axis is [axis; 0]
    s_pri = jnp.concatenate([joint_axes_rot, jnp.zeros_like(joint_axes_rot)], axis=1)

    return jnp.where(revolute_mask[:, None], s_rev, s_pri)


def _spatial_inertia(mass: Array, local_inertia: Array, transform: Array) -> Array:
    """Helper function: Spatial inertia of a single link, expressed in the root frame

    Args:
        mass (Array): Mass of the link
        local_inertia (Array): Inertia matrix about the COM, in the link frame, shape (3, 3)
        transform (Array): Transformation matrix of the link's COM frame, shape (4, 4)

    Returns:
        Array: Spatial inertia matrix, shape (6, 6)
    """
    R = transform[:3, :3]
    c = transform[:3, 3]
    cx = skew(c)
    # Rotational inertia about the origin (parallel axis theorem)
    I_rot = R @ local_inertia @ R.T + mass * (
        jnp.dot(c, c) * jnp.eye(3) - jnp.outer(c, c)
    )
    # Spatial inertia I_O = [m*I, -m*cx; m*cx, Ic - m*cx*cx]
    mcx = mass * cx
    return jnp.block([[mass * jnp.eye(3), -mcx], [mcx, I_rot]])


def get_spatial_inertias(
    link_masses: Array,
    link_local_inertias: Array,
    link_transforms: Array,
) -> Array:
    """Computes the spatial inertias for each link, expressed in the root frame.

    Args:
        link_masses (Array): Mass of each link, shape (nv,)
        link_local_inertias (Array): Local inertia matrices, shape (nv, 3, 3)
        link_transforms (Array): Transformation matrices for every link, shape (nv, 4, 4)

    Returns:
        Array: Spatial inertia matrices, shape (nv, 6, 6)
    """
    return jax.vmap(_spatial_inertia)(link_masses, link_local_inertias, link_transforms)


def spatial_motion_cross(velocity: Array, motion: Array) -> Array:
    """Computes the spatial cross product for motions

    From Featherstone, eq. 2.33 (adjusted to match [v, w] convention)
    [v0; w] x [m0; m] = [w x m0 + v0 x m; w x m]

    Args:
        velocity (Array): Spatial velocity, shape (..., 6)
        motion (Array): Spatial motion, shape (..., 6)

    Returns:
        Array: Time derivative of the spatial motion, shape (..., 6)
    """
    v0, w = velocity[..., :3], velocity[..., 3:]
    m0, m = motion[..., :3], motion[..., 3:]
    return jnp.concatenate(
        [jnp.cross(w, m0) + jnp.cross(v0, m), jnp.cross(w, m)], axis=-1
    )


def spatial_force_cross(velocity: Array, force: Array) -> Array:
    """Computes the spatial cross product for forces

    From Featherstone, eq. 2.34 (adjusted to match [v, w] convention)
    [v0; w] x [f; f0] = [w x f; w x f0 + v0 x f]

    Args:
        velocity (Array): Spatial velocity, shape (..., 6)
        force (Array): Spatial force, shape (..., 6)

    Returns:
        Array: Time derivative of the spatial force, shape (..., 6)
    """
    v0, w = velocity[..., :3], velocity[..., 3:]
    f, f0 = force[..., :3], force[..., 3:]
    return jnp.concatenate(
        [jnp.cross(w, f), jnp.cross(w, f0) + jnp.cross(v0, f)], axis=-1
    )


def shift_wrenches(wrenches: Array, point: Array) -> Array:
    """Changes the reference point of spatial forces (wrenches)

    A wrench [f; n] taken about point A becomes [f; n - p x f] about point B, where p = B - A

    Args:
        wrenches (Array): Spatial forces about their current reference points, shape (..., 6)
        point (Array): New reference points, relative to the current ones, shape (..., 3)

    Returns:
        Array: Spatial forces about the new reference points, shape (..., 6)
    """
    f, n = wrenches[..., :3], wrenches[..., 3:]
    return jnp.concatenate([f, n - jnp.cross(point, f)], axis=-1)
