"""Utility functions for rigid body dynamics algorithms

Note: we use the [linear, angular] convention, which differs
from Featherstone's textbook which uses [angular, linear]. But,
[linear, angular] is very common in other libraries like Pinocchio
and MuJoCo.
"""

import jax.numpy as jnp
from jax import Array


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


# TODO clean up this function more
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
    m = link_masses
    R_i = link_transforms[:, :3, :3]
    link_pos = link_transforms[:, :3, 3]

    Ic = jnp.einsum(
        "ijk,ikl,ilm->ijm",
        R_i,
        link_local_inertias,
        R_i.transpose(0, 2, 1),
    )
    c = link_pos
    cx = jnp.stack(
        [
            jnp.stack([jnp.zeros_like(c[:, 0]), -c[:, 2], c[:, 1]], axis=1),
            jnp.stack([c[:, 2], jnp.zeros_like(c[:, 0]), -c[:, 0]], axis=1),
            jnp.stack([-c[:, 1], c[:, 0], jnp.zeros_like(c[:, 0])], axis=1),
        ],
        axis=1,
    )

    # Spatial inertia I_O = [m*I, -m*cx; m*cx, Ic - m*cx*cx]
    I_O = jnp.concatenate(
        [
            jnp.concatenate(
                [
                    m[:, None, None] * jnp.eye(3)[None, :, :],
                    -m[:, None, None] * cx,
                ],
                axis=2,
            ),
            jnp.concatenate(
                [m[:, None, None] * cx, Ic - m[:, None, None] * (cx @ cx)], axis=2
            ),
        ],
        axis=1,
    )
    return I_O


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
