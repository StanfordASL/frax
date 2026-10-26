import jax
from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp
import numpy as np

from frax.utils.linalg_utils import skew, skew_numpy


def create_transform(rotation: ArrayLike, translation: ArrayLike) -> Array:
    """Create a transformation matrix from a rotation matrix and a translation vector

    Args:
        rotation (ArrayLike): Rotation matrix, shape (3, 3)
        translation (ArrayLike): Translation vector, shape (3,)

    Returns:
        Array: Transformation matrix, shape (4, 4)
    """
    return jnp.block(
        [
            [jnp.asarray(rotation), jnp.asarray(translation).reshape(-1, 1)],
            [jnp.array([[0.0, 0.0, 0.0, 1.0]])],
        ]
    )


def create_transform_numpy(rotation: ArrayLike, translation: ArrayLike) -> np.ndarray:
    """Create a transformation matrix from a rotation matrix and a translation vector

    Args:
        rotation (ArrayLike): Rotation matrix, shape (3, 3)
        translation (ArrayLike): Translation vector, shape (3,)

    Returns:
        Array: Transformation matrix, shape (4, 4)
    """
    return np.block(
        [
            [np.asarray(rotation), np.asarray(translation).reshape(-1, 1)],
            [np.array([[0.0, 0.0, 0.0, 1.0]])],
        ]
    )


def revolute_transform(q: float, axis: ArrayLike) -> jax.Array:
    """Create a transformation matrix for a revolute joint (child frame --> parent frame)

    Args:
        q (float): Joint angle
        axis (ArrayLike): Joint axis (in joint frame), shape (3,)

    Returns:
        jax.Array: Transformation matrix, shape (4, 4)
    """
    axis = jnp.asarray(axis)
    axis = axis / jnp.linalg.norm(axis)
    a1, a2, a3 = axis
    c = jnp.cos(q)
    s = jnp.sin(q)
    t = 1 - c
    return jnp.array(
        [
            [t * a1 * a1 + c, t * a1 * a2 - s * a3, t * a1 * a3 + s * a2, 0.0],
            [t * a1 * a2 + s * a3, t * a2 * a2 + c, t * a2 * a3 - s * a1, 0.0],
            [t * a1 * a3 - s * a2, t * a2 * a3 + s * a1, t * a3 * a3 + c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def prismatic_transform(q: float, axis: ArrayLike) -> jax.Array:
    """Create a transformation matrix for a prismatic joint (child frame --> parent frame)

    Args:
        q (float): Joint position
        axis (ArrayLike): Joint axis (in joint frame), shape (3,)

    Returns:
        jax.Array: Transformation matrix, shape (4, 4)
    """
    axis = jnp.asarray(axis)
    translation = q * axis / jnp.linalg.norm(axis)
    return jnp.array(
        [
            [1.0, 0.0, 0.0, translation[0]],
            [0.0, 1.0, 0.0, translation[1]],
            [0.0, 0.0, 1.0, translation[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def transform_point(transform: ArrayLike, point: ArrayLike) -> Array:
    """Apply a transformation matrix to a point

    Args:
        transform (ArrayLike): Transformation matrix, shape (4, 4)
        point (ArrayLike): Point to transform, shape (3,)

    Returns:
        Array: Transformed point, shape (3,)
    """
    transform = jnp.asarray(transform)
    point = jnp.asarray(point)
    assert transform.shape == (4, 4)
    assert point.shape == (3,)
    return transform[:3, :3] @ point + transform[:3, 3]


def transform_points(transform: ArrayLike, points: ArrayLike) -> Array:
    """Apply a transformation matrix to a set of points

    Args:
        transform (ArrayLike): Transformation matrix, shape (4, 4)
        points (ArrayLike): Points to transform, shape (n, 3)

    Returns:
        Array: Transformed points, shape (n, 3)
    """
    transform = jnp.asarray(transform)
    points = jnp.asarray(points)
    assert transform.shape == (4, 4)
    assert points.shape[1] == 3
    return points @ transform[:3, :3].T + transform[:3, 3]


def joint_transform(
    joint_pos: float, joint_axis: ArrayLike, joint_type: float
) -> Array:
    """Create a transformation matrix for a joint (child frame --> parent frame) based on the joint type

    Args:
        joint_pos (float): Joint position
        joint_axis (ArrayLike): Joint axis (in joint frame), shape (3,)
        joint_type (float): Joint type (0 for revolute, 1 for prismatic)

    Returns:
        Array: Transformation matrix, shape (4, 4)
    """
    return (1.0 - joint_type) * revolute_transform(
        joint_pos, joint_axis
    ) + joint_type * prismatic_transform(joint_pos, joint_axis)


def transform_jacobian(J: Array, r: Array) -> Array:
    """Transform a jacobian to another point on the same body

    Args:
        J (Array): Jacobian, shape (6, n)
        r (Array): Vector from the original point on the body where the Jacobian
            was defined, to the new point of interest. NOTE: This vector
            must be defined in the same reference frame as the Jacobian. Typically,
            this is WORLD frame. Shape (3,)

    Returns:
        Array: New Jacobian, shape (6, n)
    """
    assert J.shape[0] == 6
    assert J.ndim == 2
    assert r.shape == (3,)
    Jv = J[:3, :]
    Jw = J[3:, :]
    rx = skew(r)
    Jv_new = Jv - rx @ Jw
    return jnp.vstack([Jv_new, Jw])


def transform_jacobian_numpy(J: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Transform a jacobian to another point on the same body

    Args:
        J (np.ndarray): Jacobian, shape (6, n)
        r (np.ndarray): Vector from the original point on the body where the Jacobian
            was defined, to the new point of interest. NOTE: This vector
            must be defined in the same reference frame as the Jacobian. Typically,
            this is WORLD frame. Shape (3,)

    Returns:
        np.ndarray: New Jacobian, shape (6, n)
    """
    assert J.shape[0] == 6
    assert J.ndim == 2
    assert r.shape == (3,)
    Jv = J[:3, :]
    Jw = J[3:, :]
    rx = skew_numpy(r)
    Jv_new = Jv - rx @ Jw
    return np.vstack([Jv_new, Jw])
