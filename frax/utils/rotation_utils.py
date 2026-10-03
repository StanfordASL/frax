"""Rotation representation conversions

Note the different quaternion conventions used by different packages.
- Mujoco: WXYZ
- Pinocchio: XYZW

Also note the difference between intrinsic and extrinsic Euler angles
- Intrinsic: rotations occur about the body frame axes as they rotate
- Extrinsic: rotations occur about the fixed world frame axes
"""

import jax.numpy as jnp
from jax import Array


def Rx(theta: float) -> Array:
    """Rotation matrix for a rotation about the x axis

    Args:
        theta (float): Angle in radians

    Returns:
        Array: Rotation matrix, shape (3, 3)
    """
    return jnp.array(
        [
            [1, 0, 0],
            [0, jnp.cos(theta), -jnp.sin(theta)],
            [0, jnp.sin(theta), jnp.cos(theta)],
        ]
    )


def Ry(theta: float) -> Array:
    """Rotation matrix for a rotation about the y axis

    Args:
        theta (float): Angle in radians

    Returns:
        Array: Rotation matrix, shape (3, 3)
    """
    return jnp.array(
        [
            [jnp.cos(theta), 0, jnp.sin(theta)],
            [0, 1, 0],
            [-jnp.sin(theta), 0, jnp.cos(theta)],
        ]
    )


def Rz(theta: float) -> Array:
    """Rotation matrix for a rotation about the z axis

    Args:
        theta (float): Angle in radians

    Returns:
        Array: Rotation matrix, shape (3, 3)
    """
    return jnp.array(
        [
            [jnp.cos(theta), -jnp.sin(theta), 0],
            [jnp.sin(theta), jnp.cos(theta), 0],
            [0, 0, 1],
        ]
    )


def quat_wxyz_to_intrinsic_euler_xyz(quat_wxyz: Array) -> Array:
    """Convert a WXYZ quaternion to intrinsic XYZ Euler angles

    Intrinsic XYZ implies a rotation about the body frame x-axis,
    then the body y-axis, and finally the body z-axis

    Args:
        quat_wxyz (Array): WXYZ quaternion, shape (4,)

    Returns:
        Array: Intrinsic XYZ Euler angles, shape (3,)
    """
    quat_wxyz = jnp.asarray(quat_wxyz)
    quat_wxyz = quat_wxyz / jnp.linalg.norm(quat_wxyz)
    w, x, y, z = quat_wxyz
    roll = jnp.arctan2(-2 * (y * z - w * x), 1 - 2 * (x * x + y * y))
    pitch = jnp.arcsin(jnp.clip(2 * (x * z + w * y), -1.0, 1.0))
    yaw = jnp.arctan2(-2 * (x * y - w * z), 1 - 2 * (y * y + z * z))
    return jnp.array([roll, pitch, yaw])


def intrinsic_euler_xyz_to_quat_wxyz(euler: Array) -> Array:
    """Convert intrinsic XYZ Euler angles to WXYZ quaternion

    Intrinsic XYZ implies a rotation about the body frame x-axis,
    then the body y-axis, and finally the body z-axis

    Args:
        euler (Array): Intrinsic XYZ Euler angles, shape (3,)

    Returns:
        Array: WXYZ quaternion, shape (4,)
    """
    euler = jnp.asarray(euler)
    r, p, y = euler / 2.0

    cr, sr = jnp.cos(r), jnp.sin(r)
    cp, sp = jnp.cos(p), jnp.sin(p)
    cy, sy = jnp.cos(y), jnp.sin(y)

    w = cr * cp * cy - sr * sp * sy
    x = sr * cp * cy + cr * sp * sy
    y = cr * sp * cy - sr * cp * sy
    z = cr * cp * sy + sr * sp * cy

    return jnp.array([w, x, y, z])


def quat_wxyz_to_extrinsic_euler_xyz(quat_wxyz: Array) -> Array:
    """Convert a WXYZ quaternion to extrinsic XYZ Euler angles.

    Extrinsic XYZ implies rotations are performed about the fixed world
    axes: first X, then Y, then Z.

    Args:
        quat_wxyz (Array): WXYZ quaternion, shape (4,)

    Returns:
        Array: Extrinsic XYZ Euler angles (alpha, beta, gamma), shape (3,)
    """
    quat_wxyz = jnp.asarray(quat_wxyz)
    quat_wxyz = quat_wxyz / jnp.linalg.norm(quat_wxyz)
    w, x, y, z = quat_wxyz
    alpha = jnp.arctan2(2 * (x * w + y * z), 1 - 2 * (x**2 + y**2))
    beta = jnp.arcsin(jnp.clip(2 * (y * w - z * x), -1.0, 1.0))
    gamma = jnp.arctan2(2 * (z * w + x * y), 1 - 2 * (y**2 + z**2))
    return jnp.array([alpha, beta, gamma])


def extrinsic_euler_xyz_to_quat_wxyz(euler: Array) -> Array:
    """Convert extrinsic XYZ Euler angles to WXYZ quaternion.

    Extrinsic XYZ implies rotations are performed about the fixed world
    axes: first X, then Y, then Z.

    Args:
        euler (Array): Extrinsic XYZ Euler angles, shape (3,)

    Returns:
        Array: WXYZ quaternion, shape (4,)
    """
    euler = jnp.asarray(euler)
    ax, ay, az = euler / 2.0

    cx, sx = jnp.cos(ax), jnp.sin(ax)
    cy, sy = jnp.cos(ay), jnp.sin(ay)
    cz, sz = jnp.cos(az), jnp.sin(az)

    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz

    return jnp.array([w, x, y, z])


def quat_wxyz_to_rmat(quat_wxyz: Array) -> Array:
    """Convert WXYZ quaternion to a rotation matrix

    Args:
        quat_wxyz (Array): WXYZ quaternion, shape (4,)

    Returns:
        Array: Rotation matrix, shape (3, 3)
    """
    quat_wxyz = jnp.asarray(quat_wxyz)
    quat_wxyz = quat_wxyz / jnp.linalg.norm(quat_wxyz)
    w, x, y, z = quat_wxyz
    return jnp.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def rmat_to_quat_wxyz(R: Array) -> Array:
    """Convert a rotation matrix to an WXYZ quaternion

    Args:
        R (Array): Rotation matrix, shape (3, 3)

    Returns:
        Array: WXYZ quaternion, shape (4,)
    """
    R = jnp.asarray(R)

    tr = R[0, 0] + R[1, 1] + R[2, 2]

    def case_tr_gt_0():
        s = jnp.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
        return jnp.array([w, x, y, z])

    def case_x_max():
        s = jnp.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
        return jnp.array([w, x, y, z])

    def case_y_max():
        s = jnp.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
        return jnp.array([w, x, y, z])

    def case_z_max():
        s = jnp.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
        return jnp.array([w, x, y, z])

    # Choose the most stable case
    cond1 = tr > 0
    cond2 = (R[0, 0] > R[1, 1]) & (R[0, 0] > R[2, 2])
    cond3 = R[1, 1] > R[2, 2]

    res = jnp.where(
        cond1,
        case_tr_gt_0(),
        jnp.where(
            cond2,
            case_x_max(),
            jnp.where(
                cond3,
                case_y_max(),
                case_z_max(),
            ),
        ),
    )
    return res / jnp.linalg.norm(res)


def slerp(start: Array, end: Array, t: float) -> Array:
    """Spherical linear interpolation between two quaternions

    Note that the quaternion convention (WXYZ vs XYZW) does not matter,
    but the start and end must have the same convention

    Args:
        start (Array): Starting quaternion, shape (4,)
        end (Array): Ending quaternion, shape (4,)
        t (float): Interpolation parameter, 0 <= t <= 1.

    Returns:
        Array: Interpolated quaternion(s), shape (4,) or (n, 4)
    """
    assert start.shape == (4,)
    assert end.shape == (4,)
    start = start / jnp.linalg.norm(start)
    end = end / jnp.linalg.norm(end)

    # Ensure shortest path
    dot_prod = jnp.dot(start, end)
    end = jnp.where(dot_prod < 0.0, -end, end)
    dot_prod = jnp.abs(dot_prod)  # Update to reflect the flip

    angle = jnp.arccos(jnp.clip(dot_prod, -1.0, 1.0))

    # Linear interpolation for small angles
    small_angle = angle < 1e-6
    w1_linear = 1.0 - t
    w2_linear = t

    # SLERP for larger angles
    # Note: avoid division by zero to prevent NaN via "double where" trick
    # see https://github.com/jax-ml/jax/issues/5039
    safe_sin = jnp.where(small_angle, 1.0, jnp.sin(angle))
    w1_slerp = jnp.sin((1.0 - t) * angle) / safe_sin
    w2_slerp = jnp.sin(t * angle) / safe_sin

    # Apply LERP/SLERP weights and re-normalize
    w1 = jnp.where(small_angle, w1_linear, w1_slerp)
    w2 = jnp.where(small_angle, w2_linear, w2_slerp)
    out = w1 * start + w2 * end
    return out / jnp.linalg.norm(out)


def quat_wxyz_conjugate(q: Array) -> Array:
    """Conjugate of a WXYZ quaternion

    Args:
        q (Array): WXYZ quaternion, shape (4,)

    Returns:
        Array: WXYZ quaternion, shape (4,)
    """
    return jnp.array([q[0], -q[1], -q[2], -q[3]])


def quat_wxyz_multiply(q1: Array, q2: Array) -> Array:
    """Multiplication of two WXYZ quaternions

    Args:
        q1 (Array): WXYZ quaternion, shape (4,)
        q2 (Array): WXYZ quaternion, shape (4,)

    Returns:
        Array: WXYZ quaternion, shape (4,)
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return jnp.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _omega_from_wxyz_quaternions(
    q1: Array, q2: Array, dt: float, frame: str = "world"
) -> Array:
    """Helper function for computing angular velocity from two quaternions"""
    q1 = q1 / jnp.linalg.norm(q1)
    q2 = q2 / jnp.linalg.norm(q2)

    # Ensure shortest path
    q2 = jnp.where(jnp.dot(q1, q2) < 0.0, -q2, q2)

    if frame == "world":
        q_rel = quat_wxyz_multiply(q2, quat_wxyz_conjugate(q1))
    else:  # body
        q_rel = quat_wxyz_multiply(quat_wxyz_conjugate(q1), q2)

    w, v = q_rel[0], q_rel[1:]
    v_norm = jnp.linalg.norm(v)

    # Axis-angle with check for numerical stability
    theta = 2.0 * jnp.atan2(v_norm, w)
    axis = jnp.where(v_norm > 1e-8, v / v_norm, jnp.zeros(3))
    return axis * (theta / dt)


def omega_world_from_wxyz_quaternions(q1: Array, q2: Array, dt: float) -> Array:
    """Compute the world frame angular velocity from two WXYZ quaternions

    Args:
        q1 (Array): Starting/previous WXYZ quaternion, shape (4,)
        q2 (Array): Ending/current WXYZ quaternion, shape (4,)
        dt (float): Time between quaternion measurements

    Returns:
        Array: Angular velocity in world frame, shape (3,)
    """
    return _omega_from_wxyz_quaternions(q1, q2, dt, "world")


def omega_body_from_wxyz_quaternions(q1: Array, q2: Array, dt: float) -> Array:
    """Compute the body frame angular velocity from two WXYZ quaternions

    Args:
        q1 (Array): Starting/previous WXYZ quaternion, shape (4,)
        q2 (Array): Ending/current WXYZ quaternion, shape (4,)
        dt (float): Time between quaternion measurements

    Returns:
        Array: Angular velocity in body frame, shape (3,)
    """
    return _omega_from_wxyz_quaternions(q1, q2, dt, "body")


def orientation_error_3D(R_cur: Array, R_des: Array) -> Array:
    """Determine the angular error vector between two rotation matrices in 3D.

    Args:
        R_cur (Array): Current rotation matrix, shape (3, 3)
        R_des (Array): Desired rotation matrix, shape (3, 3)

    Returns:
        Array: Angular error, shape (3,)
    """
    return -0.5 * (
        jnp.cross(R_cur[:, 0], R_des[:, 0])
        + jnp.cross(R_cur[:, 1], R_des[:, 1])
        + jnp.cross(R_cur[:, 2], R_des[:, 2])
    )


def rotate_vector_by_quat_wxyz(q: Array, v: Array) -> Array:
    """Applies a rotation (defined by a WXYZ quaternion) to a vector

    Args:
        q (Array): WXYZ quaternion, shape (4,)
        v (Array): Vector to rotate, shape (3,)

    Returns:
        Array: Rotated vector, shape (3,)
    """
    q = q / jnp.linalg.norm(q)
    w = q[0]
    u = q[1:]
    u_x_v = jnp.cross(u, v)
    u_x_u_x_v = jnp.cross(u, u_x_v)
    return v + 2 * w * u_x_v + 2 * u_x_u_x_v


def xyzw_to_wxyz(quat: Array) -> Array:
    """Convert an XYZW quaternion to WXYZ

    Args:
        quat (Array): XYZW quaternion, shape (4,)

    Returns:
        Array: WXYZ quaternion, shape (4,)
    """
    return quat[jnp.array([3, 0, 1, 2])]


def wxyz_to_xyzw(quat: Array) -> Array:
    """Convert a WXYZ quaternion to XYZW

    Args:
        quat (Array): WXYZ quaternion, shape (4,)

    Returns:
        Array: XYZW quaternion, shape (4,)
    """
    return quat[jnp.array([1, 2, 3, 0])]


def quat_wxyz_exp(rotvec: Array) -> Array:
    """Exponential map: rotation vector (axis * angle) to a WXYZ quaternion

    This is safe to differentiate through at zero rotation

    Args:
        rotvec (Array): Rotation vector, shape (3,)

    Returns:
        Array: WXYZ quaternion, shape (4,)
    """
    theta_sq = jnp.dot(rotvec, rotvec)
    is_small = theta_sq < 1e-8
    theta = jnp.sqrt(jnp.where(is_small, 1.0, theta_sq))
    # Taylor expansions of cos(theta/2) and sin(theta/2)/theta for small angles
    w = jnp.where(is_small, 1.0 - theta_sq / 8.0, jnp.cos(theta / 2.0))
    s = jnp.where(is_small, 0.5 - theta_sq / 48.0, jnp.sin(theta / 2.0) / theta)
    return jnp.concatenate([jnp.array([w]), s * rotvec])


def quat_wxyz_log(quat_wxyz: Array) -> Array:
    """Logarithmic map: WXYZ quaternion to a rotation vector (axis * angle), via the shortest path

    This is safe to differentiate through at zero rotation

    Args:
        quat_wxyz (Array): WXYZ quaternion, shape (4,)

    Returns:
        Array: Rotation vector, shape (3,)
    """
    quat_wxyz = quat_wxyz / jnp.linalg.norm(quat_wxyz)
    # Ensure shortest path (q and -q represent the same rotation)
    quat_wxyz = jnp.where(quat_wxyz[0] < 0.0, -quat_wxyz, quat_wxyz)
    w, v = quat_wxyz[0], quat_wxyz[1:]
    s_sq = jnp.dot(v, v)
    is_small = s_sq < 1e-8
    s = jnp.sqrt(jnp.where(is_small, 1.0, s_sq))
    # Taylor expansion of 2 * atan2(s, w) / s for small angles
    scale = jnp.where(
        is_small, 2.0 / w * (1.0 - s_sq / (3.0 * w * w)), 2.0 * jnp.atan2(s, w) / s
    )
    return scale * v
