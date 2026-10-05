"""Test cases for rotation utils"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from frax.utils.rotation_utils import (
    Rx,
    Ry,
    Rz,
    extrinsic_euler_xyz_to_quat_wxyz,
    intrinsic_euler_xyz_to_quat_wxyz,
    omega_body_from_wxyz_quaternions,
    omega_world_from_wxyz_quaternions,
    orientation_error_3D,
    quat_wxyz_conjugate,
    quat_wxyz_exp,
    quat_wxyz_log,
    quat_wxyz_multiply,
    quat_wxyz_to_extrinsic_euler_xyz,
    quat_wxyz_to_intrinsic_euler_xyz,
    quat_wxyz_to_rmat,
    rmat_to_quat_wxyz,
    rotate_vector_by_quat_wxyz,
    slerp,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)

NUM_SAMPLES = 20
TOL = 1e-10


def random_quat():
    q = np.random.randn(4)
    return q / np.linalg.norm(q)


def random_euler():
    # Keep pitch away from +/- pi/2 to avoid gimbal lock
    return np.random.uniform([-np.pi, -1.4, -np.pi], [np.pi, 1.4, np.pi])


def assert_same_rotation(q1, q2, tol=TOL):
    # q and -q represent the same rotation
    sign = np.sign(np.dot(q1, q2))
    np.testing.assert_allclose(q1, sign * q2, atol=tol)


@pytest.mark.parametrize(
    "R_fn, axis",
    [(Rx, 0), (Ry, 1), (Rz, 2)],
    ids=["Rx", "Ry", "Rz"],
)
def test_elementary_rotations(R_fn, axis):
    for _ in range(NUM_SAMPLES):
        theta = np.random.uniform(-np.pi, np.pi)
        R = R_fn(theta)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=TOL)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=TOL)
        # The rotation axis is unchanged
        e = np.eye(3)[axis]
        np.testing.assert_allclose(R @ e, e, atol=TOL)
        # Rotations about the same axis compose additively
        np.testing.assert_allclose(R @ R_fn(0.3), R_fn(theta + 0.3), atol=TOL)


def test_quat_to_rmat_against_elementary_rotations():
    for _ in range(NUM_SAMPLES):
        theta = np.random.uniform(-np.pi, np.pi)
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        np.testing.assert_allclose(quat_wxyz_to_rmat([c, s, 0, 0]), Rx(theta), atol=TOL)
        np.testing.assert_allclose(quat_wxyz_to_rmat([c, 0, s, 0]), Ry(theta), atol=TOL)
        np.testing.assert_allclose(quat_wxyz_to_rmat([c, 0, 0, s]), Rz(theta), atol=TOL)


def test_quat_rmat_round_trip():
    for _ in range(NUM_SAMPLES):
        q = random_quat()
        assert_same_rotation(rmat_to_quat_wxyz(quat_wxyz_to_rmat(q)), q)


@pytest.mark.parametrize(
    "R",
    [
        np.eye(3),
        np.diag([1.0, -1.0, -1.0]),  # 180 deg about x
        np.diag([-1.0, 1.0, -1.0]),  # 180 deg about y
        np.diag([-1.0, -1.0, 1.0]),  # 180 deg about z
    ],
    ids=["identity", "pi_x", "pi_y", "pi_z"],
)
def test_rmat_to_quat_branches(R):
    # Exercises each branch of the trace-based conversion
    np.testing.assert_allclose(quat_wxyz_to_rmat(rmat_to_quat_wxyz(R)), R, atol=TOL)


def test_intrinsic_euler():
    for _ in range(NUM_SAMPLES):
        r, p, y = euler = random_euler()
        q = intrinsic_euler_xyz_to_quat_wxyz(euler)
        # Intrinsic XYZ: post-multiply about the body axes
        np.testing.assert_allclose(
            quat_wxyz_to_rmat(q), Rx(r) @ Ry(p) @ Rz(y), atol=TOL
        )
        np.testing.assert_allclose(quat_wxyz_to_intrinsic_euler_xyz(q), euler, atol=TOL)


def test_extrinsic_euler():
    for _ in range(NUM_SAMPLES):
        a, b, g = euler = random_euler()
        q = extrinsic_euler_xyz_to_quat_wxyz(euler)
        # Extrinsic XYZ: pre-multiply about the world axes
        np.testing.assert_allclose(
            quat_wxyz_to_rmat(q), Rz(g) @ Ry(b) @ Rx(a), atol=TOL
        )
        np.testing.assert_allclose(quat_wxyz_to_extrinsic_euler_xyz(q), euler, atol=TOL)


def test_quat_convention_swaps():
    q = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(wxyz_to_xyzw(q), [2.0, 3.0, 4.0, 1.0])
    np.testing.assert_array_equal(xyzw_to_wxyz(q), [4.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(xyzw_to_wxyz(wxyz_to_xyzw(q)), q)


def test_conjugate_and_multiply():
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    for _ in range(NUM_SAMPLES):
        q1, q2 = random_quat(), random_quat()
        np.testing.assert_allclose(
            quat_wxyz_multiply(q1, quat_wxyz_conjugate(q1)), identity, atol=TOL
        )
        np.testing.assert_allclose(quat_wxyz_multiply(q1, identity), q1, atol=TOL)
        # Quaternion product corresponds to rotation matrix product
        np.testing.assert_allclose(
            quat_wxyz_to_rmat(quat_wxyz_multiply(q1, q2)),
            quat_wxyz_to_rmat(q1) @ quat_wxyz_to_rmat(q2),
            atol=TOL,
        )


def test_rotate_vector():
    for _ in range(NUM_SAMPLES):
        q = random_quat()
        v = np.random.randn(3)
        np.testing.assert_allclose(
            rotate_vector_by_quat_wxyz(q, v), quat_wxyz_to_rmat(q) @ v, atol=TOL
        )
        # Unnormalized quaternions are handled
        np.testing.assert_allclose(
            rotate_vector_by_quat_wxyz(3.0 * q, v), quat_wxyz_to_rmat(q) @ v, atol=TOL
        )


def test_slerp():
    for _ in range(NUM_SAMPLES):
        q1, q2 = random_quat(), random_quat()
        assert_same_rotation(slerp(q1, q2, 0.0), q1)
        assert_same_rotation(slerp(q1, q2, 1.0), q2)
        # Midpoint is equidistant (in angle) from both ends
        mid = slerp(q1, q2, 0.5)
        np.testing.assert_allclose(
            np.linalg.norm(
                quat_wxyz_log(quat_wxyz_multiply(quat_wxyz_conjugate(q1), mid))
            ),
            np.linalg.norm(
                quat_wxyz_log(quat_wxyz_multiply(quat_wxyz_conjugate(mid), q2))
            ),
            atol=1e-8,
        )
        # Equivalent to scaling the relative rotation vector
        t = np.random.uniform()
        q_rel = quat_wxyz_multiply(quat_wxyz_conjugate(q1), q2)
        expected = quat_wxyz_multiply(q1, quat_wxyz_exp(t * quat_wxyz_log(q_rel)))
        assert_same_rotation(slerp(q1, q2, t), expected, tol=1e-8)


def test_slerp_small_angle():
    q = random_quat()
    out = slerp(q, q, 0.5)
    assert np.all(np.isfinite(out))
    assert_same_rotation(out, q)


@pytest.mark.parametrize(
    "frame, omega_fn",
    [
        ("world", omega_world_from_wxyz_quaternions),
        ("body", omega_body_from_wxyz_quaternions),
    ],
    ids=["world", "body"],
)
def test_omega_from_quaternions(frame, omega_fn):
    dt = 0.01
    for _ in range(NUM_SAMPLES):
        q1 = random_quat()
        omega = np.random.uniform(-1.0, 1.0, 3)
        dq = quat_wxyz_exp(omega * dt)
        if frame == "world":
            q2 = quat_wxyz_multiply(dq, q1)
        else:
            q2 = quat_wxyz_multiply(q1, dq)
        np.testing.assert_allclose(omega_fn(q1, q2, dt), omega, atol=1e-8)
        # Sign flip of q2 does not change the result (shortest path)
        np.testing.assert_allclose(omega_fn(q1, -q2, dt), omega, atol=1e-8)
    # Zero rotation gives zero velocity
    np.testing.assert_allclose(omega_fn(q1, q1, dt), np.zeros(3), atol=TOL)


def test_orientation_error():
    R = quat_wxyz_to_rmat(random_quat())
    np.testing.assert_allclose(orientation_error_3D(R, R), np.zeros(3), atol=TOL)
    for _ in range(NUM_SAMPLES):
        R_cur = quat_wxyz_to_rmat(random_quat())
        # For small (world-frame) perturbations, the error matches the rotation vector
        delta = np.random.uniform(-1.0, 1.0, 3) * 1e-4
        R_des = quat_wxyz_to_rmat(quat_wxyz_exp(delta)) @ R_cur
        np.testing.assert_allclose(
            orientation_error_3D(R_cur, R_des), -delta, rtol=1e-3, atol=1e-10
        )


def test_exp_log_round_trip():
    for _ in range(NUM_SAMPLES):
        # Angles within (0, pi), so the log map returns the same rotation vector
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        rotvec = axis * np.random.uniform(0.0, np.pi - 1e-3)
        q = quat_wxyz_exp(rotvec)
        np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=TOL)
        np.testing.assert_allclose(quat_wxyz_log(q), rotvec, atol=TOL)
        q = random_quat()
        assert_same_rotation(quat_wxyz_exp(quat_wxyz_log(q)), q)


def test_exp_matches_rmat():
    for _ in range(NUM_SAMPLES):
        theta = np.random.uniform(-np.pi, np.pi)
        np.testing.assert_allclose(
            quat_wxyz_to_rmat(quat_wxyz_exp(np.array([0.0, 0.0, theta]))),
            Rz(theta),
            atol=TOL,
        )


def test_log_shortest_path():
    q = random_quat()
    np.testing.assert_allclose(quat_wxyz_log(-q), quat_wxyz_log(q), atol=TOL)
    assert np.linalg.norm(quat_wxyz_log(q)) <= np.pi + TOL


@pytest.mark.parametrize("scale", [0.0, 1e-6, 1e-4 * (1 - 1e-6), 1e-4 * (1 + 1e-6)])
def test_exp_log_small_angles(scale):
    # Values near the Taylor expansion threshold should be continuous and finite
    rotvec = scale * np.array([0.6, 0.0, 0.8])
    q = quat_wxyz_exp(rotvec)
    # sin(theta/2)/theta, written via np.sinc to be well-defined at zero
    s = 0.5 * np.sinc(scale / (2 * np.pi))
    expected = np.concatenate([[np.cos(scale / 2)], s * rotvec])
    np.testing.assert_allclose(q, expected, atol=1e-15)
    np.testing.assert_allclose(quat_wxyz_log(q), rotvec, atol=1e-15)


def test_exp_log_gradients_at_zero():
    # The Taylor branches should avoid NaN gradients at the identity
    jac_exp = jax.jacobian(quat_wxyz_exp)(jnp.zeros(3))
    jac_log = jax.jacobian(quat_wxyz_log)(jnp.array([1.0, 0.0, 0.0, 0.0]))
    assert np.all(np.isfinite(jac_exp))
    assert np.all(np.isfinite(jac_log))
    np.testing.assert_allclose(jac_exp[1:], 0.5 * np.eye(3), atol=TOL)
    np.testing.assert_allclose(jac_log[:, 1:], 2.0 * np.eye(3), atol=TOL)
