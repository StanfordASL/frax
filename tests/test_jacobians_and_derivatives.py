"""Test cases for analytical jacobians and derivatives, compared against autodiff

For speed, each test builds a single jitted function that computes both the analytical
and autodiff results, vmapped over a batch of random samples. This way, each test only
pays for one compilation, rather than dispatching every autodiff op eagerly

Autodiff is taken w.r.t. the configuration q (shape nq), so it is mapped to the velocity
space (shape nv) via the configuration velocity map E(q). For robots without a
quaternion floating base, E(q) is the identity
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from frax.core.humanoid import Humanoid
from frax.core.manipulator import Manipulator
from frax.core.quadruped import Quadruped
from frax.robots.franka_panda import load_panda
from frax.robots.kuka_iiwa import load_iiwa
from frax.robots.unitree_a2 import load_a2
from frax.robots.unitree_g1 import load_fixed_root_g1, load_g1

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

NUM_SAMPLES = 5

ROBOT_LOADERS = {
    # "g1_euler": lambda: load_g1(floating_base="euler"),
    "g1_quaternion": lambda: load_g1(floating_base="quaternion"),
    "g1_fixed_root": load_fixed_root_g1,
    "a2_euler": lambda: load_a2(floating_base="euler"),
    # "a2_quaternion": lambda: load_a2(floating_base="quaternion"),
    "panda": load_panda,
    "iiwa": load_iiwa,
}


def ee_names(robot):
    """Names of the end-effector frames for a robot, such that the robot has methods
    `{name}_transform`, `{name}_jacobian`, and `{name}_jacobian_and_derivative`"""
    if isinstance(robot, Humanoid):
        return ["left_hand", "right_hand", "left_foot", "right_foot"]
    if isinstance(robot, Quadruped):
        return [
            "front_left_foot",
            "front_right_foot",
            "hind_left_foot",
            "hind_right_foot",
        ]
    if isinstance(robot, Manipulator):
        return ["ee"]
    raise TypeError(f"Unknown robot type: {type(robot)}")


def angular_jacobians_autodiff(tfs_func, q):
    """Computes the angular Jacobians Jw such that w = Jw @ q_dot for a batch of frames,
    using the relationship skew(w) = R_dot @ R^T

    Args:
        tfs_func (Callable): Maps q to a batch of transforms, shape (N, 4, 4)
        q (Array): Configuration, shape (nq,)

    Returns:
        Array: Angular Jacobians, shape (N, 3, nq)
    """
    R = tfs_func(q)[:, :3, :3]
    # dR/dq has shape (N, 3, 3, nq)
    dR_dq = jax.jacfwd(lambda q_in: tfs_func(q_in)[:, :3, :3])(q)
    # Jw_cols[i] = unskew( (dR/dq_i) @ R.T ), shape (N, 3, 3, nq)
    res = jnp.einsum("nijk,nlj->nilk", dR_dq, R)
    return jnp.stack([res[:, 2, 1], res[:, 0, 2], res[:, 1, 0]], axis=1)


@pytest.fixture(scope="module", params=list(ROBOT_LOADERS), ids=list(ROBOT_LOADERS))
def robot(request):
    return ROBOT_LOADERS[request.param]()


@pytest.fixture(scope="module")
def states(robot):
    """Batch of random (q, qd) samples, with shapes (NUM_SAMPLES, nq) and (NUM_SAMPLES, nv)"""
    rng = np.random.default_rng(42)
    qs = rng.uniform(-1.0, 1.0, (NUM_SAMPLES, robot.nq))
    qds = rng.uniform(-1.0, 1.0, (NUM_SAMPLES, robot.nv))
    if robot.is_quaternion_base:
        qs[:, 3:7] /= np.linalg.norm(qs[:, 3:7], axis=1, keepdims=True)
    return qs, qds


def test_center_of_mass_jacobian(robot, states):
    """Test COM Jacobian against autodiff of COM position"""
    qs, _ = states

    @jax.jit
    @jax.vmap
    def compute(q):
        E = robot.configuration_velocity_map(q)
        J_analytical = robot.center_of_mass_jacobian(q)
        J_autodiff = jax.jacfwd(robot.center_of_mass)(q) @ E
        return J_analytical, J_autodiff

    J_analytical, J_autodiff = compute(qs)
    np.testing.assert_allclose(
        J_analytical, J_autodiff, atol=1e-7, err_msg="COM Jacobian mismatch"
    )


def test_ee_jacobians(robot, states, subtests):
    """Test EE Jacobians (linear & angular) against autodiff"""
    qs, _ = states
    names = ee_names(robot)

    @jax.jit
    @jax.vmap
    def compute(q):
        E = robot.configuration_velocity_map(q)
        results = {}
        for name in names:
            jac_func = getattr(robot, f"{name}_jacobian")
            tf_func = getattr(robot, f"{name}_transform")
            J_analytical = jac_func(q)
            Jv_autodiff = jax.jacfwd(lambda q_in: tf_func(q_in)[:3, 3])(q)
            Jw_autodiff = angular_jacobians_autodiff(
                lambda q_in: tf_func(q_in)[None], q
            )[0]
            J_autodiff = jnp.vstack([Jv_autodiff, Jw_autodiff]) @ E
            results[name] = (J_analytical, J_autodiff)
        return results

    for name, (J_analytical, J_autodiff) in compute(qs).items():
        with subtests.test(ee=name):
            np.testing.assert_allclose(
                J_analytical, J_autodiff, atol=1e-7, err_msg=f"{name} Jacobian mismatch"
            )


def test_ee_jacobian_derivatives(robot, states, subtests):
    """Test EE Jacobian derivatives (linear & angular) against JVP of Jacobian"""
    qs, qds = states
    names = ee_names(robot)

    @jax.jit
    @jax.vmap
    def compute(q, qd):
        q_dot = robot.configuration_velocity_map(q) @ qd
        results = {}
        for name in names:
            jac_dot_func = getattr(robot, f"{name}_jacobian_and_derivative")
            jac_func = getattr(robot, f"{name}_jacobian")
            # Analytical Jdot
            _, Jdot_analytical = jac_dot_func(q, qd)
            # Autodiff Jdot via JVP of analytical Jacobian function
            _, Jdot_autodiff = jax.jvp(jac_func, (q,), (q_dot,))
            results[name] = (Jdot_analytical, Jdot_autodiff)
        return results

    for name, (Jdot_analytical, Jdot_autodiff) in compute(qs, qds).items():
        with subtests.test(ee=name):
            np.testing.assert_allclose(
                Jdot_analytical,
                Jdot_autodiff,
                atol=1e-6,
                err_msg=f"{name} Jdot mismatch",
            )


def test_link_jacobians(robot, states):
    """Test internal _link_linear_jacobians and _link_angular_jacobians against autodiff"""
    qs, _ = states

    @jax.jit
    @jax.vmap
    def compute(q):
        E = robot.configuration_velocity_map(q)
        tfs = robot.joint_to_world_transforms(q)
        Jvs_analytical = robot._link_linear_jacobians(tfs)
        Jws_analytical = robot._link_angular_jacobians(tfs)
        Jvs_autodiff = jax.jacfwd(robot.link_com_positions)(q) @ E
        Jws_autodiff = angular_jacobians_autodiff(robot.link_to_world_transforms, q) @ E
        return Jvs_analytical, Jws_analytical, Jvs_autodiff, Jws_autodiff

    Jvs_analytical, Jws_analytical, Jvs_autodiff, Jws_autodiff = compute(qs)
    np.testing.assert_allclose(
        Jvs_analytical,
        Jvs_autodiff,
        atol=1e-7,
        err_msg="Link linear Jacobians mismatch",
    )
    np.testing.assert_allclose(
        Jws_analytical,
        Jws_autodiff,
        atol=1e-7,
        err_msg="Link angular Jacobians mismatch",
    )


def test_joint_jacobians(robot, states):
    """Test internal _joint_jacobians (linear & angular) against autodiff"""
    qs, _ = states

    @jax.jit
    @jax.vmap
    def compute(q):
        E = robot.configuration_velocity_map(q)
        tfs = robot.joint_to_world_transforms(q)
        Jvs_analytical, Jws_analytical = robot._joint_jacobians(tfs)
        Jvs_autodiff = (
            jax.jacfwd(lambda q_in: robot.joint_to_world_transforms(q_in)[:, :3, 3])(q)
            @ E
        )
        Jws_autodiff = (
            angular_jacobians_autodiff(robot.joint_to_world_transforms, q) @ E
        )
        return Jvs_analytical, Jws_analytical, Jvs_autodiff, Jws_autodiff

    Jvs_analytical, Jws_analytical, Jvs_autodiff, Jws_autodiff = compute(qs)
    np.testing.assert_allclose(
        Jvs_analytical,
        Jvs_autodiff,
        atol=1e-7,
        err_msg="Joint linear Jacobians mismatch",
    )
    np.testing.assert_allclose(
        Jws_analytical,
        Jws_autodiff,
        atol=1e-7,
        err_msg="Joint angular Jacobians mismatch",
    )
