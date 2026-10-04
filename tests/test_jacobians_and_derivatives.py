"""Test cases for analytical jacobians and derivatives, compared against autodiff

For speed, each test builds a single jitted function that computes both the analytical
and autodiff results, vmapped over a batch of random samples. This way, each test only
pays for one compilation, rather than dispatching every autodiff op eagerly
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from frax.robots.unitree_g1 import load_g1

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

NUM_SAMPLES = 5


def angular_jacobians_autodiff(tfs_func, q):
    """Computes the angular Jacobians Jw such that w = Jw @ qd for a batch of frames,
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


class TestJacobiansAndDerivatives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = load_g1(floating_base="euler")
        cls.nq = cls.robot.nq
        cls.nv = cls.robot.nv
        rng = np.random.default_rng(42)
        cls.qs = rng.uniform(-1.0, 1.0, (NUM_SAMPLES, cls.nq))
        cls.qds = rng.uniform(-1.0, 1.0, (NUM_SAMPLES, cls.nv))

    def test_center_of_mass_jacobian(self):
        """Test COM Jacobian against autodiff of COM position"""
        robot = self.robot

        @jax.jit
        @jax.vmap
        def compute(q):
            J_analytical = robot.center_of_mass_jacobian(q)
            J_autodiff = jax.jacfwd(robot.center_of_mass)(q)
            return J_analytical, J_autodiff

        J_analytical, J_autodiff = compute(self.qs)
        for i in range(NUM_SAMPLES):
            with self.subTest(sample=i):
                np.testing.assert_allclose(
                    J_analytical[i], J_autodiff[i], atol=1e-7, err_msg="COM Jacobian mismatch"
                )

    def test_ee_jacobians(self):
        """Test EE Jacobians (linear & angular) against autodiff"""
        robot = self.robot
        # fmt: off
        ee_funcs = {
            "left_hand": (robot.left_hand_jacobian, robot.left_hand_transform),
            "right_hand": (robot.right_hand_jacobian, robot.right_hand_transform),
            "left_foot": (robot.left_foot_jacobian, robot.left_foot_transform),
            "right_foot": (robot.right_foot_jacobian, robot.right_foot_transform),
        }
        # fmt: on

        @jax.jit
        @jax.vmap
        def compute(q):
            results = {}
            for name, (jac_func, tf_func) in ee_funcs.items():
                J_analytical = jac_func(q)
                Jv_autodiff = jax.jacfwd(lambda q_in: tf_func(q_in)[:3, 3])(q)
                Jw_autodiff = angular_jacobians_autodiff(
                    lambda q_in: tf_func(q_in)[None], q
                )[0]
                J_autodiff = jnp.vstack([Jv_autodiff, Jw_autodiff])
                results[name] = (J_analytical, J_autodiff)
            return results

        results = compute(self.qs)
        for name, (J_analytical, J_autodiff) in results.items():
            for i in range(NUM_SAMPLES):
                with self.subTest(ee=name, sample=i):
                    np.testing.assert_allclose(
                        J_analytical[i],
                        J_autodiff[i],
                        atol=1e-7,
                        err_msg=f"{name} Jacobian mismatch",
                    )

    def test_ee_jacobian_derivatives(self):
        """Test EE Jacobian derivatives (linear & angular) against JVP of Jacobian"""
        robot = self.robot
        # fmt: off
        ee_funcs = {
            "left_hand": (robot.left_hand_jacobian_and_derivative, robot.left_hand_jacobian),
            "right_hand": (robot.right_hand_jacobian_and_derivative, robot.right_hand_jacobian),
            "left_foot": (robot.left_foot_jacobian_and_derivative, robot.left_foot_jacobian),
            "right_foot": (robot.right_foot_jacobian_and_derivative, robot.right_foot_jacobian),
        }
        # fmt: on

        @jax.jit
        @jax.vmap
        def compute(q, qd):
            results = {}
            for name, (jac_dot_func, jac_func) in ee_funcs.items():
                # Analytical Jdot
                _, Jdot_analytical = jac_dot_func(q, qd)
                # Autodiff Jdot via JVP of analytical Jacobian function
                _, Jdot_autodiff = jax.jvp(jac_func, (q,), (qd,))
                results[name] = (Jdot_analytical, Jdot_autodiff)
            return results

        results = compute(self.qs, self.qds)
        for name, (Jdot_analytical, Jdot_autodiff) in results.items():
            for i in range(NUM_SAMPLES):
                with self.subTest(ee=name, sample=i):
                    np.testing.assert_allclose(
                        Jdot_analytical[i],
                        Jdot_autodiff[i],
                        atol=1e-6,
                        err_msg=f"{name} Jdot mismatch",
                    )

    def test_link_jacobians(self):
        """Test internal _link_linear_jacobians and _link_angular_jacobians against autodiff"""
        robot = self.robot

        @jax.jit
        @jax.vmap
        def compute(q):
            tfs = robot.joint_to_world_transforms(q)
            Jvs_analytical = robot._link_linear_jacobians(tfs)
            Jws_analytical = robot._link_angular_jacobians(tfs)
            Jvs_autodiff = jax.jacfwd(robot.link_com_positions)(q)
            Jws_autodiff = angular_jacobians_autodiff(robot.link_to_world_transforms, q)
            return Jvs_analytical, Jws_analytical, Jvs_autodiff, Jws_autodiff

        Jvs_analytical, Jws_analytical, Jvs_autodiff, Jws_autodiff = compute(self.qs)
        for i in range(NUM_SAMPLES):
            with self.subTest(sample=i):
                np.testing.assert_allclose(
                    Jvs_analytical[i],
                    Jvs_autodiff[i],
                    atol=1e-7,
                    err_msg="Link linear Jacobians mismatch",
                )
                np.testing.assert_allclose(
                    Jws_analytical[i],
                    Jws_autodiff[i],
                    atol=1e-7,
                    err_msg="Link angular Jacobians mismatch",
                )

    def test_joint_jacobians(self):
        """Test internal _joint_jacobians (linear & angular) against autodiff"""
        robot = self.robot

        @jax.jit
        @jax.vmap
        def compute(q):
            tfs = robot.joint_to_world_transforms(q)
            Jvs_analytical, Jws_analytical = robot._joint_jacobians(tfs)
            Jvs_autodiff = jax.jacfwd(
                lambda q_in: robot.joint_to_world_transforms(q_in)[:, :3, 3]
            )(q)
            Jws_autodiff = angular_jacobians_autodiff(robot.joint_to_world_transforms, q)
            return Jvs_analytical, Jws_analytical, Jvs_autodiff, Jws_autodiff

        Jvs_analytical, Jws_analytical, Jvs_autodiff, Jws_autodiff = compute(self.qs)
        for i in range(NUM_SAMPLES):
            with self.subTest(sample=i):
                np.testing.assert_allclose(
                    Jvs_analytical[i],
                    Jvs_autodiff[i],
                    atol=1e-7,
                    err_msg="Joint linear Jacobians mismatch",
                )
                np.testing.assert_allclose(
                    Jws_analytical[i],
                    Jws_autodiff[i],
                    atol=1e-7,
                    err_msg="Joint angular Jacobians mismatch",
                )


if __name__ == "__main__":
    unittest.main()
