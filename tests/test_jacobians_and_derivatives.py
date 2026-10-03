"""Test cases for analytical jacobians and derivatives, compared against autodiff"""

# TODO this is currently super slow -- should be smarter about jitting here

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from frax.robots.unitree_g1 import load_g1

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)


class TestJacobiansAndDerivatives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = load_g1(floating_base="euler")
        cls.nq = cls.robot.nq
        cls.nv = cls.robot.nv
        np.random.seed(42)

    def _get_random_state(self):
        q = np.random.uniform(-1.0, 1.0, self.nq)
        qd = np.random.uniform(-1.0, 1.0, self.nv)
        return q, qd

    def _compute_angular_jacobian_autodiff(self, q, tf_func):
        """Computes the angular Jacobian Jw such that w = Jw @ qd,
        using the relationship skew(w) = R_dot @ R^T
        """
        R = tf_func(q)[:3, :3]
        # dR/dq has shape (3, 3, nq)
        dR_dq = jax.jacobian(lambda q_in: tf_func(q_in)[:3, :3])(q)

        # Jw_cols[i] = unskew( (dR/dq_i) @ R.T )
        # res has shape (3, 3, nq)
        res = jnp.einsum("ijk,lj->ilk", dR_dq, R)

        # Extract unskewed for each column
        Jw = jnp.stack([res[2, 1, :], res[0, 2, :], res[1, 0, :]], axis=0)
        return Jw

    def test_center_of_mass_jacobian(self):
        """Test COM Jacobian against autodiff of COM position"""
        for i in range(5):
            q, _ = self._get_random_state()

            # Analytical
            J_analytical = self.robot.center_of_mass_jacobian(q)

            # Autodiff
            J_autodiff = jax.jacobian(self.robot.center_of_mass)(q)

            with self.subTest(sample=i):
                np.testing.assert_allclose(
                    J_analytical, J_autodiff, atol=1e-7, err_msg="COM Jacobian mismatch"
                )

    def test_ee_jacobians(self):
        """Test EE Jacobians (linear & angular) against autodiff"""
        # fmt: off
        ee_funcs = [
            (self.robot.left_hand_jacobian, self.robot.left_hand_transform, "left_hand"),
            (self.robot.right_hand_jacobian, self.robot.right_hand_transform, "right_hand"),
            (self.robot.left_foot_jacobian, self.robot.left_foot_transform, "left_foot"),
            (self.robot.right_foot_jacobian, self.robot.right_foot_transform, "right_foot"),
        ]
        # fmt: on

        for jac_func, tf_func, name in ee_funcs:
            for i in range(5):
                q, _ = self._get_random_state()

                # Analytical
                J_analytical = jac_func(q)

                # Autodiff Linear
                def pos_func(q_in):
                    return tf_func(q_in)[:3, 3]

                Jv_autodiff = jax.jacobian(pos_func)(q)

                # Autodiff Angular
                Jw_autodiff = self._compute_angular_jacobian_autodiff(q, tf_func)

                J_autodiff = jnp.vstack([Jv_autodiff, Jw_autodiff])

                with self.subTest(ee=name, sample=i):
                    np.testing.assert_allclose(
                        J_analytical,
                        J_autodiff,
                        atol=1e-7,
                        err_msg=f"{name} Jacobian mismatch",
                    )

    def test_ee_jacobian_derivatives(self):
        """Test EE Jacobian derivatives (linear & angular) against JVP of Jacobian"""
        # fmt: off
        ee_funcs = [
            (self.robot.left_hand_jacobian_and_derivative, self.robot.left_hand_jacobian, "left_hand"),
            (self.robot.right_hand_jacobian_and_derivative, self.robot.right_hand_jacobian, "right_hand"),
            (self.robot.left_foot_jacobian_and_derivative, self.robot.left_foot_jacobian, "left_foot"),
            (self.robot.right_foot_jacobian_and_derivative, self.robot.right_foot_jacobian, "right_foot"),
        ]
        # fmt: on

        for jac_dot_func, jac_func, name in ee_funcs:
            for i in range(5):
                q, qd = self._get_random_state()

                # Analytical Jdot
                _, Jdot_analytical = jac_dot_func(q, qd)

                # Autodiff Jdot via JVP of analytical Jacobian function
                _, Jdot_autodiff = jax.jvp(jac_func, (q,), (qd,))

                with self.subTest(ee=name, sample=i):
                    np.testing.assert_allclose(
                        Jdot_analytical,
                        Jdot_autodiff,
                        atol=1e-6,
                        err_msg=f"{name} Jdot mismatch",
                    )

    def test_link_jacobians(self):
        """Test internal _link_linear_jacobians and _link_angular_jacobians against autodiff"""
        for i in range(5):
            q, _ = self._get_random_state()

            # Analytical
            tfs = self.robot.joint_to_world_transforms(q)
            link_Jvs_analytical = self.robot._link_linear_jacobians(tfs)
            link_Jws_analytical = self.robot._link_angular_jacobians(tfs)

            # Autodiff Linear
            def link_coms_func(q_in):
                return self.robot.link_com_positions(q_in)

            link_Jvs_autodiff = jax.jacobian(link_coms_func)(q)

            # Autodiff Angular
            def link_tf_func(link_idx):
                def tf_func(q_in):
                    return self.robot.link_to_world_transforms(q_in)[link_idx]

                return tf_func

            # Compare linear components
            np.testing.assert_allclose(
                link_Jvs_analytical,
                link_Jvs_autodiff,
                atol=1e-7,
                err_msg="Link linear Jacobians mismatch",
            )

            # Compare angular components for each link
            for link_idx in range(
                self.robot.nv
            ):  # number of links usually equals num_joints for Humanoid
                Jw_auto = self._compute_angular_jacobian_autodiff(
                    q, link_tf_func(link_idx)
                )
                np.testing.assert_allclose(
                    link_Jws_analytical[link_idx],
                    Jw_auto,
                    atol=1e-7,
                    err_msg=f"Link {link_idx} angular Jacobian mismatch",
                )

    def test_joint_jacobians(self):
        """Test internal _joint_jacobians (linear & angular) against autodiff"""
        for i in range(5):
            q, _ = self._get_random_state()

            # Analytical
            tfs = self.robot.joint_to_world_transforms(q)
            Jv_joints_analytical, Jw_joints_analytical = self.robot._joint_jacobians(
                tfs
            )

            # Autodiff Linear
            def joint_pos_func(q_in):
                return self.robot.joint_to_world_transforms(q_in)[:, :3, 3]

            Jv_joints_autodiff = jax.jacobian(joint_pos_func)(q)

            # Compare linear
            np.testing.assert_allclose(
                Jv_joints_analytical,
                Jv_joints_autodiff,
                atol=1e-7,
                err_msg="Joint linear Jacobians mismatch",
            )

            # Compare angular
            def joint_tf_func(joint_idx):
                def tf_func(q_in):
                    return self.robot.joint_to_world_transforms(q_in)[joint_idx]

                return tf_func

            for joint_idx in range(self.robot.nv):
                Jw_auto = self._compute_angular_jacobian_autodiff(
                    q, joint_tf_func(joint_idx)
                )
                np.testing.assert_allclose(
                    Jw_joints_analytical[joint_idx],
                    Jw_auto,
                    atol=1e-7,
                    err_msg=f"Joint {joint_idx} angular Jacobian mismatch",
                )


if __name__ == "__main__":
    unittest.main()
