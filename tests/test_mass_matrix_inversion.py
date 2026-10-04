"""Test cases for mass matrix inversion methods

See timing/time_matrix_inverses.py for a speed comparison of these methods
"""

import unittest
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from frax.robots.unitree_g1 import load_fixed_root_g1, load_g1
from frax.utils.linalg_utils import (
    fast_spd_inverse,
    schur_spd_inverse,
)

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)


@partial(jax.jit, static_argnums=(0,))
def minv_regular(robot, q):
    M = robot.mass_matrix(q)
    return jnp.linalg.inv(M)


@partial(jax.jit, static_argnums=(0,))
def minv_spd(robot, q):
    M = robot.mass_matrix(q)
    return fast_spd_inverse(M)


@partial(jax.jit, static_argnums=(0,))
def minv_schur(robot, q):
    M = robot.mass_matrix(q)
    return schur_spd_inverse(M, split_idx=6)


@partial(jax.jit, static_argnums=(0,))
def minv_cho(robot, q):
    M = robot.mass_matrix(q)
    L, low = jsp.linalg.cho_factor(M, lower=True)
    return jsp.linalg.cho_solve((L, low), jnp.eye(M.shape[0]))


def check_inversion_accuracy(mat, inv, atol, rtol):
    n = mat.shape[0]
    assert inv.shape == mat.shape == (n, n)
    try:
        np.testing.assert_allclose(inv @ mat, np.eye(n), atol=atol, rtol=rtol)
    except AssertionError:
        # If my identity check did not pass then make sure that the standard inverse is also
        # having some numerical difficulty on this matrix
        try:
            # If this assertion PASSES then the standard inverse performs better on this edge case
            # and thus we have introduced a problem with our custom method
            np.testing.assert_allclose(
                np.linalg.inv(mat) @ mat,
                np.eye(n),
                atol=atol,
                rtol=rtol,
            )
            raise  # The previous error
        except AssertionError:
            # If this assertion FAILS then our inverse performs the same as the standard (this is ok)
            pass


@jax.tree_util.register_static
class TestMInv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.floating_root_robot = load_g1()
        cls.fixed_root_robot = load_fixed_root_g1()
        np.random.seed(0)

    def _check_methods(self, robot, minv_funcs):
        atol = 1e-5
        rtol = 1e-5
        for _ in range(10):
            q = np.random.rand(robot.nq)
            M = robot.mass_matrix(q)
            for minv in minv_funcs:
                check_inversion_accuracy(M, minv(robot, q), atol, rtol)

    def test_fixed_base(self):
        self._check_methods(self.fixed_root_robot, [minv_regular, minv_spd, minv_cho])

    def test_floating_base(self):
        self._check_methods(
            self.floating_root_robot, [minv_regular, minv_spd, minv_schur, minv_cho]
        )


if __name__ == "__main__":
    unittest.main()
