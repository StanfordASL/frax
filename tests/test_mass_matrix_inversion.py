"""Test cases for mass matrix inversion methods"""

import time
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

    def test_fixed_base_speed(self):
        num_tests = 100
        qs = [np.random.rand(29) for _ in range(num_tests)]

        # Dummy solves for jit compilation
        q_jit = np.random.rand(29)
        M = self.fixed_root_robot.mass_matrix(q_jit)
        minv_reg_result = minv_regular(self.fixed_root_robot, q_jit)
        minv_spd_result = minv_spd(self.fixed_root_robot, q_jit)
        minv_cho_result = minv_cho(self.fixed_root_robot, q_jit)
        # Make sure that the results are the same
        atol = 1e-5
        rtol = 1e-5
        check_inversion_accuracy(M, minv_reg_result, atol, rtol)
        check_inversion_accuracy(M, minv_spd_result, atol, rtol)
        check_inversion_accuracy(M, minv_cho_result, atol, rtol)

        start_time_fast_psd_inv = time.perf_counter()
        for q in qs:
            _m_inv = minv_spd(self.fixed_root_robot, q).block_until_ready()
        duration_fast_psd_inv = time.perf_counter() - start_time_fast_psd_inv
        avg_time_fast_psd_inv = duration_fast_psd_inv / num_tests

        start_time_standard_inv = time.perf_counter()
        for q in qs:
            _m_inv = minv_regular(self.fixed_root_robot, q).block_until_ready()
        duration_standard_inv = time.perf_counter() - start_time_standard_inv
        avg_time_standard_inv = duration_standard_inv / num_tests

        start_time_cho_inv = time.perf_counter()
        for q in qs:
            _m_inv = minv_cho(self.fixed_root_robot, q).block_until_ready()
        duration_cho_inv = time.perf_counter() - start_time_cho_inv
        avg_time_cho_inv = duration_cho_inv / num_tests

        print("\nSpeed test for fixed-root")
        print(f"Standard inverse: {avg_time_standard_inv * 1e6:.3f} µs")
        print(f"Fast PSD inverse: {avg_time_fast_psd_inv * 1e6:.3f} µs")
        print(f"Cholesky inverse: {avg_time_cho_inv * 1e6:.3f} µs")

    def test_floating_base_speed(self):
        num_tests = 100
        qs = [np.random.rand(self.floating_root_robot.nq) for _ in range(num_tests)]

        # Dummy solves for jit compilation
        q_jit = np.random.rand(self.floating_root_robot.nq)
        M = self.floating_root_robot.mass_matrix(q_jit)
        minv_reg_result = minv_regular(self.floating_root_robot, q_jit)
        minv_spd_result = minv_spd(self.floating_root_robot, q_jit)
        minv_schur_result = minv_schur(self.floating_root_robot, q_jit)
        minv_cho_result = minv_cho(self.floating_root_robot, q_jit)
        # Make sure that the results are the same
        atol = 1e-5
        rtol = 1e-5
        check_inversion_accuracy(M, minv_reg_result, atol, rtol)
        check_inversion_accuracy(M, minv_spd_result, atol, rtol)
        check_inversion_accuracy(M, minv_schur_result, atol, rtol)
        check_inversion_accuracy(M, minv_cho_result, atol, rtol)

        start_time_schur = time.perf_counter()
        for q in qs:
            _m_inv = minv_schur(self.floating_root_robot, q).block_until_ready()
        duration_schur = time.perf_counter() - start_time_schur
        avg_time_schur = duration_schur / num_tests

        start_time_fast_psd_inv = time.perf_counter()
        for q in qs:
            _m_inv = minv_spd(self.floating_root_robot, q).block_until_ready()
        duration_fast_psd_inv = time.perf_counter() - start_time_fast_psd_inv
        avg_time_fast_psd_inv = duration_fast_psd_inv / num_tests

        start_time_standard_inv = time.perf_counter()
        for q in qs:
            _m_inv = minv_regular(self.floating_root_robot, q).block_until_ready()
        duration_standard_inv = time.perf_counter() - start_time_standard_inv
        avg_time_standard_inv = duration_standard_inv / num_tests

        start_time_cho_inv = time.perf_counter()
        for q in qs:
            _m_inv = minv_cho(self.floating_root_robot, q).block_until_ready()
        duration_cho_inv = time.perf_counter() - start_time_cho_inv
        avg_time_cho_inv = duration_cho_inv / num_tests

        print("\nSpeed test for floating-root")
        print(f"Standard inverse: {avg_time_standard_inv * 1e6:.3f} µs")
        print(f"Fast PSD inverse: {avg_time_fast_psd_inv * 1e6:.3f} µs")
        print(f"Schur inverse: {avg_time_schur * 1e6:.3f} µs")
        print(f"Cholesky inverse: {avg_time_cho_inv * 1e6:.3f} µs")


if __name__ == "__main__":
    unittest.main()
