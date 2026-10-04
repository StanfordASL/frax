"""Test cases for math/linalg utils"""

import unittest

import jax
import numpy as np

from frax.utils.linalg_utils import (
    fast_spd_inverse,
    random_spd_matrix,
    schur_spd_inverse,
)

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)


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


class TestLinalg(unittest.TestCase):
    DIMS = [5, 10, 20, 30]
    NUM_ACCURACY_TESTS = 100
    RTOL = 1e-10
    ATOL = 1e-10

    def _test_spd_inv_accuracy(self, n):
        matrices = [random_spd_matrix(n) for _ in range(self.NUM_ACCURACY_TESTS)]

        @jax.jit
        def jit_fast_spd_inv(mat):
            return fast_spd_inverse(mat)

        custom_invs = [jit_fast_spd_inv(m) for m in matrices]

        for mat, inv in zip(matrices, custom_invs):
            check_inversion_accuracy(mat, inv, self.ATOL, self.RTOL)

    def test_schur_inv_accuracy(self):
        for n in self.DIMS:
            self._test_schur_inv_accuracy(n)

    def _test_schur_inv_accuracy(self, n):
        matrices = [random_spd_matrix(n) for _ in range(self.NUM_ACCURACY_TESTS)]

        @jax.jit
        def jit_schur_inv(mat):
            return schur_spd_inverse(mat, split_idx=mat.shape[0] // 4)

        custom_invs = [jit_schur_inv(m) for m in matrices]

        for mat, inv in zip(matrices, custom_invs):
            check_inversion_accuracy(mat, inv, self.ATOL, self.RTOL)

    def test_psd_inv_accuracy(self):
        for n in self.DIMS:
            self._test_spd_inv_accuracy(n)


if __name__ == "__main__":
    unittest.main()
