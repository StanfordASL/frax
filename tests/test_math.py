"""Test cases for math/linalg utils, including inversion of the mass matrix

See timing/time_matrix_inverses.py for a speed comparison of the inversion methods
"""

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import pytest

from frax.robots.unitree_g1 import load_fixed_root_g1, load_g1
from frax.utils.linalg_utils import (
    fast_spd_inverse,
    random_spd_matrix,
    schur_spd_inverse,
)

NUM_SAMPLES = 100
TOL = 1e-10
MASS_MATRIX_TOL = 1e-5


def cholesky_inverse(M):
    L, low = jsp.linalg.cho_factor(M, lower=True)
    return jsp.linalg.cho_solve((L, low), jnp.eye(M.shape[0]))


INVERSES = {
    "standard": jnp.linalg.inv,
    "fast_spd": fast_spd_inverse,
    "cholesky": cholesky_inverse,
    "schur": None,  # Split index depends on the matrix, see below
}


def check_inversion_accuracy(mat, inv, tol):
    """Check that inv @ mat is the identity, to within tol, or for ill-conditioned matrices,
    to within the accuracy of the standard inverse (up to a small factor)"""
    n = mat.shape[0]
    assert inv.shape == mat.shape == (n, n)
    err = np.max(np.abs(inv @ mat - np.eye(n)))
    standard_err = np.max(np.abs(np.linalg.inv(mat) @ mat - np.eye(n)))
    assert err <= max(tol, 10 * standard_err), (
        f"Inverse error {err:.2e} is worse than the standard inverse error {standard_err:.2e}"
    )


@pytest.mark.parametrize("method", ["fast_spd", "schur"])
@pytest.mark.parametrize("n", [5, 10, 20, 30])
def test_spd_inverse(n, method):
    inverse = INVERSES[method] or partial(schur_spd_inverse, split_idx=n // 4)
    inverse = jax.jit(inverse)
    for _ in range(NUM_SAMPLES):
        M = random_spd_matrix(n)
        check_inversion_accuracy(M, inverse(M), TOL)


@pytest.fixture(
    scope="module",
    params=[load_fixed_root_g1, load_g1],
    ids=["g1_fixed_root", "g1_floating_root"],
)
def robot(request):
    return request.param()


@pytest.mark.parametrize("method", list(INVERSES))
def test_mass_matrix_inverse(robot, method):
    # For the floating base, the Schur complement splits off the 6 floating base DOFs
    inverse = INVERSES[method] or partial(schur_spd_inverse, split_idx=6)
    inverse = jax.jit(inverse)
    mass_matrix = jax.jit(robot.mass_matrix)
    for _ in range(10):
        q = np.random.uniform(-1.0, 1.0, robot.nq)
        if robot.is_quaternion_base:
            q[3:7] /= np.linalg.norm(q[3:7])
        M = mass_matrix(q)
        check_inversion_accuracy(M, inverse(M), MASS_MATRIX_TOL)
