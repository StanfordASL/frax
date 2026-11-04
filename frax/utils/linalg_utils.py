"""Linalg utils"""

import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax import Array


def fast_spd_inverse(A: Array) -> Array:
    """Inverse of a symmetric positive definite matrix via scipy.linalg.solve

    Args:
        A (Array): SPD matrix to compute the inverse of, shape (n, n)

    Returns:
        Array: Matrix inverse, shape (n, n)
    """
    return jsp.linalg.solve(A, jnp.eye(A.shape[0]), assume_a="pos")


def cholesky_spd_inverse(A: Array) -> Array:
    """Inverse of a symmetric positive definite matrix via cholesky factorization

    Args:
        A (Array): SPD matrix to compute the inverse of, shape (n, n)

    Returns:
        Array: Matrix inverse, shape (n, n)
    """
    L, low = jsp.linalg.cho_factor(A, lower=True)
    return jsp.linalg.cho_solve((L, low), jnp.eye(A.shape[0]))


def schur_spd_inverse(M: Array, split_idx: int) -> Array:
    """Invert a symmetric positive definite matrix using the Schur complement.

    This assumes that the matrix can be split into blocks where we have
    M = [[A, B], [B.T, D]]

    Args:
        M (Array): SPD matrix to invert, shape (n, n)
        split_idx (int): The index at which to split the matrix into blocks (A, B, D).
            For a robot with a free-floating 6-dof base, this is 6 when considering
            the mass matrix structure.

    Returns:
        Array: Matrix inverse, shape (n, n)
    """
    A = M[:split_idx, :split_idx]
    B = M[:split_idx, split_idx:]
    D = M[split_idx:, split_idx:]

    # Invert A with cholesky (SPD)
    # L_A, low_A = jsp.linalg.cho_factor(A, lower=True)
    # A_inv = jsp.linalg.cho_solve((L_A, low_A), jnp.eye(split_idx))
    A_inv = cholesky_spd_inverse(A)

    # Schur complement
    S = D - B.T @ A_inv @ B

    # Invert schur complement with cholesky(SPD)
    # L_S, low_S = jsp.linalg.cho_factor(S, lower=True)
    # S_inv = jsp.linalg.cho_solve((L_S, low_S), jnp.eye(M.shape[0] - split_idx))
    S_inv = cholesky_spd_inverse(S)

    # Block matrix inversion
    X = A_inv @ B
    XS_inv = X @ S_inv
    M_inv_top_left = A_inv + XS_inv @ B.T @ A_inv
    M_inv_top_right = -XS_inv
    M_inv_bottom_left = M_inv_top_right.T
    M_inv_bottom_right = S_inv

    return jnp.block(
        [[M_inv_top_left, M_inv_top_right], [M_inv_bottom_left, M_inv_bottom_right]]
    )


def random_spd_matrix(n: int, eps: float = 1e-10) -> np.ndarray:
    """Generate a random symmetric positive definite matrix

    Args:
        n (int): Size of the square matrix
        eps (float, optional): Regularization tolerance. Defaults to 1e-10.

    Returns:
        np.ndarray: A random SPD matrix of shape (n, n)
    """
    A = np.random.normal(size=(n, n))
    return A.T @ A + eps * np.eye(n)


def skew(v: Array) -> Array:
    """Skew-symmetric matrix form of a vector in R3

    Args:
        v (Array): Vector to convert, shape (3,)

    Returns:
        Array: (3, 3) skew-symmetric matrix
    """
    assert v.shape == (3,)
    return jnp.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def skew_numpy(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix form of a vector in R3

    Args:
        v (np.ndarray): Vector to convert, shape (3,)

    Returns:
        np.ndarray: (3, 3) skew-symmetric matrix
    """
    assert v.shape == (3,)
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
