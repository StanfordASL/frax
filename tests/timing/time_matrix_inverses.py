"""Timing script comparing SPD matrix inversion methods

Covers random SPD matrices of various sizes, as well as the G1 mass matrix
(fixed root, and floating root where the Schur complement method applies)
"""

# ruff: noqa: E402 (jax must be imported after configure_env)

import argparse

from timing_utils import benchmark_function, configure_env, sample_state

parser = argparse.ArgumentParser()
parser.add_argument("--n_calls", type=int, default=10000)
args = parser.parse_args()
configure_env("cpu")

import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from frax.robots.unitree_g1 import load_fixed_root_g1, load_g1
from frax.utils.linalg_utils import (
    fast_spd_inverse,
    random_spd_matrix,
    schur_spd_inverse,
)


def cholesky_inverse(M):
    L, low = jsp.linalg.cho_factor(M, lower=True)
    return jsp.linalg.cho_solve((L, low), jnp.eye(M.shape[0]))


def main(n_calls: int):
    np.random.seed(0)
    g1_fixed = load_fixed_root_g1()
    g1_floating = load_g1()

    # (name, matrix, Schur split index (None to skip the Schur method))
    cases = [(f"Random {n}x{n}", random_spd_matrix(n), n // 4) for n in (5, 10, 20, 30)]
    cases += [
        ("G1 fixed root", g1_fixed.mass_matrix(sample_state(g1_fixed)[0]), None),
        ("G1 floating root", g1_floating.mass_matrix(sample_state(g1_floating)[0]), 6),
    ]

    methods = ["Standard", "Fast SPD", "Cholesky", "Schur"]
    header = f"{'Matrix':<18}" + "".join(f"{m + ' (us)':>16}" for m in methods)
    print(header)
    print("-" * len(header))
    for name, M, split_idx in cases:
        funcs = {
            "Standard": jnp.linalg.inv,
            "Fast SPD": fast_spd_inverse,
            "Cholesky": cholesky_inverse,
            "Schur": None
            if split_idx is None
            else lambda M, k=split_idx: schur_spd_inverse(M, split_idx=k),
        }
        row = f"{name:<18}"
        for method in methods:
            if funcs[method] is None:
                row += f"{'-':>16}"
                continue
            avg_time, _ = benchmark_function(funcs[method], (M,), n_calls)
            row += f"{avg_time * 1e6:>16.2f}"
        print(row)


if __name__ == "__main__":
    main(args.n_calls)
