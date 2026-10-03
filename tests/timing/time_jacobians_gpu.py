"""Timing script for checking performance regressions

This script: Jacobians and derivatives on GPU
"""

import os

os.environ["JAX_PLATFORMS"] = "cuda"


import jax
import numpy as np
from frax import load_g1, load_panda
from timing_utils import benchmark_function


def main():
    panda = load_panda()
    g1 = load_g1()

    @jax.jit
    def batched_panda_j_jdot(qs, qds):
        return jax.vmap(panda.ee_jacobian_and_derivative)(qs, qds)

    @jax.jit
    def batched_g1_j_jdot(qs, qds):
        return jax.vmap(g1.left_hand_jacobian_and_derivative)(qs, qds)

    # Initial state and dummy control input
    batch_size = 4096
    np.random.seed(0)
    q_panda = np.random.uniform(-0.1, 0.1, (batch_size, panda.nv))
    qd_panda = np.random.uniform(-0.1, 0.1, (batch_size, panda.nv))
    panda_args = (q_panda, qd_panda)

    avg_time, jit_time = benchmark_function(
        batched_panda_j_jdot, panda_args, n_calls=10000
    )

    print("--- PANDA ---")
    print("JIT time: ", jit_time)
    print("Steps per second: ", batch_size / avg_time)

    # Initial state and dummy control input
    q_g1 = np.random.uniform(-0.1, 0.1, (batch_size, g1.nq))
    qd_g1 = np.random.uniform(-0.1, 0.1, (batch_size, g1.nv))
    g1_args = (q_g1, qd_g1)

    avg_time, jit_time = benchmark_function(batched_g1_j_jdot, g1_args, n_calls=10000)

    print("--- G1 ---")
    print("JIT time: ", jit_time)
    print("Steps per second: ", batch_size / avg_time)


if __name__ == "__main__":
    main()
