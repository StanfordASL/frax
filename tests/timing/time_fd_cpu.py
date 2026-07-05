"""Timing script for checking performance regressions

This script: forward dynamics on CPU
"""

import os

os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["JAX_ENABLE_X64"] = "True"
os.environ["JAX_PLATFORMS"] = "cpu"


import jax
import numpy as np
from frax import load_g1, load_panda
from timing_utils import benchmark_function


def main():
    panda = load_panda()
    g1 = load_g1()

    @jax.jit
    def panda_fd(q, qd, tau):
        fext = None
        return panda.forward_dynamics(q, qd, tau, fext)

    @jax.jit
    def g1_fd(q, qd, tau):
        fext = None
        return g1.forward_dynamics(q, qd, tau, fext)

    # Initial state and dummy control input
    np.random.seed(0)
    q_panda = np.random.uniform(-0.1, 0.1, panda.num_joints)
    qd_panda = np.zeros(panda.num_joints)
    tau_panda = np.random.uniform(-0.1, 0.1, panda.num_joints)
    panda_args = (q_panda, qd_panda, tau_panda)

    avg_time, jit_time = benchmark_function(panda_fd, panda_args)

    print("--- PANDA ---")
    print("JIT time: ", jit_time)
    print("Steps per second: ", 1 / avg_time)

    # Initial state and dummy control input
    q_g1 = np.random.uniform(-0.1, 0.1, g1.num_joints)
    qd_g1 = np.zeros(g1.num_joints)
    tau_g1 = np.random.uniform(-0.1, 0.1, g1.num_joints)
    g1_args = (q_g1, qd_g1, tau_g1)

    avg_time, jit_time = benchmark_function(g1_fd, g1_args)

    print("--- G1 ---")
    print("JIT time: ", jit_time)
    print("Steps per second: ", 1 / avg_time)


if __name__ == "__main__":
    main()
