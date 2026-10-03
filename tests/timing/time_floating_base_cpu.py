"""Timing script comparing the floating base representations (euler vs quaternion)

This script: G1 kinematics and dynamics on CPU, single-state and batched
"""

import os
from importlib.metadata import version as package_version

from packaging import version

os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
if version.parse(package_version("jax")) >= version.parse("0.9.1"):
    os.environ["XLA_FLAGS"] += (
        " --xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED"
    )
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["JAX_ENABLE_X64"] = "True"
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse

import jax
import numpy as np
from frax import load_g1
from timing_utils import benchmark_function


def make_functions(robot):
    def fk(q, qd):
        return robot.joint_to_world_transforms(q)

    def mass_matrix(q, qd):
        return robot.mass_matrix(q)

    def bias(q, qd):
        return robot.nonlinear_bias(q, qd)

    def forward_dynamics(q, qd):
        return robot.forward_dynamics(q, qd, qd, None)

    def hand_J_Jdot(q, qd):
        return robot.left_hand_jacobian_and_derivative(q, qd)

    def osc_terms(q, qd):
        # A typical whole-body controller's per-step dynamics terms
        tfs = robot.joint_to_world_transforms(q)
        M = robot._mass_matrix(tfs)
        M_inv = robot.mass_matrix_inverse(M)
        b = robot._nonlinear_bias(qd, tfs)
        J_lh, Jd_lh = robot._left_hand_jacobian_and_derivative(qd, tfs)
        J_rh, Jd_rh = robot._right_hand_jacobian_and_derivative(qd, tfs)
        J_com = robot._center_of_mass_jacobian(tfs)
        return M, M_inv, b, J_lh, Jd_lh, J_rh, Jd_rh, J_com

    return {
        "FK": fk,
        "Mass matrix": mass_matrix,
        "Nonlinear bias": bias,
        "Forward dynamics": forward_dynamics,
        "Hand J + Jdot": hand_J_Jdot,
        "OSC terms": osc_terms,
    }


def sample_state(robot, batch_size=None):
    shape = () if batch_size is None else (batch_size,)
    q = np.random.uniform(-0.5, 0.5, shape + (robot.nq,))
    if robot.is_quaternion_base:
        quat = np.random.randn(*shape, 4)
        q[..., 3:7] = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    qd = np.random.uniform(-0.5, 0.5, shape + (robot.nv,))
    return q, qd


def main(n_calls: int, n_trials: int, batch_size: int):
    np.random.seed(0)
    robots = {
        "euler": load_g1(floating_base="euler"),
        "quaternion": load_g1(floating_base="quaternion"),
    }
    funcs = {name: make_functions(robot) for name, robot in robots.items()}

    for batched in (False, True):
        if batched:
            print(f"\n=== Batched (vmap, batch size {batch_size}): time per batch ===")
        else:
            print("\n=== Single state: time per call ===")
        header = f"{'Function':<18} {'euler (us)':>12} {'quat (us)':>12} {'quat/euler':>11}"
        print(header)
        print("-" * len(header))
        for func_name in funcs["euler"]:
            times = {}
            for rep, robot in robots.items():
                f = funcs[rep][func_name]
                if batched:
                    f = jax.vmap(f)
                args = sample_state(robot, batch_size if batched else None)
                # Take the best of several trials to reduce noise from the OS
                n = n_calls // 50 if batched else n_calls
                trials = [benchmark_function(f, args, n)[0] for _ in range(n_trials)]
                times[rep] = min(trials) * 1e6
            ratio = times["quaternion"] / times["euler"]
            print(
                f"{func_name:<18} {times['euler']:>12.2f} {times['quaternion']:>12.2f} {ratio:>11.3f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_calls", type=int, default=20000)
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()
    main(args.n_calls, args.n_trials, args.batch_size)
