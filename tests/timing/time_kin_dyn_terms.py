"""Timing script for checking performance regressions in kinematics and dynamics terms

Usage:
    python time_kin_dyn_terms.py                                   # CPU, single state
    python time_kin_dyn_terms.py --batch_size 1024 --n_calls 100   # CPU, batched
    python time_kin_dyn_terms.py --device gpu --batch_size 4096    # GPU, batched
"""

# ruff: noqa: E402 (jax must be imported after configure_env)

import argparse

from timing_utils import benchmark_function, configure_env, sample_state

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--n_calls", type=int, default=10000)
args = parser.parse_args()
configure_env(args.device)

import jax
import numpy as np
from frax import load_g1, load_panda


def make_functions(robot, ee_jacobian_and_derivative):
    """Functions to time, all with the signature f(q, qd)"""
    return {
        "FK": lambda q, qd: robot.joint_to_world_transforms(q),
        "EE J + Jdot": ee_jacobian_and_derivative,
        "Collision positions": lambda q, qd: robot.link_collision_positions(q),
        "Mass matrix": lambda q, qd: robot.mass_matrix(q),
        "Gravity vector": lambda q, qd: robot.gravity_vector(q),
        "Nonlinear bias": robot.nonlinear_bias,
        "Forward dynamics": lambda q, qd: robot.forward_dynamics(q, qd, qd, None),
    }


def main(batch_size: int | None, n_calls: int):
    np.random.seed(0)
    panda = load_panda()
    g1 = load_g1()
    robots = {
        "Panda": (panda, panda.ee_jacobian_and_derivative),
        "G1": (g1, g1.left_hand_jacobian_and_derivative),
    }
    n_states = 1 if batch_size is None else batch_size

    mode = "single state" if batch_size is None else f"batch size {batch_size}"
    print(f"\n=== {jax.default_backend().upper()}, {mode} ===")
    header = f"{'Robot':<7} {'Function':<20} {'JIT (s)':>9} {'Time (us)':>11} {'States/s':>12}"
    print(header)
    print("-" * len(header))
    for robot_name, (robot, ee_func) in robots.items():
        state = sample_state(robot, batch_size)
        for func_name, func in make_functions(robot, ee_func).items():
            if batch_size is not None:
                func = jax.vmap(func)
            avg_time, jit_time = benchmark_function(func, state, n_calls)
            print(
                f"{robot_name:<7} {func_name:<20} {jit_time:>9.3f} "
                f"{avg_time * 1e6:>11.2f} {n_states / avg_time:>12.0f}"
            )


if __name__ == "__main__":
    main(args.batch_size, args.n_calls)
