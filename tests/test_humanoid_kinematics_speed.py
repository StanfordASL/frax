"""Test cases for humanoid kinematics"""

import time
from typing import Tuple, Callable

import jax
import numpy as np

from frax.robots.unitree_g1 import load_g1

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)


def test_speed(func: Callable, inputs: Tuple[Tuple]) -> Tuple[float, float]:
    jit_func = jax.jit(func)
    n = len(inputs)

    # Warm up jit compilation
    jit_start_time = time.perf_counter()
    _ = jit_func(inputs[0])
    jit_duration = time.perf_counter() - jit_start_time

    # Eval speed on repeated calls
    start_time = time.perf_counter()
    for i in range(1, n):
        _ = jit_func(inputs[i]).block_until_ready()
    duration = time.perf_counter() - start_time
    avg_duration = duration / (n - 1)
    return jit_duration, avg_duration


def sample_qs(num_joints, n: int):
    return np.random.uniform(-np.pi / 2, np.pi / 2, (n, num_joints))


def main():
    robot = load_g1()
    n_evals = 10000
    inputs = sample_qs(robot.nq, n_evals)

    print("Testing forward kinematics")
    jit_dur, avg_dur = test_speed(robot.joint_to_world_transforms, inputs)
    print("Jit duration (seconds): ", jit_dur)
    print("Average duration (milliseconds): ", avg_dur * 1e3)

    print("Testing right hand jacobian")
    jit_dur, avg_dur = test_speed(robot.right_hand_jacobian, inputs)
    print("Jit duration (seconds): ", jit_dur)
    print("Average duration (milliseconds): ", avg_dur * 1e3)

    print("Testing collision data")
    jit_dur, avg_dur = test_speed(robot.link_collision_positions, inputs)
    print("Jit duration (seconds): ", jit_dur)
    print("Average duration (milliseconds): ", avg_dur * 1e3)


if __name__ == "__main__":
    main()
