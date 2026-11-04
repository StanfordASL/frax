"""Example: Multi-sample SQP global IK with the Franka Panda,
with each sample parallelized across CPU cores (multiple JAX CPU devices)

Note: frax.ik performs best on differential IK problems, but this example
shows that you can use it for more global problems. I recommend *not* using
this method in an online setting since it is a bit slow, but this can be
very handy for cases where you perform one expensive call at the start, and
then fast online diff IK calls from there on.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false "
    "--xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED"
)
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import jax

# One device per core
jax.config.update("jax_num_cpu_devices", os.cpu_count())

import time

import numpy as np

from frax import load_panda
from frax.ik import (
    DiffIKSolver,
    EETask,
    GlobalIKSolver,
    JointLimits,
    JointTask,
)

NUM_SAMPLES = 32
NUM_PROBLEMS = 20


def build_solver(robot):
    tasks = [
        EETask(robot, gain_pos=10.0, gain_rot=10.0),
        JointTask(robot, np.arange(7), "q_home", gain=1.0, weight=0.01),
    ]
    constraints = [JointLimits(robot)]
    return DiffIKSolver(robot, tasks, constraints)


def main():
    robot = load_panda()
    solver = build_solver(robot)
    q_min, q_max = robot.actuated_joint_lower_limits, robot.actuated_joint_upper_limits
    q_home = np.asarray(robot.default_configuration)

    rng = np.random.default_rng(0)
    targets = jax.vmap(robot.ee_transform)(rng.uniform(q_min, q_max, (NUM_PROBLEMS, 7)))
    refs = [
        {"ee_pos": target[:3, 3], "ee_rmat": target[:3, :3], "q_home": q_home}
        for target in targets
    ]
    keys = jax.random.split(jax.random.key(0), NUM_PROBLEMS)
    print(f"{NUM_SAMPLES} starts per problem, {len(jax.devices())} devices")

    # "default": every sample on JAX's default device alone
    for name, devs in (("One device", "default"), ("All devices", "all")):
        # The sampling range defaults to the solver's JointLimits
        global_ik = GlobalIKSolver(solver, num_samples=NUM_SAMPLES, devices=devs)
        solve = jax.jit(global_ik.solve)
        jax.block_until_ready(solve(q_home, refs[0], keys[0]))  # Compile
        times, errors = [], []
        for ref, key in zip(refs, keys, strict=True):
            start = time.perf_counter()
            q = jax.block_until_ready(solve(q_home, ref, key).q)
            times.append(time.perf_counter() - start)
            errors.append(np.linalg.norm(robot.ee_transform(q)[:3, 3] - ref["ee_pos"]))
        print(
            f"{name}: {1e3 * np.median(times):.1f} ms per problem, "
            + f"{np.mean(np.array(errors) < 5e-3):.0%} within 5 mm"
        )


if __name__ == "__main__":
    main()
