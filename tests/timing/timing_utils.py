"""Timing utilities

NOTE: call `configure_env` before importing jax (or anything that imports jax), since some
of these settings are only read when jax is first loaded
"""

import os
import time
from importlib.metadata import version as package_version
from typing import Callable, Tuple

from packaging import version


def configure_env(device: str = "cpu") -> None:
    """Set the recommended environment variables for benchmarking frax on CPU or GPU

    Args:
        device (str, optional): "cpu" or "gpu". Defaults to "cpu".
    """
    assert device in ("cpu", "gpu")
    os.environ["JAX_ENABLE_X64"] = "True"
    if device == "gpu":
        os.environ["JAX_PLATFORMS"] = "cuda"
        return
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
    if version.parse(package_version("jax")) >= version.parse("0.9.1"):
        os.environ["XLA_FLAGS"] += (
            " --xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED"
        )


def benchmark_function(
    func: Callable, args: Tuple, n_calls: int = 100000, n_warmup: int = 10
) -> Tuple[float, float]:
    """Benchmark the timing performance of a JAX function

    Args:
        func (Callable): JAX function to test
        args (Tuple): Example arguments for the function
        n_calls (int, optional): Number of calls to use for timing. Defaults to 100000.
        n_warmup (int, optional): Number of 'warmup' calls before timing. Defaults to 10.

    Returns:
        Tuple[float, float]:
            avg_time (float): Average time per function call after JIT
            jit_time (float): Time taken to JIT the function
    """
    import jax  # Imported here so that configure_env can run first

    func_jit = jax.jit(func)

    # JIT timing
    start_time = time.perf_counter()
    res = func_jit(*args)
    jax.block_until_ready(res)
    jit_time = time.perf_counter() - start_time

    # Warmup
    for _ in range(n_warmup):
        res = func_jit(*args)
        jax.block_until_ready(res)

    # Timing
    start = time.perf_counter()
    for _ in range(n_calls):
        res = func_jit(*args)
        jax.block_until_ready(res)
    elapsed = time.perf_counter() - start
    avg_time = elapsed / n_calls

    return avg_time, jit_time


def sample_state(robot, batch_size: int | None = None) -> Tuple:
    """Sample a random (q, qd), with a leading batch dimension if batch_size is provided"""
    import numpy as np  # Imported here so that configure_env can run first

    shape = () if batch_size is None else (batch_size,)
    q = np.random.uniform(-0.5, 0.5, shape + (robot.nq,))
    if robot.is_quaternion_base:
        quat = np.random.randn(*shape, 4)
        q[..., 3:7] = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    qd = np.random.uniform(-0.5, 0.5, shape + (robot.nv,))
    return q, qd
