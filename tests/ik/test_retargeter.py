"""Online and offline modes run the same step, so they must agree"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from toy_robot import ToyRobot

from frax.ik import (
    DiffIKSolver,
    GlobalIKSolver,
    JointLimits,
    JointTask,
    OfflineRetargeter,
    OnlineRetargeter,
    ResidualTask,
    Retargeter,
)

DT = 0.02


def _assert_online_matches_offline(retargeter, frames):
    q, infos, _ = retargeter.retarget_clip(frames, DT)
    online = OnlineRetargeter(retargeter)
    online_q = np.stack([online.step(frame, DT) for frame in np.asarray(frames)])
    np.testing.assert_allclose(q, online_q, atol=1e-9)
    assert np.all(infos.qp_converged)
    return q


def _joint_retargeter(n: int, preprocess=lambda frame: {"q": frame}) -> Retargeter:
    """Tracks joint targets, from zero, with velocity limits"""
    robot = ToyRobot(n)
    solver = DiffIKSolver(
        robot,
        [JointTask(robot, np.arange(n), "q", gain=10.0)],
        [JointLimits(robot, v_max=5 * np.ones(n))],
    )
    return Retargeter(solver, preprocess, initial_q=lambda frame: jnp.zeros(n))


def _sinusoids(T: int, phase: float = 0.0) -> np.ndarray:
    t = np.arange(T)[:, None] * DT
    return np.sin(2 * np.pi * t * np.array([0.5, 1.0, 2.0]) + phase)


def test_online_matches_offline():
    frames = jnp.asarray(_sinusoids(30))
    q = _assert_online_matches_offline(_joint_retargeter(3), frames)
    assert q.shape == (30, 3)


def _compiles_while_stepping(online, frames) -> int:
    """Number of XLA compilations while stepping through frames"""
    compiles = []

    def listener(name, duration, **kwargs):
        if name == "/jax/core/compile/backend_compile_duration":
            compiles.append(name)

    jax.monitoring.register_event_duration_secs_listener(listener)
    try:
        for frame in frames:
            online.step(frame, DT)
    finally:
        jax.monitoring.unregister_event_duration_listener(listener)
    return len(compiles)


def test_online_compile_ahead_of_time():
    """compile() builds init and step without running them, so stepping
    afterwards compiles nothing (and without it, stepping compiles)"""
    frames = _sinusoids(5)
    online = OnlineRetargeter(_joint_retargeter(3))
    online.compile(jax.ShapeDtypeStruct(frames.shape[1:], frames.dtype))
    assert online.state is None
    assert _compiles_while_stepping(online, frames) == 0
    assert _compiles_while_stepping(OnlineRetargeter(_joint_retargeter(3)), frames) > 0


@pytest.mark.parametrize("num_devices", [1, 4])
def test_offline_matches_retarget_clip(num_devices):
    """Each clip's configurations match retargeting it alone, in input order,
    with clips of uneven lengths (including one frame), more clips than
    devices, and pytree frames. Compiled shapes are reused across calls"""
    retargeter = _joint_retargeter(3, preprocess=lambda frame: {"q": frame["target"]})
    lengths = [5, 23, 1, 17, 9, 30, 12]
    clips = [{"target": _sinusoids(T, phase=k)} for k, T in enumerate(lengths)]
    devices = jax.devices()[:num_devices]
    retarget = OfflineRetargeter(retargeter, DT, bucket=8, devices=devices)
    qs = retarget(clips)
    assert [len(q) for q in qs] == lengths
    for clip, q in zip(clips, qs, strict=True):
        expected, _, _ = retargeter.retarget_clip(clip, DT)
        np.testing.assert_allclose(q, expected, atol=1e-9)
    # Groups of num_devices clips, longest first, each padded to a multiple of 8
    groups = [
        sorted(lengths, reverse=True)[i : i + num_devices]
        for i in range(0, 7, num_devices)
    ]
    assert len(retarget.compiled) == len({-(-g[0] // 8) for g in groups})
    compiled = dict(retarget.compiled)
    retarget(clips[::-1])
    assert retarget.compiled == compiled


def test_global_ik_init_escapes_local_minimum():
    """The first frame converges from initial_q to a local minimum, unless
    solved by global IK (the cubic of test_global_ik.py: e(q) = q^3 - 3q + 3,
    with a local minimum of e^2 at q = 1, and its root at q ~ -2.104). The
    sampling range defaults to the solver's joint limits"""
    robot = ToyRobot(1)
    task = ResidualTask(robot, lambda q, ref: q**3 - 3 * q + ref["c"], 1, gain=10.0)
    limits = JointLimits(
        robot, q_min=np.array([-3.0]), q_max=np.array([3.0]), v_max=np.array([10.0])
    )
    solver = DiffIKSolver(robot, [task], [limits])
    kwargs = {
        "preprocess": lambda frame: {"c": frame},
        "initial_q": lambda frame: jnp.array([1.5]),
        "converge_max_iters": 500,
    }
    frames = jnp.full(5, 3.0)
    local, _, _ = Retargeter(solver, **kwargs).retarget_clip(frames, DT)
    np.testing.assert_allclose(local[0], [1.0], atol=1e-2)
    # As converge onto the first frame: 500 iterations, to its tolerance
    global_ik = GlobalIKSolver(solver, num_samples=16, max_iters=500, tol=1e-3)
    retargeter = Retargeter(solver, global_ik_init=global_ik, **kwargs)
    q, _, _ = retargeter.retarget_clip(frames, DT)
    np.testing.assert_allclose(q, -2.1038, atol=1e-3)
