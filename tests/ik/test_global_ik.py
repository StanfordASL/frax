"""Multi-start global IK"""

import jax
import jax.numpy as jnp
import numpy as np
from toy_robot import ToyRobot

from frax.ik import (
    DiffIKSolver,
    EETask,
    GlobalIKSolver,
    JointLimits,
    ResidualTask,
    sample_configurations,
)


def _cubic_solver():
    """e(q) = q^3 - 3q + 3: e^2 has a local minimum at q = 1 (e = 1), and its
    only root is at q ~ -2.104. A velocity limit keeps the Gauss-Newton steps
    near the minimum (de/dq -> 0) from jumping over to the root"""
    robot = ToyRobot(1)
    task = ResidualTask(robot, lambda q, ref: q**3 - 3 * q + 3, 1, gain=1.0)
    limits = JointLimits(
        robot, q_min=np.array([-3.0]), q_max=np.array([3.0]), v_max=np.array([1.0])
    )
    return DiffIKSolver(robot, [task], [limits])


def test_escapes_local_minimum():
    solver = _cubic_solver()
    q = jnp.array([1.5])
    local, iters = solver.converge(solver.init(q), {}, 500, 0.1, 1e-8)
    np.testing.assert_allclose(local.q, [1.0], atol=1e-3)  # Stuck at the local minimum
    # ... where no step lowers the cost, so it stops early (the Gauss-Newton
    # step flips sign across the minimum, and a fixed stepsize would chatter)
    assert iters < 100

    global_ik = GlobalIKSolver(
        solver,
        num_samples=16,
        q_min=-3.0 * np.ones(1),
        q_max=3.0 * np.ones(1),
        max_iters=500,
    )
    result = jax.jit(global_ik.solve)(q, {}, jax.random.key(0))
    np.testing.assert_allclose(result.q, [-2.1038], atol=1e-3)
    # Converged to the default tolerance (on |v|) vs a cost of ~1 at the local minimum
    assert result.cost < 1e-5
    assert solver.cost(local.q, {}) > 0.5
    # The first start is q itself, which converges to the local minimum
    np.testing.assert_allclose(result.qs[0], local.q, atol=1e-3)


def _two_roots_solver():
    """Two exact solutions of q^2 = 1, at +-1"""
    robot = ToyRobot(1)
    task = ResidualTask(robot, lambda q, ref: q**2 - 1, 1, gain=1.0)
    limits = JointLimits(robot, q_min=np.array([-3.0]), q_max=np.array([3.0]))
    return DiffIKSolver(robot, [task], [limits])


def test_cost_tol_keeps_the_nearest_acceptable_solution():
    """e(q) = 1 + q^2 + 0.1 q^3 - 0.2 q^4 has a local minimum of e^2 at q = 0
    (e = 1), where q = 0.1 gets stuck, and roots at -2.216 and 2.653. With
    cost_tol, q alone isn't acceptable, so it samples, finds both roots, and
    keeps the nearer one"""
    robot = ToyRobot(1)
    task = ResidualTask(
        robot, lambda q, ref: 1 + q**2 + 0.1 * q**3 - 0.2 * q**4, 1, gain=1.0
    )
    limits = JointLimits(
        robot, q_min=np.array([-3.0]), q_max=np.array([3.0]), v_max=np.array([1.0])
    )
    solver = DiffIKSolver(robot, [task], [limits])
    global_ik = GlobalIKSolver(solver, num_samples=16, max_iters=500, cost_tol=1e-4)
    result = global_ik.solve(jnp.array([0.1]), {}, jax.random.key(1))
    assert result.sampled
    qs = np.asarray(result.qs)[:, 0]
    assert np.any(np.abs(qs + 2.2159) < 1e-3) and np.any(np.abs(qs - 2.6527) < 1e-3)
    np.testing.assert_allclose(result.q, [-2.2159], atol=1e-3)


def test_cost_tol_tries_q_alone_first():
    # q = 0.8 converges to the root at 1 alone: no sampling
    global_ik = GlobalIKSolver(
        _two_roots_solver(), num_samples=16, max_iters=500, cost_tol=1e-6
    )
    result = jax.jit(global_ik.solve)(jnp.array([0.8]), {}, jax.random.key(1))
    np.testing.assert_allclose(result.q, [1.0], atol=1e-3)
    assert not result.sampled
    # On the cubic, q = 1.5 is stuck at the local minimum (cost ~1), so it samples
    global_ik = GlobalIKSolver(
        _cubic_solver(), num_samples=16, max_iters=500, cost_tol=1e-4
    )
    result = jax.jit(global_ik.solve)(jnp.array([1.5]), {}, jax.random.key(0))
    np.testing.assert_allclose(result.q, [-2.1038], atol=1e-3)
    assert result.sampled


def test_devices_match_single_device(panda):
    """Starts split across devices (10 over 4, so padded) give the same answer
    as on one device. Not bitwise: XLA rounds differently for different batch
    sizes, and the accept/reject steps of converge amplify that into different
    paths for some starts (on one device too, e.g. batches of 3 vs 10)"""
    devices = tuple(jax.devices("cpu"))
    assert len(devices) == 4
    q_min, q_max = panda.actuated_joint_lower_limits, panda.actuated_joint_upper_limits
    solver = DiffIKSolver(
        panda,
        [EETask(panda, gain_pos=10.0, gain_rot=10.0)],
        [JointLimits(panda, q_min=q_min, q_max=q_max)],
    )
    tf = np.asarray(panda.ee_transform(np.random.default_rng(5).uniform(q_min, q_max)))
    ref = {"ee_pos": tf[:3, 3], "ee_rmat": tf[:3, :3]}
    args = (jnp.asarray(panda.default_configuration), ref, jax.random.key(0))

    def solve(devices="default"):
        global_ik = GlobalIKSolver(solver, num_samples=10, devices=devices)
        return jax.jit(global_ik.solve)(*args)

    single = solve()
    split = solve(devices)
    assert split.qs.shape == single.qs.shape == (10, 7)
    assert split.iters.shape == split.costs.shape == (10,)
    # The first start is q itself, unpadded: same path on any batch size here
    np.testing.assert_allclose(split.qs[0], single.qs[0], atol=1e-8)
    for result in (single, split):
        q = np.asarray(result.q)
        np.testing.assert_allclose(
            np.asarray(panda.ee_transform(q))[:3, 3], tf[:3, 3], atol=1e-3
        )
        assert result.cost < 1e-6
    # One device in the tuple is the unsplit path
    one = solve(devices[:1])
    np.testing.assert_array_equal(one.qs, single.qs)


def test_samples_within_limits_keep_floating_base(g1, g1_q_random):
    q_min, q_max = g1.actuated_joint_lower_limits, g1.actuated_joint_upper_limits
    qs = sample_configurations(g1, jnp.asarray(g1_q_random), jax.random.key(2), 8)
    assert qs.shape == (8, g1.nq)
    np.testing.assert_allclose(qs[0], g1_q_random)
    np.testing.assert_allclose(qs[:, :7], np.tile(g1_q_random[:7], (8, 1)))
    assert np.all(qs[1:, 7:] >= q_min) and np.all(qs[1:, 7:] <= q_max)


def test_sampling_range_narrows_robot_limits(panda):
    """The default range is the robot's limits, narrowed by the solver's
    JointLimits. A JointLimits bound of None leaves the robot's"""
    lower, upper = panda.actuated_joint_lower_limits, panda.actuated_joint_upper_limits
    q_max = np.array(upper, dtype=float)
    q_max[0] = 0.5
    limits = JointLimits(panda, q_min=None, q_max=q_max)
    solver = DiffIKSolver(panda, [EETask(panda)], [limits])
    global_ik = GlobalIKSolver(solver)
    np.testing.assert_allclose(global_ik.q_min, lower)
    np.testing.assert_allclose(global_ik.q_max, q_max)
