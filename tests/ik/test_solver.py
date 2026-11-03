"""Closed-loop behavior: constraints hold when tracking pushes against them"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from toy_robot import ToyRobot

from frax.ik import (
    BarrierConstraint,
    DiffIKSolver,
    EETask,
    ElasticEquality,
    FrameTask,
    JointLimits,
    JointTask,
    ResidualEquality,
    ResidualTask,
    SelfCollisionConstraint,
)
from frax.utils.rotation_utils import (
    quat_wxyz_conjugate,
    quat_wxyz_exp,
    quat_wxyz_log,
    quat_wxyz_multiply,
    quat_wxyz_to_rmat,
)

DT = 0.02


def _rollout(solver, q0, refs):
    """Run the solver over stacked references. Returns q (T, nq) and QP info"""

    @jax.jit
    def run(q0):
        def step(state, ref):
            state, _, info = solver.step(state, ref, DT)
            return state, (state.q, info)

        return jax.lax.scan(step, solver.init(q0), refs)[1]

    qs, info = run(jnp.asarray(q0))
    return np.asarray(qs), info


def test_penalties_broadcast():
    """Scalar and per-row penalties are stacked in constraint order"""
    robot = ToyRobot(2)
    solver = DiffIKSolver(
        robot,
        [JointTask(robot, [0, 1], "q")],
        [
            JointLimits(robot, v_max=np.ones(2), penalty=5.0),
            BarrierConstraint(robot, lambda q, ref: q, 2, penalty=np.array([1.0, 2.0])),
        ],
    )
    np.testing.assert_allclose(solver.penalty, [5, 5, 5, 5, 1, 2])


def test_elastic_equality_penalties_and_shapes():
    """Elastic equalities add an upper and a lower inequality row each, in
    constraint order. Hard equalities are QP equality rows"""
    robot = ToyRobot(3)
    solver = DiffIKSolver(
        robot,
        [JointTask(robot, [0, 1, 2], "q")],
        [
            ElasticEquality(
                ResidualEquality(robot, lambda q, ref: q[:2], 2),
                penalty=np.array([3.0, 4.0]),
            ),
            JointLimits(robot, v_max=np.ones(3), penalty=5.0),
            ResidualEquality(robot, lambda q, ref: q[2:], 1),
        ],
    )
    np.testing.assert_allclose(solver.penalty, [3, 4, 3, 4] + [5] * 6)
    _, y, z = solver.init(jnp.zeros(3)).qp_warm_start
    assert y.shape == (1,) and z.shape == (10,)


def _diagonal_target(elastic):
    """Track (1, 0), with an equality q0 = q1 and no other constraints"""
    robot = ToyRobot(2)
    eq = ResidualEquality(robot, lambda q, ref: q[:1] - q[1:], 1)
    if elastic:
        eq = ElasticEquality(eq)
    return DiffIKSolver(robot, [JointTask(robot, [0, 1], "q", gain=10.0)], [eq])


@pytest.mark.parametrize("elastic", [True, False])
def test_equality_holds(elastic):
    refs = {"q": jnp.tile(jnp.array([1.0, 0.0]), (200, 1))}
    qs, info = _rollout(_diagonal_target(elastic), np.zeros(2), refs)
    np.testing.assert_allclose(qs[:, 0], qs[:, 1], atol=1e-8)
    np.testing.assert_allclose(qs[-1], [0.5, 0.5], atol=1e-6)
    assert np.all(info.qp_converged)


@pytest.mark.parametrize("elastic", [True, False])
def test_elastic_equality_yields_to_obstacle(elastic):
    """A point held on the line q1 = 0 tracks a target past a disc obstacle on
    the line. An elastic line (penalized below the tracking pull) gives way, so
    the point goes around the obstacle to the target. A hard line can't, so the
    point stops at the obstacle"""
    center, radius = jnp.array([1.0, 0.05]), 0.3
    robot = ToyRobot(2)
    line = ResidualEquality(robot, lambda q, ref: q[1:], 1, gain=10.0)
    if elastic:
        line = ElasticEquality(line, penalty=1.0)
    disc = BarrierConstraint(
        robot, lambda q, ref: jnp.array([jnp.linalg.norm(q - center) - radius]), 1
    )
    solver = DiffIKSolver(
        robot, [JointTask(robot, [0, 1], "q", gain=10.0)], [line, disc]
    )
    refs = {"q": jnp.tile(jnp.array([2.0, 0.0]), (400, 1))}
    qs, info = _rollout(solver, np.zeros(2), refs)
    assert np.all(info.qp_converged)
    assert np.linalg.norm(qs - np.asarray(center), axis=1).min() >= radius - 1e-3
    if elastic:
        assert np.abs(qs[:, 1]).max() > 0.2  # Left the line to pass the obstacle
        np.testing.assert_allclose(qs[-1], [2.0, 0.0], atol=1e-3)
    else:
        np.testing.assert_allclose(qs[:, 1], 0.0, atol=1e-8)
        assert qs[-1, 0] < 1.0 - radius + 0.05


def test_rejects_unknown_constraint_type():
    """Constraints must be InequalityConstraints or EqualityConstraints"""
    with pytest.raises(AssertionError, match="neither"):
        robot = ToyRobot(1)
        DiffIKSolver(robot, [JointTask(robot, [0], "q")], [object()])


def test_rejects_terms_for_another_robot():
    """Every task and constraint must be built for the solver's robot"""
    robot = ToyRobot(2)
    with pytest.raises(AssertionError, match="different robot"):
        DiffIKSolver(robot, [JointTask(ToyRobot(2), [0], "q")])


def test_rejects_wrong_q_shape(g1):
    """q must be the configuration (nq,), not e.g. a 6-DOF euler root"""
    solver = DiffIKSolver(
        g1,
        [JointTask(g1, [0], "q")],
        [
            JointLimits(
                g1, q_min=None, q_max=None, v_max=np.ones(g1.num_actuated_joints)
            )
        ],
    )
    with pytest.raises(AssertionError, match="expected"):
        solver.step(solver.init(jnp.zeros(g1.nv)), {"q": jnp.zeros(1)}, DT)


@pytest.mark.parametrize("task_type", ["frame", "residual"])
def test_quaternion_base_converges(g1, g1_q0, task_type):
    """Rotate the root by more than 90 degrees in pitch (the euler singularity)
    while holding the root position and joints, via an analytic (FrameTask) or
    autodiff (ResidualTask, through the quaternion in q) Jacobian"""
    quat_target = quat_wxyz_multiply(
        quat_wxyz_exp(jnp.array([0.0, 2.0, 0.0])),
        quat_wxyz_exp(jnp.array([0.5, 0.0, 0.0])),
    )
    R_target = np.asarray(quat_wxyz_to_rmat(quat_target))
    if task_type == "frame":
        base = g1.nv_floating - 1  # The base's frame is the last virtual body's
        root_task = FrameTask(g1, [base], gain_pos=10.0, gain_rot=10.0)
    else:

        def root_error(q, ref):
            # Rotation vector of the root relative to the target
            return quat_wxyz_log(
                quat_wxyz_multiply(quat_wxyz_conjugate(quat_target), q[3:7])
            )

        root_task = ResidualTask(g1, root_error, 3, gain=10.0)
    hold = np.r_[0:3, 6 : g1.nv]  # Velocity indices with q_dot = v
    tasks = [root_task, JointTask(g1, hold, "q", gain=10.0)]
    solver = DiffIKSolver(
        g1,
        tasks,
        [
            JointLimits(
                g1, q_min=None, q_max=None, v_max=10.0 * np.ones(g1.num_actuated_joints)
            )
        ],
    )
    ref = {
        "frame_pos": g1_q0[None, :3],
        "frame_rmat": R_target[None],
        "q": g1_q0[[g1.velocity_to_configuration_index[i] for i in hold]],
    }
    state, iters = jax.jit(solver.converge, static_argnums=(2, 3, 4))(
        solver.init(g1_q0), ref, 500, 0.05, 1e-8
    )
    q = np.asarray(state.q)
    assert iters < 500
    np.testing.assert_allclose(np.linalg.norm(q[3:7]), 1.0, atol=1e-12)
    np.testing.assert_allclose(quat_wxyz_to_rmat(q[3:7]), R_target, atol=1e-6)
    np.testing.assert_allclose(
        q[[g1.velocity_to_configuration_index[i] for i in hold]], ref["q"], atol=1e-6
    )


def test_joint_task_converges():
    robot = ToyRobot(3)
    solver = DiffIKSolver(
        robot,
        [JointTask(robot, [0, 2], "q", gain=10.0)],
        [JointLimits(robot, v_max=10 * np.ones(3))],
    )
    refs = {"q": jnp.tile(jnp.array([0.5, -0.3]), (200, 1))}
    qs, info = _rollout(solver, np.zeros(3), refs)
    np.testing.assert_allclose(qs[-1], [0.5, 0.0, -0.3], atol=1e-6)
    assert np.all(info.qp_converged)


def test_position_limits_hold():
    robot = ToyRobot(2)
    limits = JointLimits(
        robot, q_min=np.array([-np.inf, -0.2]), q_max=np.array([np.inf, 0.2])
    )
    solver = DiffIKSolver(robot, [JointTask(robot, [0, 1], "q", gain=10.0)], [limits])
    refs = {"q": jnp.tile(jnp.array([1.0, 1.0]), (200, 1))}
    qs, _ = _rollout(solver, np.zeros(2), refs)
    assert qs[:, 1].max() <= 0.2 + 1e-6
    np.testing.assert_allclose(qs[-1], [1.0, 0.2], atol=1e-5)


def test_joint_limits_hold():
    """Velocity limits bind early, position limits later"""
    robot = ToyRobot(3)
    solver = DiffIKSolver(
        robot,
        [JointTask(robot, [0, 1, 2], "q", gain=10.0)],
        [
            JointLimits(
                robot,
                q_min=np.array([-np.inf, -0.2, -0.5]),
                q_max=np.array([np.inf, 0.2, 0.5]),
                v_max=np.ones(3),
            )
        ],
    )
    refs = {"q": jnp.tile(jnp.array([1.0, 1.0, -1.0]), (200, 1))}
    qs, _ = _rollout(solver, np.zeros(3), refs)
    np.testing.assert_allclose(np.abs(np.diff(qs[:10], axis=0)).max(), DT, rtol=1e-6)
    assert np.abs(np.diff(qs, axis=0)).max() <= DT + 1e-6
    assert qs[:, 1].max() <= 0.2 + 1e-6 and qs[:, 2].min() >= -0.5 - 1e-6
    np.testing.assert_allclose(qs[-1], [1.0, 0.2, -0.5], atol=1e-5)


def test_barrier_holds():
    """Nonlinear barrier (stay inside the unit circle) against a target outside it"""
    robot = ToyRobot(2)
    solver = DiffIKSolver(
        robot,
        [JointTask(robot, [0, 1], "q", gain=10.0, weight=100.0)],
        [BarrierConstraint(robot, lambda q, ref: jnp.array([1.0 - q @ q]), 1)],
    )
    refs = {"q": jnp.tile(jnp.array([3.0, 4.0]), (300, 1))}
    qs, _ = _rollout(solver, np.zeros(2), refs)
    assert np.linalg.norm(qs, axis=1).max() <= 1.0 + 1e-3
    # Ends at the closest point on the boundary
    np.testing.assert_allclose(qs[-1], [0.6, 0.8], atol=1e-3)


def _crossed_arms_goal(g1, q0):
    """Arms swung inward across the torso, legs crossed: ~10 sphere pairs overlap"""

    def names(name):
        return g1.velocity_to_configuration_index[g1.joint_name_to_index[name]]

    q = q0.copy()
    for side, sgn in (("left", 1), ("right", -1)):
        q[names(f"{side}_shoulder_pitch_joint")] = -0.6
        q[names(f"{side}_shoulder_roll_joint")] = -sgn * 0.6
        q[names(f"{side}_shoulder_yaw_joint")] = 0.0
        q[names(f"{side}_elbow_joint")] = 0.6
        q[names(f"{side}_hip_roll_joint")] = -sgn * 0.35
    return q


def test_self_collision_holds(g1, g1_q0):
    q_goal = _crossed_arms_goal(g1, g1_q0)
    assert g1.self_collision_distances(q_goal).min() < -0.05

    T = 150
    s = np.clip(np.arange(T) / 75, 0, 1)[:, None]
    # The root orientation (identity in both q0 and the goal) is tracked separately
    idxs = np.r_[0:3, 6 : g1.nv]
    q_ref = (1 - s) * g1_q0 + s * q_goal
    refs = {
        "q": jnp.asarray(
            q_ref[:, [g1.velocity_to_configuration_index[i] for i in idxs]]
        )
    }
    base = g1.nv_floating - 1  # The base's frame is the last virtual body's
    tasks = [
        JointTask(g1, idxs, "q", gain=10.0),
        FrameTask(g1, [base], gain_pos=10.0, gain_rot=10.0),
    ]
    refs["frame_pos"] = jnp.tile(g1_q0[:3], (T, 1, 1))
    refs["frame_rmat"] = jnp.tile(jnp.eye(3), (T, 1, 1, 1))
    vel = JointLimits(g1, q_min=None, q_max=None)
    min_dist = jax.jit(jax.vmap(lambda q: g1.self_collision_distances(q).min()))

    def run(constraints):
        return _rollout(DiffIKSolver(g1, tasks, constraints), g1_q0, refs)[0]

    q_free = run([vel])
    q_all = run([vel, SelfCollisionConstraint(g1)])

    assert min_dist(q_free).min() < -0.05  # Collides without the constraint
    assert min_dist(q_all).min() > -1e-4

    # Enough rows for every active pair: the same step as keeping all pairs, from
    # each state along the motion. Whole motions aren't compared: the arms end up
    # pressed head-on into each other, an unstable equilibrium (the pose and the
    # collision model are symmetric) where rounding differences grow until one arm
    # slips past the other
    def velocities(constraint):
        solver = DiffIKSolver(g1, tasks, [vel, constraint])
        step = jax.vmap(lambda q, ref: solver.step(solver.init(q), ref, DT)[1])
        return jax.jit(step)(
            jnp.asarray(q_all[:-1]), jax.tree.map(lambda x: x[1:], refs)
        )

    np.testing.assert_allclose(
        velocities(SelfCollisionConstraint(g1, max_rows=32)),
        velocities(SelfCollisionConstraint(g1)),
        atol=1e-6,
    )


PANDA_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])


def _panda_ee_solver(panda):
    """EE pose, with a posture task toward home in its nullspace (by weight)"""
    limits = JointLimits(panda)
    tasks = [
        EETask(panda, gain_pos=10.0, gain_rot=10.0),
        JointTask(panda, np.arange(7), "q_home", gain=1.0, weight=0.1),
    ]
    return DiffIKSolver(panda, tasks, [limits])


def test_converges_from_large_rotation_error(panda):
    """An EE target 170 deg about the flange axis (reachable with the last joint)
    is reached: the rotation error keeps growing up to 180 deg, so the commanded
    angular velocity doesn't vanish there"""
    q_goal = PANDA_HOME.copy()
    q_goal[6] -= np.radians(170.0)
    tf = np.asarray(panda.ee_transform(q_goal))
    ref = {"ee_pos": tf[:3, 3], "ee_rmat": tf[:3, :3], "q_home": PANDA_HOME}
    solver = _panda_ee_solver(panda)
    state, iters = jax.jit(solver.converge, static_argnums=(2, 3, 4))(
        solver.init(PANDA_HOME), ref, 300, 0.02, 1e-3
    )
    assert iters < 300
    # Reached, up to the posture task's pull toward home (170 deg away at the wrist)
    T = np.asarray(panda.ee_transform(state.q))
    angle = np.arccos(np.clip((np.trace(T[:3, :3].T @ tf[:3, :3]) - 1) / 2, -1, 1))
    assert np.degrees(angle) < 1.0
    assert np.linalg.norm(T[:3, 3] - tf[:3, 3]) < 0.01


def test_converge_stops_out_of_reach(panda):
    """A target 2 m away: no step lowers the objective once the arm is stretched
    toward it, so converge stops on its own (well before max_iters), at a lower
    objective, and with the EE nearer the target"""
    target = np.array(panda.ee_transform(PANDA_HOME))
    target[:3, 3] += [2.0, 0.0, 0.0]
    ref = {"ee_pos": target[:3, 3], "ee_rmat": target[:3, :3], "q_home": PANDA_HOME}
    solver = _panda_ee_solver(panda)
    state, iters = jax.jit(solver.converge, static_argnums=(2, 3, 4))(
        solver.init(PANDA_HOME), ref, 1000, 0.02, 1e-3
    )
    assert iters < 1000
    assert solver.objective(state.q, ref) < solver.objective(
        jnp.asarray(PANDA_HOME), ref
    )
    reach = np.linalg.norm(
        np.asarray(panda.ee_transform(state.q))[:3, 3] - target[:3, 3]
    )
    assert reach < 2.0 - 0.3


def test_damping_settles_out_of_reach(panda):
    """A target 1.5 m away: with large residuals, Gauss-Newton underestimates the
    curvature, so the undamped step overshoots and cycles. Levenberg-Marquardt
    damping heads to the compromise without reversing"""
    direction = np.array([0.876, -0.426, 0.225])
    target = np.asarray(panda.ee_transform(PANDA_HOME))
    ref = {
        "ee_pos": target[:3, 3] + 1.5 * direction / np.linalg.norm(direction),
        "ee_rmat": target[:3, :3],
        "q_home": PANDA_HOME,
    }
    refs = jax.tree.map(lambda x: jnp.broadcast_to(x, (1500, *np.shape(x))), ref)
    undamped = _panda_ee_solver(panda)
    damped = DiffIKSolver(panda, undamped.tasks, undamped.constraints, damping=3e-3)

    def reversals(qs):
        """Steps after which some joint turns back, both steps over 1e-4"""
        dq = np.diff(qs[-200:], axis=0)
        return np.sum(np.any((dq[1:] * dq[:-1] < 0) & (np.abs(dq[1:]) > 1e-4), -1))

    qs, _ = _rollout(undamped, PANDA_HOME, refs)
    assert reversals(qs) > 100
    qs, info = _rollout(damped, PANDA_HOME, refs)
    assert reversals(qs) == 0
    assert np.all(np.diff(np.asarray(info.objective)[-1000:]) <= 1e-9)


def test_info_objective(panda):
    """SolverInfo's objective is DiffIKSolver.objective at the step's q"""
    target = np.asarray(panda.ee_transform(PANDA_HOME + 0.3))
    ref = {"ee_pos": target[:3, 3], "ee_rmat": target[:3, :3], "q_home": PANDA_HOME}
    solver = _panda_ee_solver(panda)
    q = jnp.asarray(PANDA_HOME)
    _, _, info = solver.step(solver.init(q), ref, 0.02)
    np.testing.assert_allclose(info.objective, solver.objective(q, ref), rtol=1e-12)
    assert info.objective > 0
