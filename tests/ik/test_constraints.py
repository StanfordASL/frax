"""Constraint rows against reference implementations and autodiff"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from toy_robot import ToyRobot

from frax.ik import (
    BarrierConstraint,
    FloorConstraint,
    JointLimits,
    ObstacleConstraint,
    ResidualEquality,
    SelfCollisionConstraint,
    SphereObstacles,
)

TOL = 1e-12


def _cbf_rows(h_fn, q, gain, robot=None):
    """Reference (G, h) for a kinematic CBF: -dh/dq E(q) v <= gain * h, with
    E(q) from the robot (identity if None)"""
    J = jax.jacfwd(h_fn)(q)
    if robot is not None:
        J = J @ robot.velocity_to_qdot_map(q)
    return -J, gain * h_fn(q)


def test_joint_limits_velocity_only():
    con = JointLimits(
        ToyRobot(2), v_max=np.array([1.0, 2.0]), v_min=np.array([-3.0, -4.0])
    )
    G, h = con.rows(jnp.zeros(2), {})
    np.testing.assert_allclose(G, [[1, 0], [0, 1], [-1, 0], [0, -1]])
    np.testing.assert_allclose(h, [1, 2, 3, 4])


def test_joint_limits_position_only():
    """A linear CBF on the position limits, with no rows for unlimited entries"""
    q = jnp.array([0.0, 0.5, -0.2, 9.0])
    idxs = np.array([1, 2])
    q_min = np.array([-np.inf, -1.0, -0.5, -np.inf])
    q_max = np.array([np.inf, 1.0, 0.5, np.inf])
    con = JointLimits(ToyRobot(4), q_min=q_min, q_max=q_max, gain=2.0)
    G, h = con.rows(q, {})
    assert con.num_rows == 4

    def h_fn(q):
        return jnp.concatenate([q_max[idxs] - q[idxs], q[idxs] - q_min[idxs]])

    G_ref, h_ref = _cbf_rows(h_fn, q, 2.0)
    np.testing.assert_allclose(G, G_ref)
    np.testing.assert_allclose(h, h_ref)


def test_joint_limits_tighter_of_both():
    """Each bound is the tighter of the velocity and position bounds, with
    rows only for entries and sides with a finite limit"""
    q = jnp.array([0.0, 0.5, -0.2, 9.0])
    con = JointLimits(
        ToyRobot(4),
        q_min=np.array([-np.inf, -1.0, -0.5, -np.inf]),
        q_max=np.array([np.inf, 1.0, 0.5, 10.0]),
        v_max=np.array([1.0, 0.5, 3.0, np.inf]),
        v_min=np.array([-4.0, -3.0, -2.0, -np.inf]),
        gain=2.0,
    )
    G, h = con.rows(q, {})
    # Entry 3 has only an upper (position) limit
    np.testing.assert_allclose(G, np.vstack([np.eye(4), -np.eye(4)[:3]]))
    # Upper: [v, v, q, q] bounds. Lower: [v, q, q]
    np.testing.assert_allclose(h, [1.0, 0.5, 1.4, 2.0, 4.0, 3.0, 0.6])


def test_joint_limits_none_is_no_limit():
    """None turns a limit off, and v_min defaults to -v_max (or None with it)"""
    robot = ToyRobot(2)
    con = JointLimits(robot, v_max=np.array([1.0, 2.0]), v_min=None)
    G, h = con.rows(jnp.zeros(2), {})
    np.testing.assert_allclose(G, np.eye(2))  # Upper rows only
    np.testing.assert_allclose(h, [1, 2])
    con = JointLimits(robot, q_min=-np.ones(2), q_max=np.ones(2), v_max=None)
    assert np.all(np.isinf(con.v_min)) and np.all(np.isinf(con.v_max))


def test_joint_limits_requires_a_limit():
    robot = ToyRobot(3)
    with pytest.raises(AssertionError, match="No finite"):
        JointLimits(robot)
    with pytest.raises(AssertionError, match="No finite"):
        JointLimits(robot, v_max=np.full(3, np.inf))
    with pytest.raises(AssertionError, match="shape"):
        JointLimits(robot, q_min=-np.ones(3), q_max=np.ones(2))
    with pytest.raises(AssertionError, match="shape"):
        JointLimits(ToyRobot(4), v_max=np.ones(3))
    with pytest.raises(AssertionError, match='"robot"'):
        JointLimits(robot, v_max="robt")


def test_joint_limits_quaternion_base(g1, g1_q_random):
    """Limits cover the actuated joints only: zero columns for the floating base,
    and positions read past the quaternion"""
    q_min, q_max = g1.actuated_joint_lower_limits, g1.actuated_joint_upper_limits
    v_max = g1.actuated_joint_max_velocities
    con = JointLimits(g1, gain=2.0)  # The robot's own limits
    G, h = con.rows(jnp.asarray(g1_q_random), {})
    n = g1.num_actuated_joints
    np.testing.assert_allclose(
        G, np.block([[np.zeros((n, 6)), np.eye(n)], [np.zeros((n, 6)), -np.eye(n)]])
    )
    pos = g1_q_random[7:]
    upper = np.minimum(v_max, 2.0 * (q_max - pos))
    lower = np.maximum(-v_max, 2.0 * (q_min - pos))
    np.testing.assert_allclose(h, np.r_[upper, -lower])
    with pytest.raises(AssertionError, match="shape"):
        JointLimits(g1, v_max=np.ones(g1.nv))


@pytest.mark.parametrize("robot_name", ["g1", "panda"])
def test_self_collision_matches_frax(robot_name, request):
    """Matches frax's self-collision distances and their autodiff Jacobian.
    The Panda also has spheres fixed to the world root"""
    robot = request.getfixturevalue(robot_name)
    rng = np.random.default_rng(1)
    q = robot.integrate(
        robot.default_configuration, rng.uniform(-1.0, 1.0, robot.nv), 1.0
    )
    con = SelfCollisionConstraint(robot, gain=3.0)
    G, h = jax.jit(con.rows)(q, {})
    G_ref, h_ref = _cbf_rows(robot.self_collision_distances, q, 3.0, robot)
    assert con.num_rows == len(h_ref)
    np.testing.assert_allclose(h, h_ref, atol=TOL)
    np.testing.assert_allclose(G, G_ref, atol=TOL)


def test_self_collision_joint_pairs_matches_autodiff(g1, g1_q_random):
    """Joint pairs check every sphere on one joint against every sphere on the
    other"""
    idx = g1.joint_name_to_index
    hand, hip = idx["left_wrist_yaw_joint"], idx["left_hip_roll_joint"]
    pairs = [(hand, hip, 0.0), (hand, idx["right_elbow_joint"], -0.02)]
    con = SelfCollisionConstraint(g1, pairs=pairs, gain=3.0)
    q = jnp.asarray(g1_q_random)
    G, h = jax.jit(con.rows)(q, {})
    joints = np.asarray(con.sphere_joints)

    def distances(q):
        pos, radii = g1.link_collision_data(q)
        return jnp.stack(
            [
                jnp.linalg.norm(pos[j] - pos[i]) - radii[i] - radii[j] - tol
                for a, b, tol in pairs
                for i in np.flatnonzero(joints == a)
                for j in np.flatnonzero(joints == b)
            ]
        )

    G_ref, h_ref = _cbf_rows(distances, q, 3.0, g1)
    assert con.num_rows == len(h_ref)
    np.testing.assert_allclose(h, h_ref, atol=TOL)
    np.testing.assert_allclose(G, G_ref, atol=TOL)


def _obstacle_distances(robot, q, obstacles, sphere_idxs):
    pos, radii = robot.link_collision_data(q)
    pos, radii = pos[sphere_idxs], radii[sphere_idxs]
    d = (
        jnp.linalg.norm(pos[None] - obstacles.centers[:, None], axis=-1)
        - radii[None]
        - obstacles.radii[:, None]
    )
    return d.reshape(-1)


@pytest.fixture
def obstacles(g1_q_random):
    rng = np.random.default_rng(2)
    centers = g1_q_random[:3] + rng.uniform(-0.5, 0.5, (3, 3))
    return SphereObstacles(
        centers, np.array([0.05, 0.1, 0.2]), np.array([True, False, True])
    )


def test_obstacles_match_reference(g1, g1_q_random, obstacles):
    sphere_idxs = np.arange(10, 46)
    con = ObstacleConstraint(g1, obstacles, sphere_idxs=sphere_idxs, gain=3.0)
    q = jnp.asarray(g1_q_random)
    G, h = jax.jit(con.rows)(q, {})
    assert con.num_rows == 3 * len(sphere_idxs)

    G_ref, h_ref = _cbf_rows(
        lambda q: _obstacle_distances(g1, q, obstacles, sphere_idxs), q, 3.0, g1
    )
    active = np.repeat(obstacles.active, len(sphere_idxs))
    np.testing.assert_allclose(h[active], h_ref[active], atol=TOL)
    np.testing.assert_allclose(G[active], G_ref[active], atol=TOL)
    # Inactive obstacles: trivially satisfied rows
    assert np.all(G[~active] == 0.0)
    assert np.all(h[~active] > 0.0)


def test_obstacles_from_ref(g1, g1_q_random, obstacles):
    """ref[ref_key] replaces the constructor's obstacles"""
    con = ObstacleConstraint(g1, obstacles, ref_key="obstacles", gain=1.0)
    moved = SphereObstacles(
        jnp.asarray(obstacles.centers) + 0.1,
        jnp.asarray(obstacles.radii),
        jnp.ones(3, bool),
    )
    q = jnp.asarray(g1_q_random)
    _, h = jax.jit(con.rows)(q, {"obstacles": moved})
    h_ref = _obstacle_distances(g1, q, moved, np.arange(len(con.radii)))
    np.testing.assert_allclose(h, h_ref, atol=TOL)


@pytest.mark.parametrize("k", [1, 5, 20])
def test_max_rows_keeps_closest(g1, g1_q_random, k):
    q = jnp.asarray(g1_q_random)
    G_all, h_all = SelfCollisionConstraint(g1).rows(q, {})
    con = SelfCollisionConstraint(g1, max_rows=k)
    G, h = con.rows(q, {})
    assert con.num_rows == k
    closest = np.argsort(np.asarray(h_all))[:k]
    order = np.argsort(np.asarray(h))
    np.testing.assert_allclose(h[order], h_all[closest])
    np.testing.assert_allclose(G[order], G_all[closest])


def test_max_rows_above_candidates_keeps_all(g1):
    assert SelfCollisionConstraint(g1, max_rows=10_000).num_rows == len(
        g1.body_sc_pairs
    )


def test_barrier_matches_autodiff(g1, g1_q_random):
    num_rows = 6
    links = np.tile([27, 34, 24, 31, 9, 15], 4)[:num_rows]
    scale = 1.0 + 0.1 * jnp.arange(num_rows)

    def h_fn(q, ref):
        return g1.link_to_world_transforms(q)[links, 2, 3] * scale - ref["z_min"]

    ref = {"z_min": 0.3}
    q = jnp.asarray(g1_q_random)
    con = BarrierConstraint(g1, h_fn, num_rows, gain=4.0)
    G, h = jax.jit(con.rows)(q, ref)
    G_ref, h_ref = _cbf_rows(lambda q: h_fn(q, ref), q, 4.0, g1)
    np.testing.assert_allclose(h, h_ref, atol=TOL)
    np.testing.assert_allclose(G, G_ref, atol=TOL)


def test_barrier_checks_shape():
    con = BarrierConstraint(ToyRobot(2), lambda q, ref: q @ q, num_rows=1)  # Not (1,)
    with pytest.raises(AssertionError, match="expected"):
        con.rows(jnp.ones(2), {})


def _equality_rows(e_fn, q, gain, robot=None):
    """Reference (A, b) for a kinematic equality: de/dq E(q) v = -gain * e"""
    G, h = _cbf_rows(e_fn, q, gain, robot)
    return -G, -h


def test_residual_equality_matches_autodiff(g1, g1_q_random):
    num_rows = 6
    links = np.tile([27, 34, 24, 31, 9, 15], 4)[:num_rows]

    def e_fn(q, ref):
        return g1.link_to_world_transforms(q)[links, 0, 3] - ref["x"]

    ref = {"x": 0.1}
    q = jnp.asarray(g1_q_random)
    con = ResidualEquality(g1, e_fn, num_rows, gain=4.0)
    A, b = jax.jit(con.rows)(q, ref)
    A_ref, b_ref = _equality_rows(lambda q: e_fn(q, ref), q, 4.0, g1)
    np.testing.assert_allclose(A, A_ref, atol=TOL)
    np.testing.assert_allclose(b, b_ref, atol=TOL)


def test_floor_matches_autodiff(g1, g1_q_random):
    """Sphere bottoms (excluding some joints' spheres), then points, above z = 0"""
    exclude = [
        g1.joint_name_to_index[n] for n in ("left_knee_joint", "waist_pitch_joint")
    ]
    points = [
        (
            g1.joint_name_to_index["left_ankle_roll_joint"],
            np.array([0.1, 0.02, -0.035]),
        ),
        (g1.joint_name_to_index["right_wrist_yaw_joint"], np.array([0.0, 0.0, 0.0])),
    ]
    con = FloorConstraint(g1, exclude_joints=exclude, points=points, gain=3.0)
    q = jnp.asarray(g1_q_random)
    G, h = jax.jit(con.rows)(q, {})
    kept = ~np.isin(con.sphere_joints, exclude)
    assert 0 < kept.sum() < len(kept)
    assert con.num_rows == kept.sum() + len(points)

    def heights(q):
        pos, radii = g1.link_collision_data(q)
        tfs = g1.joint_to_world_transforms(q)
        pts = [tfs[j, :3, :3] @ o + tfs[j, :3, 3] for j, o in points]
        return jnp.concatenate([(pos[:, 2] - radii)[kept], jnp.stack(pts)[:, 2]])

    G_ref, h_ref = _cbf_rows(heights, q, 3.0, g1)
    np.testing.assert_allclose(h, h_ref, atol=TOL)
    np.testing.assert_allclose(G, G_ref, atol=TOL)


def test_sphere_constraint_margins(g1, g1_q_random, obstacles):
    """A margin shifts every barrier value down by the margin, rows unchanged"""
    q = jnp.asarray(g1_q_random)
    for make in (
        lambda m: SelfCollisionConstraint(g1, gain=2.0, margin=m),
        lambda m: ObstacleConstraint(g1, obstacles, gain=2.0, margin=m),
        lambda m: FloorConstraint(g1, points=[(5, np.zeros(3))], gain=2.0, margin=m),
    ):
        G0, h0 = make(0.0).rows(q, {})
        G1, h1 = make(0.03).rows(q, {})
        active = np.asarray(h0) < 100.0  # Skip inactive obstacles' rows
        np.testing.assert_allclose(G1, G0)
        np.testing.assert_allclose(h1[active], h0[active] - 2.0 * 0.03, atol=1e-12)
