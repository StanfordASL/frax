"""Test cases for the Humanoid class"""

import unittest
from typing import Callable, Tuple

import jax
import numpy as np
import pinocchio as pin

from frax.core.humanoid import Humanoid
from frax.robots.unitree_g1 import load_fixed_root_g1, load_g1
from frax.assets import G1_ASSETS_DIR
from frax.utils.transform_utils import transform_jacobian_numpy

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

fixed_root_urdf = G1_ASSETS_DIR / "g1_29dof_rev_1_0.urdf"
floating_root_urdf = G1_ASSETS_DIR / "floating_g1_29dof_rev_1_0.urdf"


# Test case options:
# Model has a pinocchio freeflyer base attached to the 29dof urdf
# Model has 6dof pseudo joints attached (35dof)
# Model has fixed pelvis (29dof)


def sample_q(nq: int, nv: int) -> Tuple[np.ndarray, np.ndarray]:
    # This function assumes we're working with a 29dof humanoid like the unitree g1
    num_actuated_joints = 29
    num_unactuated_joints = 6
    q_actuated = np.random.uniform(-np.pi / 2, np.pi / 2, num_actuated_joints)
    if nq == 36 and nv == 35:
        # Working with the standard 29DOF URDF with a pinocchio freeflyer joint added
        zero_position_and_orientation_pin = pin.SE3ToXYZQUAT(pin.SE3.Identity())
        zero_q_floating = np.zeros(6)
        q_pin = np.concatenate([zero_position_and_orientation_pin, q_actuated])
        q_mine = np.concatenate([zero_q_floating, q_actuated])
    elif nq == 29 and nv == 29:
        # Working with the standard 29DOF URDF with a fixed pelvis
        q_mine = q_actuated
        q_pin = q_actuated
    elif nq == 35 and nv == 35:
        # Working with the 35DOF URDF (which has 6 joints to model the floating base)
        q_floating = np.random.uniform(-np.pi / 2, np.pi / 2, num_unactuated_joints)
        q_mine = np.concatenate([q_floating, q_actuated])
        q_pin = q_mine
    else:
        raise ValueError("Unrecognized nq and nv")
    return q_mine, q_pin


def sample_v(nq: int, nv: int) -> np.ndarray:
    # This function assumes we're working with a 29dof humanoid like the unitree g1
    num_actuated_joints = 29
    num_unactuated_joints = 6

    if nq == 36 and nv == 35:
        # Working with the standard 29DOF URDF with a pinocchio freeflyer joint added
        # NOTE: I'm not entirely sure how pinocchio defines velocities of the freeflyer
        # so we'll set them to 0 just for now
        zero_freeflyer_velocity = np.zeros(num_unactuated_joints)
        v = np.concatenate(
            [zero_freeflyer_velocity, np.random.rand(num_actuated_joints)]
        )
    elif (nq == 29 and nv == 29) or (nq == 35 and nv == 35):
        # Working with the standard 29DOF URDF with a fixed pelvis
        # OR working with the 35DOF URDF (which has 6 joints to model the floating base)
        # For both cases we can just sample nv velocities
        v = np.random.rand(nv)
    else:
        raise ValueError("Unrecognized nq and nv")
    return v


def check_mass_matrix(
    model: pin.Model, data: pin.Data, mass_matrix_func: Callable
) -> None:
    jit_mass_matrix = jax.jit(mass_matrix_func)

    for i in range(10):
        q_mine, q_pin = sample_q(model.nq, model.nv)
        v = np.zeros(model.nv)
        pin.forwardKinematics(model, data, q_pin, v)
        pin.updateFramePlacements(model, data)
        M = jit_mass_matrix(q_mine)
        Mpin = pin.crba(model, data, q_pin)
        np.testing.assert_array_almost_equal(M, Mpin, decimal=4)


def check_gravity_vector(
    model: pin.Model, data: pin.Data, gravity_vector_func: Callable
) -> None:
    jit_gravity_vector = jax.jit(gravity_vector_func)
    for i in range(10):
        q_mine, q_pin = sample_q(model.nq, model.nv)
        # If v = 0 then the bias from pinocchio will just be the grav vector
        v = np.zeros(model.nv)
        pin.forwardKinematics(model, data, q_pin, v)
        pin.updateFramePlacements(model, data)
        bias = pin.nle(model, data, q_pin, v)
        G = jit_gravity_vector(q_mine)
        np.testing.assert_array_almost_equal(G, bias, decimal=4)


def check_nonlinear_effects(
    model: pin.Model,
    data: pin.Data,
    gravity_vector_func: Callable,
    cc_vector_func: Callable,
    bias_func: Callable,
) -> None:

    # NOTE: We have two possible ways of computing the nonlinear effects (bias):
    # - Sum the gravity vector and the centrifugal/coriolis vector, each computed individually
    # - Directly compute the sum via rnea
    # We want to make sure that both methods yield the same result

    def my_nle(q, dq):
        return gravity_vector_func(q) + cc_vector_func(q, dq)

    def my_bias(q, dq):
        return bias_func(q, dq)

    jit_my_nle = jax.jit(my_nle)
    jit_my_bias = jax.jit(my_bias)
    for i in range(10):
        q_mine, q_pin = sample_q(model.nq, model.nv)
        v = sample_v(model.nq, model.nv)
        pin.forwardKinematics(model, data, q_pin, v)
        pin.updateFramePlacements(model, data)
        pin_bias = pin.nle(model, data, q_pin, v)
        my_nle_result = jit_my_nle(q_mine, v)
        my_bias_result = jit_my_bias(q_mine, v)
        np.testing.assert_array_almost_equal(my_nle_result, pin_bias, decimal=4)
        np.testing.assert_array_almost_equal(my_bias_result, pin_bias, decimal=4)


def check_kinematics_and_jacobians(
    robot: Humanoid, model: pin.Model, data: pin.Data
) -> None:

    # NOTE: Joint numbering is different if pinocchio has loaded the 29DOF URDF and added a 6DOF freeflyer joint to it
    # So, we'll use this variable to check if the pinocchio model has the ff joint
    pin_ff_joint = model.nq == 36 and model.nv == 35

    @jax.jit
    def get_my_data(q):
        tfs = robot.joint_to_world_transforms(q)
        joint_Jv_mine, joint_Jw_mine = robot._joint_jacobians(tfs)
        link_Jv_mine = robot._link_linear_jacobians(tfs)
        link_Jw_mine = robot._link_angular_jacobians(tfs)
        link_com_tfs = robot._link_to_world_transforms(tfs)
        return (
            joint_Jv_mine,
            joint_Jw_mine,
            link_Jv_mine,
            link_Jw_mine,
            tfs,
            link_com_tfs,
        )

    for i in range(10):
        q_mine, q_pin = sample_q(model.nq, model.nv)
        (
            joint_Jv_mine,
            joint_Jw_mine,
            link_Jv_mine,
            link_Jw_mine,
            joint_tfs,
            link_com_tfs,
        ) = get_my_data(q_mine)
        pin.forwardKinematics(model, data, q_pin)
        pin.updateFramePlacements(model, data)
        pin.computeJointJacobians(model, data, q_pin)

        for my_idx in range(robot.nv):
            # Note: pinocchio considers joint 0 as the connection to the universe, so we need to shift by 1
            # And if pinocchio is using their freeflyer joint, there will be a difference in the number of ff joints
            if pin_ff_joint and my_idx < 6:
                # Skip analysis between pinocchio's freeflyer and our 6DOF PPPRRR
                continue
            elif pin_ff_joint:
                # For joints that are not associated with the freeflyer, we can test their values
                pin_idx = my_idx - 4
            else:
                # If no freeflyer joint, we just need to account for the difference with "universe" joint
                pin_idx = my_idx + 1

            J_pin = pin.getJointJacobian(
                model, data, pin_idx, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )

            # fmt: off
            # Check the forward kinematics on the joint frame
            np.testing.assert_array_almost_equal(joint_tfs[my_idx, :3, :3], data.oMi[pin_idx].rotation)
            np.testing.assert_array_almost_equal(joint_tfs[my_idx, :3, 3], data.oMi[pin_idx].translation)
            # Check that the link inertial tensors/frames were parsed properly
            np.testing.assert_array_almost_equal(np.asarray(robot.link_local_inertias[my_idx]), model.inertias[pin_idx].inertia)
            np.testing.assert_array_almost_equal(np.asarray(robot.link_masses[my_idx]), model.inertias[pin_idx].mass)
            # Note: not sure how to check on the inertial rotation right now
            # fmt: on

            # Shift the jacobian to the link COM to match what we are computing
            # Note on pinocchio variables: oMi contains the vector of joint placements wrt the world
            p_joint = data.oMi[pin_idx].translation
            p_com = data.oMi[pin_idx].act(model.inertias[pin_idx].lever)
            # Check that the link COM position is where we expect
            np.testing.assert_array_almost_equal(link_com_tfs[my_idx, :3, 3], p_com)
            r = p_com - p_joint
            J_pin_shifted = transform_jacobian_numpy(J_pin, r)
            # Note: since these points (joint origin and link COM) are on the same body,
            # they have the same angular velocity, hence we don't need to shift Jw

            # Confirm jacobians are correct
            np.testing.assert_array_almost_equal(
                joint_Jv_mine[my_idx], J_pin[:3, :], decimal=5
            )
            np.testing.assert_array_almost_equal(
                joint_Jw_mine[my_idx], J_pin[3:, :], decimal=5
            )
            np.testing.assert_array_almost_equal(
                link_Jv_mine[my_idx], J_pin_shifted[:3, :], decimal=5
            )
            np.testing.assert_array_almost_equal(
                link_Jw_mine[my_idx], J_pin_shifted[3:, :], decimal=5
            )


def check_ee_tfs_and_jacobians(
    robot: Humanoid, model: pin.Model, data: pin.Data
) -> None:

    # NOTE: Joint numbering is different if pinocchio has loaded the 29DOF URDF and added a 6DOF freeflyer joint to it
    # So, we'll use this variable to check if the pinocchio model has the ff joint
    pin_ff_joint = model.nq == 36 and model.nv == 35

    @jax.jit
    def get_ee_jacobians(tfs):
        J_lh = robot._left_hand_jacobian(tfs)
        J_rh = robot._right_hand_jacobian(tfs)
        J_lf = robot._left_foot_jacobian(tfs)
        J_rf = robot._right_foot_jacobian(tfs)
        return J_lh, J_rh, J_lf, J_rf

    @jax.jit
    def get_ee_tfs(tfs):
        T_lh = robot._left_hand_transform(tfs)
        T_rh = robot._right_hand_transform(tfs)
        T_lf = robot._left_foot_transform(tfs)
        T_rf = robot._right_foot_transform(tfs)
        return T_lh, T_rh, T_lf, T_rf

    def _test_appendage(idx, ee_tf, J, joint_transforms):
        # Note: We haven't explicitly defined named frames in the URDF for pinocchio
        # to parse. So, we can't get the jacobians of our defined ee offsets directly.
        # But, we can transform the pinocchio joint jacobian to where the offset is defined
        # and confirm that we are the same
        pin_idx = idx - 4 if pin_ff_joint else idx + 1
        J_joint_pin = pin.getJointJacobian(
            model, data, pin_idx, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        r_joint_to_ee = ee_tf[:3, 3] - joint_transforms[idx, :3, 3]
        J_pin = transform_jacobian_numpy(J_joint_pin, r_joint_to_ee)
        np.testing.assert_allclose(J, J_pin)

    for i in range(5):
        q_mine, q_pin = sample_q(model.nq, model.nv)
        joint_transforms = robot.joint_to_world_transforms(q_mine)
        J_lh, J_rh, J_lf, J_rf = get_ee_jacobians(joint_transforms)
        T_lh, T_rh, T_lf, T_rf = get_ee_tfs(joint_transforms)
        pin.forwardKinematics(model, data, q_pin)
        pin.updateFramePlacements(model, data)
        pin.computeJointJacobians(model, data, q_pin)
        # LEFT HAND
        _test_appendage(
            int(robot.left_hand_parent_chain[-1]), T_lh, J_lh, joint_transforms
        )
        # RIGHT HAND
        _test_appendage(
            int(robot.right_hand_parent_chain[-1]), T_rh, J_rh, joint_transforms
        )
        # LEFT FOOT
        _test_appendage(
            int(robot.left_foot_parent_chain[-1]), T_lf, J_lf, joint_transforms
        )
        # RIGHT FOOT
        _test_appendage(
            int(robot.right_foot_parent_chain[-1]), T_rf, J_rf, joint_transforms
        )


class FreeflyerRootDynamicsTest(unittest.TestCase):
    """Test cases to compare my dynamics with 6 virtual links against Pinocchio's freeflyer joint"""

    @classmethod
    def setUpClass(cls):
        print("Testing floating root dynamics against Pinocchio's freeflyer joint")
        cls.model = pin.buildModelFromUrdf(fixed_root_urdf, pin.JointModelFreeFlyer())
        cls.data = pin.Data(cls.model)
        cls.robot = load_g1(floating_base="euler")
        cls.num_joints = cls.robot.nv
        cls.num_actuated_joints = cls.num_joints - 6
        np.random.seed(0)

    def test_mass_matrix(self):
        return check_mass_matrix(self.model, self.data, self.robot.mass_matrix)

    def test_gravity_vector(self):
        return check_gravity_vector(self.model, self.data, self.robot.gravity_vector)

    def test_nonlinear_effects(self):
        return check_nonlinear_effects(
            self.model,
            self.data,
            self.robot.gravity_vector,
            self.robot.centrifugal_coriolis_vector,
            self.robot.nonlinear_bias,
        )

    def test_kinematics_and_jacobians(self):
        return check_kinematics_and_jacobians(self.robot, self.model, self.data)

    def test_ee(self):
        return check_ee_tfs_and_jacobians(self.robot, self.model, self.data)


class FixedRootDynamicsTest(unittest.TestCase):
    """Test cases to compare my dynamics with a fixed pelvis against Pinocchio"""

    @classmethod
    def setUpClass(cls):
        print("Testing fixed root dynamics against Pinocchio")
        cls.model = pin.buildModelFromUrdf(fixed_root_urdf)
        cls.data = pin.Data(cls.model)
        cls.robot = load_fixed_root_g1()
        cls.num_joints = cls.robot.nv
        np.random.seed(0)

    def test_mass_matrix(self):
        return check_mass_matrix(self.model, self.data, self.robot.mass_matrix)

    def test_gravity_vector(self):
        return check_gravity_vector(self.model, self.data, self.robot.gravity_vector)

    def test_nonlinear_effects(self):
        return check_nonlinear_effects(
            self.model,
            self.data,
            self.robot.gravity_vector,
            self.robot.centrifugal_coriolis_vector,
            self.robot.nonlinear_bias,
        )

    def test_kinematics_and_jacobians(self):
        return check_kinematics_and_jacobians(self.robot, self.model, self.data)

    def test_ee(self):
        return check_ee_tfs_and_jacobians(self.robot, self.model, self.data)


class FloatingRootDynamicsTest(unittest.TestCase):
    """Test cases to compare my dynamics with 6 virtual links against Pinocchio with the same 6 virtual links"""

    @classmethod
    def setUpClass(cls):
        print("Testing floating root dynamics against Pinocchio")
        cls.model = pin.buildModelFromUrdf(floating_root_urdf)
        cls.data = pin.Data(cls.model)
        cls.robot = load_g1(floating_base="euler")
        cls.num_joints = cls.robot.nv
        np.random.seed(0)

    def test_mass_matrix(self):
        return check_mass_matrix(self.model, self.data, self.robot.mass_matrix)

    def test_gravity_vector(self):
        return check_gravity_vector(self.model, self.data, self.robot.gravity_vector)

    def test_nonlinear_effects(self):
        return check_nonlinear_effects(
            self.model,
            self.data,
            self.robot.gravity_vector,
            self.robot.centrifugal_coriolis_vector,
            self.robot.nonlinear_bias,
        )

    def test_kinematics_and_jacobians(self):
        return check_kinematics_and_jacobians(self.robot, self.model, self.data)

    def test_ee(self):
        return check_ee_tfs_and_jacobians(self.robot, self.model, self.data)


if __name__ == "__main__":
    unittest.main()
