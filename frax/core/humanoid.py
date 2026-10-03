"""Humanoid kinematics and dynamics"""

from typing import Tuple, Optional

import jax
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np

from frax.core.robot import Robot


@jax.tree_util.register_static
class Humanoid(Robot):
    """Humanoid kinematics and dynamics

    Args:
        urdf_filename (str): Path to the URDF file to load
        left_hand_parent_joint_name (str): Name of the left hand EE's parent joint
        right_hand_parent_joint_name (str): Name of the right hand EE's parent joint
        left_foot_parent_joint_name (str): Name of the left foot EE's parent joint
        right_foot_parent_joint_name (str): Name of the right foot EE's parent joint
        left_hand_ee_offset (Optional[ArrayLike]): Transformation matrix specifying the left hand EE
            offset from the parent joint frame. Defaults to None.
        right_hand_ee_offset (Optional[ArrayLike]): Transformation matrix specifying the right hand EE
            offset from the parent joint frame. Defaults to None.
        left_foot_ee_offset (Optional[ArrayLike]): Transformation matrix specifying the left foot EE
            offset from the parent joint frame. Defaults to None.
        right_foot_ee_offset (Optional[ArrayLike]): Transformation matrix specifying the right foot EE
            offset from the parent joint frame. Defaults to None.
        collision_data (Optional[dict]): Collision information. Contains info
            on body/root collision data, as well as self-collision pairs.
            See collision_utils for more detail. Defaults to None.
        joint_ordering (Optional[list[str]]): A specific joint ordering to use.
            Defaults to None (infer ordering from URDF)
        floating_base (Optional[str]): How to model the free-floating base: None (fixed base),
            "quaternion", or "euler". See Robot for details. Defaults to "quaternion".
    """

    def __init__(
        self,
        urdf_filename,
        left_hand_parent_joint_name: str,
        right_hand_parent_joint_name: str,
        left_foot_parent_joint_name: str,
        right_foot_parent_joint_name: str,
        left_hand_ee_offset: Optional[ArrayLike] = None,
        right_hand_ee_offset: Optional[ArrayLike] = None,
        left_foot_ee_offset: Optional[ArrayLike] = None,
        right_foot_ee_offset: Optional[ArrayLike] = None,
        collision_data: Optional[dict] = None,
        joint_ordering: Optional[list[str]] = None,
        floating_base: Optional[str] = "quaternion",
    ):
        super().__init__(
            urdf_filename, collision_data, joint_ordering, floating_base
        )

        self.left_hand_parent_chain = np.flatnonzero(
            self.ancestor_mask[self.joint_name_to_index[left_hand_parent_joint_name]]
        )
        self.right_hand_parent_chain = np.flatnonzero(
            self.ancestor_mask[self.joint_name_to_index[right_hand_parent_joint_name]]
        )
        self.left_foot_parent_chain = np.flatnonzero(
            self.ancestor_mask[self.joint_name_to_index[left_foot_parent_joint_name]]
        )
        self.right_foot_parent_chain = np.flatnonzero(
            self.ancestor_mask[self.joint_name_to_index[right_foot_parent_joint_name]]
        )

        if left_hand_ee_offset is None:
            self.left_hand_ee_offset = np.eye(4)
        else:
            left_hand_ee_offset = np.asarray(left_hand_ee_offset, dtype=float)
            assert left_hand_ee_offset.shape == (4, 4)
            self.left_hand_ee_offset = left_hand_ee_offset

        if right_hand_ee_offset is None:
            self.right_hand_ee_offset = np.eye(4)
        else:
            right_hand_ee_offset = np.asarray(right_hand_ee_offset, dtype=float)
            assert right_hand_ee_offset.shape == (4, 4)
            self.right_hand_ee_offset = right_hand_ee_offset

        if left_foot_ee_offset is None:
            self.left_foot_ee_offset = np.eye(4)
        else:
            left_foot_ee_offset = np.asarray(left_foot_ee_offset, dtype=float)
            assert left_foot_ee_offset.shape == (4, 4)
            self.left_foot_ee_offset = left_foot_ee_offset

        if right_foot_ee_offset is None:
            self.right_foot_ee_offset = np.eye(4)
        else:
            right_foot_ee_offset = np.asarray(right_foot_ee_offset, dtype=float)
            assert right_foot_ee_offset.shape == (4, 4)
            self.right_foot_ee_offset = right_foot_ee_offset

    # HANDS AND FEET TRANSFORMS

    def left_hand_transform(self, q: Array) -> Array:
        """Transformation matrix of the left hand (w.r.t world), shape (4, 4)"""
        transforms = self.joint_to_world_transforms(q)
        return self._left_hand_transform(transforms)

    def _left_hand_transform(self, joint_transforms: Array) -> Array:
        return self._frame_transform(
            joint_transforms, self.left_hand_ee_offset, self.left_hand_parent_chain[-1]
        )

    def right_hand_transform(self, q: Array) -> Array:
        """Transformation matrix of the right hand (w.r.t world), shape (4, 4)"""
        transforms = self.joint_to_world_transforms(q)
        return self._right_hand_transform(transforms)

    def _right_hand_transform(self, joint_transforms: Array) -> Array:
        return self._frame_transform(
            joint_transforms,
            self.right_hand_ee_offset,
            self.right_hand_parent_chain[-1],
        )

    def left_foot_transform(self, q: Array) -> Array:
        """Transformation matrix of the left foot (w.r.t world), shape (4, 4)"""
        transforms = self.joint_to_world_transforms(q)
        return self._left_foot_transform(transforms)

    def _left_foot_transform(self, joint_transforms: Array) -> Array:
        return self._frame_transform(
            joint_transforms, self.left_foot_ee_offset, self.left_foot_parent_chain[-1]
        )

    def right_foot_transform(self, q: Array) -> Array:
        """Transformation matrix of the right foot (w.r.t world), shape (4, 4)"""
        transforms = self.joint_to_world_transforms(q)
        return self._right_foot_transform(transforms)

    def _right_foot_transform(self, joint_transforms: Array) -> Array:
        return self._frame_transform(
            joint_transforms,
            self.right_foot_ee_offset,
            self.right_foot_parent_chain[-1],
        )

    # HANDS AND FEET JACOBIANS

    def left_hand_jacobian(self, q: Array) -> Array:
        """Left hand Jacobian (w.r.t world), [Jv; Jw], shape (6, nv)"""
        transforms = self.joint_to_world_transforms(q)
        return self._left_hand_jacobian(transforms)

    def _left_hand_jacobian(self, joint_transforms: Array):
        return self._frame_jacobian(
            joint_transforms, self.left_hand_ee_offset, self.left_hand_parent_chain
        )

    def left_hand_jacobian_and_derivative(
        self, q: Array, qd: Array
    ) -> Tuple[Array, Array]:
        """Left hand Jacobian and its time derivative (w.r.t world), both of shape (6, nv)"""
        transforms = self.joint_to_world_transforms(q)
        return self._left_hand_jacobian_and_derivative(qd, transforms)

    def _left_hand_jacobian_and_derivative(
        self, qd: Array, joint_transforms: Array
    ) -> Tuple[Array, Array]:
        return self._frame_jacobian_and_derivative(
            qd, joint_transforms, self.left_hand_ee_offset, self.left_hand_parent_chain
        )

    def right_hand_jacobian(self, q: Array) -> Array:
        """Right hand Jacobian (w.r.t world), [Jv; Jw], shape (6, nv)"""
        transforms = self.joint_to_world_transforms(q)
        return self._right_hand_jacobian(transforms)

    def _right_hand_jacobian(self, joint_transforms: Array):
        return self._frame_jacobian(
            joint_transforms, self.right_hand_ee_offset, self.right_hand_parent_chain
        )

    def right_hand_jacobian_and_derivative(
        self, q: Array, qd: Array
    ) -> Tuple[Array, Array]:
        """Right hand Jacobian and its time derivative (w.r.t world), both of shape (6, nv)"""
        transforms = self.joint_to_world_transforms(q)
        return self._right_hand_jacobian_and_derivative(qd, transforms)

    def _right_hand_jacobian_and_derivative(
        self, qd: Array, joint_transforms: Array
    ) -> Tuple[Array, Array]:
        return self._frame_jacobian_and_derivative(
            qd,
            joint_transforms,
            self.right_hand_ee_offset,
            self.right_hand_parent_chain,
        )

    def left_foot_jacobian(self, q: Array) -> Array:
        """Left foot Jacobian (w.r.t world), [Jv; Jw], shape (6, nv)"""
        transforms = self.joint_to_world_transforms(q)
        return self._left_foot_jacobian(transforms)

    def _left_foot_jacobian(self, joint_transforms: Array):
        return self._frame_jacobian(
            joint_transforms, self.left_foot_ee_offset, self.left_foot_parent_chain
        )

    def left_foot_jacobian_and_derivative(
        self, q: Array, qd: Array
    ) -> Tuple[Array, Array]:
        """Left foot Jacobian and its time derivative (w.r.t world), both of shape (6, nv)"""
        transforms = self.joint_to_world_transforms(q)
        return self._left_foot_jacobian_and_derivative(qd, transforms)

    def _left_foot_jacobian_and_derivative(
        self, qd: Array, joint_transforms: Array
    ) -> Tuple[Array, Array]:
        return self._frame_jacobian_and_derivative(
            qd, joint_transforms, self.left_foot_ee_offset, self.left_foot_parent_chain
        )

    def right_foot_jacobian(self, q: Array) -> Array:
        """Right foot Jacobian (w.r.t world), [Jv; Jw], shape (6, nv)"""
        transforms = self.joint_to_world_transforms(q)
        return self._right_foot_jacobian(transforms)

    def _right_foot_jacobian(self, joint_transforms: Array):
        return self._frame_jacobian(
            joint_transforms, self.right_foot_ee_offset, self.right_foot_parent_chain
        )

    def right_foot_jacobian_and_derivative(
        self, q: Array, qd: Array
    ) -> Tuple[Array, Array]:
        """Right foot Jacobian and its time derivative (w.r.t world), both of shape (6, nv)"""
        transforms = self.joint_to_world_transforms(q)
        return self._right_foot_jacobian_and_derivative(qd, transforms)

    def _right_foot_jacobian_and_derivative(
        self, qd: Array, joint_transforms: Array
    ) -> Tuple[Array, Array]:
        return self._frame_jacobian_and_derivative(
            qd,
            joint_transforms,
            self.right_foot_ee_offset,
            self.right_foot_parent_chain,
        )

    # MANIPULABILITY INDICES
    # NOTE: Defining hand manipulability based on the actuated joints from pelvis -> hand
    # TODO: Decide if this should only be restricted to the arm joints

    def left_hand_manipulability_index(self, q: Array) -> float:
        joint_transforms = self.joint_to_world_transforms(q)
        return self._left_hand_manipulability_index(joint_transforms)

    def _left_hand_manipulability_index(self, joint_transforms: Array) -> float:
        J_full = self._left_hand_jacobian(joint_transforms)
        chain_idxs = self.left_hand_parent_chain
        if self.includes_floating_dof:
            chain_idxs = chain_idxs[6:]
        return self._manipulability_index_helper(J_full, chain_idxs)

    def right_hand_manipulability_index(self, q: Array) -> float:
        joint_transforms = self.joint_to_world_transforms(q)
        return self._right_hand_manipulability_index(joint_transforms)

    def _right_hand_manipulability_index(self, joint_transforms: Array) -> float:
        J_full = self._right_hand_jacobian(joint_transforms)
        chain_idxs = self.right_hand_parent_chain
        if self.includes_floating_dof:
            chain_idxs = chain_idxs[6:]
        return self._manipulability_index_helper(J_full, chain_idxs)

    def left_foot_manipulability_index(self, q: Array) -> float:
        joint_transforms = self.joint_to_world_transforms(q)
        return self._left_foot_manipulability_index(joint_transforms)

    def _left_foot_manipulability_index(self, joint_transforms: Array) -> float:
        J_full = self._left_foot_jacobian(joint_transforms)
        chain_idxs = self.left_foot_parent_chain
        if self.includes_floating_dof:
            chain_idxs = chain_idxs[6:]
        return self._manipulability_index_helper(J_full, chain_idxs)

    def right_foot_manipulability_index(self, q: Array) -> float:
        joint_transforms = self.joint_to_world_transforms(q)
        return self._right_foot_manipulability_index(joint_transforms)

    def _right_foot_manipulability_index(self, joint_transforms: Array) -> float:
        J_full = self._right_foot_jacobian(joint_transforms)
        chain_idxs = self.right_foot_parent_chain
        if self.includes_floating_dof:
            chain_idxs = chain_idxs[6:]
        return self._manipulability_index_helper(J_full, chain_idxs)
