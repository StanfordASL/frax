"""Quadruped kinematics and dynamics"""

from typing import Tuple, Optional

import jax
from jax import Array
from jax.typing import ArrayLike
import numpy as np

from frax.core.robot import Robot


@jax.tree_util.register_static
class Quadruped(Robot):
    """Quadruped kinematics and dynamics

    Args:
        urdf_filename (str): Path to the URDF file to load
        front_left_foot_parent_joint_name (str): Name of the front left foot's parent joint
        front_right_foot_parent_joint_name (str): Name of the front right foot's parent joint
        hind_left_foot_parent_joint_name (str): Name of the hind left foot's parent joint
        hind_right_foot_parent_joint_name (str): Name of the hind right foot's parent joint
        front_left_foot_offset (Optional[ArrayLike]): Transformation matrix specifying the front
            left foot offset from the parent joint frame. Defaults to None.
        front_right_foot_offset (Optional[ArrayLike]): Transformation matrix specifying the front
            right foot offset from the parent joint frame. Defaults to None.
        hind_left_foot_offset (Optional[ArrayLike]): Transformation matrix specifying the hind
            left foot offset from the parent joint frame. Defaults to None.
        hind_right_foot_offset (Optional[ArrayLike]): Transformation matrix specifying the hind
            right foot offset from the parent joint frame. Defaults to None.
        collision_data (Optional[dict]): Collision information. Contains info
            on body/root collision data, as well as self-collision pairs.
            See collision_utils for more detail. Defaults to None.
        joint_ordering (Optional[list[str]]): A specific joint ordering to use.
            Defaults to None (infer ordering from URDF)
        floating_base (Optional[str]): How to model the free-floating base: None (fixed base),
            "quaternion", or "euler". See Robot for details. Defaults to "quaternion".
        default_configuration (Optional[ArrayLike]): The default configuration, shape (nq,).
            See Robot for details. Defaults to None (identity base pose, all joints at zero)
    """

    def __init__(
        self,
        urdf_filename,
        front_left_foot_parent_joint_name: str,
        front_right_foot_parent_joint_name: str,
        hind_left_foot_parent_joint_name: str,
        hind_right_foot_parent_joint_name: str,
        front_left_foot_offset: Optional[ArrayLike] = None,
        front_right_foot_offset: Optional[ArrayLike] = None,
        hind_left_foot_offset: Optional[ArrayLike] = None,
        hind_right_foot_offset: Optional[ArrayLike] = None,
        collision_data: Optional[dict] = None,
        joint_ordering: Optional[list[str]] = None,
        floating_base: Optional[str] = "quaternion",
        default_configuration: Optional[ArrayLike] = None,
    ):
        super().__init__(
            urdf_filename,
            collision_data,
            joint_ordering,
            floating_base,
            default_configuration,
        )

        self.front_left_foot_parent_chain = np.flatnonzero(
            self.ancestor_mask[
                self.joint_name_to_index[front_left_foot_parent_joint_name]
            ]
        )
        self.front_right_foot_parent_chain = np.flatnonzero(
            self.ancestor_mask[
                self.joint_name_to_index[front_right_foot_parent_joint_name]
            ]
        )
        self.hind_left_foot_parent_chain = np.flatnonzero(
            self.ancestor_mask[
                self.joint_name_to_index[hind_left_foot_parent_joint_name]
            ]
        )
        self.hind_right_foot_parent_chain = np.flatnonzero(
            self.ancestor_mask[
                self.joint_name_to_index[hind_right_foot_parent_joint_name]
            ]
        )

        if front_left_foot_offset is None:
            self.front_left_foot_offset = np.eye(4)
        else:
            front_left_foot_offset = np.asarray(front_left_foot_offset, dtype=float)
            assert front_left_foot_offset.shape == (4, 4)
            self.front_left_foot_offset = front_left_foot_offset

        if front_right_foot_offset is None:
            self.front_right_foot_offset = np.eye(4)
        else:
            front_right_foot_offset = np.asarray(front_right_foot_offset, dtype=float)
            assert front_right_foot_offset.shape == (4, 4)
            self.front_right_foot_offset = front_right_foot_offset

        if hind_left_foot_offset is None:
            self.hind_left_foot_offset = np.eye(4)
        else:
            hind_left_foot_offset = np.asarray(hind_left_foot_offset, dtype=float)
            assert hind_left_foot_offset.shape == (4, 4)
            self.hind_left_foot_offset = hind_left_foot_offset

        if hind_right_foot_offset is None:
            self.hind_right_foot_offset = np.eye(4)
        else:
            hind_right_foot_offset = np.asarray(hind_right_foot_offset, dtype=float)
            assert hind_right_foot_offset.shape == (4, 4)
            self.hind_right_foot_offset = hind_right_foot_offset

    # FEET TRANSFORMS

    def front_left_foot_transform(self, q: Array) -> Array:
        """Transformation matrix of the front left foot (w.r.t world), shape (4, 4)"""
        transforms = self.joint_to_world_transforms(q)
        return self._front_left_foot_transform(transforms)

    def _front_left_foot_transform(self, joint_transforms: Array) -> Array:
        return self._frame_transform(
            joint_transforms,
            self.front_left_foot_offset,
            self.front_left_foot_parent_chain[-1],
        )

    def front_right_foot_transform(self, q: Array) -> Array:
        """Transformation matrix of the front right foot (w.r.t world), shape (4, 4)"""
        transforms = self.joint_to_world_transforms(q)
        return self._front_right_foot_transform(transforms)

    def _front_right_foot_transform(self, joint_transforms: Array) -> Array:
        return self._frame_transform(
            joint_transforms,
            self.front_right_foot_offset,
            self.front_right_foot_parent_chain[-1],
        )

    def hind_left_foot_transform(self, q: Array) -> Array:
        """Transformation matrix of the hind left foot (w.r.t world), shape (4, 4)"""
        transforms = self.joint_to_world_transforms(q)
        return self._hind_left_foot_transform(transforms)

    def _hind_left_foot_transform(self, joint_transforms: Array) -> Array:
        return self._frame_transform(
            joint_transforms,
            self.hind_left_foot_offset,
            self.hind_left_foot_parent_chain[-1],
        )

    def hind_right_foot_transform(self, q: Array) -> Array:
        """Transformation matrix of the hind right foot (w.r.t world), shape (4, 4)"""
        transforms = self.joint_to_world_transforms(q)
        return self._hind_right_foot_transform(transforms)

    def _hind_right_foot_transform(self, joint_transforms: Array) -> Array:
        return self._frame_transform(
            joint_transforms,
            self.hind_right_foot_offset,
            self.hind_right_foot_parent_chain[-1],
        )

    # FEET JACOBIANS

    def front_left_foot_jacobian(self, q: Array) -> Array:
        """Front left foot Jacobian (w.r.t world), [Jv; Jw], shape (6, nv)"""
        transforms = self.joint_to_world_transforms(q)
        return self._front_left_foot_jacobian(transforms)

    def _front_left_foot_jacobian(self, joint_transforms: Array):
        return self._frame_jacobian(
            joint_transforms,
            self.front_left_foot_offset,
            self.front_left_foot_parent_chain,
        )

    def front_left_foot_jacobian_and_derivative(
        self, q: Array, qd: Array
    ) -> Tuple[Array, Array]:
        """Front left foot Jacobian and its time derivative (w.r.t world), both of shape (6, nv)"""
        transforms = self.joint_to_world_transforms(q)
        return self._front_left_foot_jacobian_and_derivative(qd, transforms)

    def _front_left_foot_jacobian_and_derivative(
        self, qd: Array, joint_transforms: Array
    ) -> Tuple[Array, Array]:
        return self._frame_jacobian_and_derivative(
            qd,
            joint_transforms,
            self.front_left_foot_offset,
            self.front_left_foot_parent_chain,
        )

    def front_right_foot_jacobian(self, q: Array) -> Array:
        """Front right foot Jacobian (w.r.t world), [Jv; Jw], shape (6, nv)"""
        transforms = self.joint_to_world_transforms(q)
        return self._front_right_foot_jacobian(transforms)

    def _front_right_foot_jacobian(self, joint_transforms: Array):
        return self._frame_jacobian(
            joint_transforms,
            self.front_right_foot_offset,
            self.front_right_foot_parent_chain,
        )

    def front_right_foot_jacobian_and_derivative(
        self, q: Array, qd: Array
    ) -> Tuple[Array, Array]:
        """Front right foot Jacobian and its time derivative (w.r.t world), both of shape (6, nv)"""
        transforms = self.joint_to_world_transforms(q)
        return self._front_right_foot_jacobian_and_derivative(qd, transforms)

    def _front_right_foot_jacobian_and_derivative(
        self, qd: Array, joint_transforms: Array
    ) -> Tuple[Array, Array]:
        return self._frame_jacobian_and_derivative(
            qd,
            joint_transforms,
            self.front_right_foot_offset,
            self.front_right_foot_parent_chain,
        )

    def hind_left_foot_jacobian(self, q: Array) -> Array:
        """Hind left foot Jacobian (w.r.t world), [Jv; Jw], shape (6, nv)"""
        transforms = self.joint_to_world_transforms(q)
        return self._hind_left_foot_jacobian(transforms)

    def _hind_left_foot_jacobian(self, joint_transforms: Array):
        return self._frame_jacobian(
            joint_transforms,
            self.hind_left_foot_offset,
            self.hind_left_foot_parent_chain,
        )

    def hind_left_foot_jacobian_and_derivative(
        self, q: Array, qd: Array
    ) -> Tuple[Array, Array]:
        """Hind left foot Jacobian and its time derivative (w.r.t world), both of shape (6, nv)"""
        transforms = self.joint_to_world_transforms(q)
        return self._hind_left_foot_jacobian_and_derivative(qd, transforms)

    def _hind_left_foot_jacobian_and_derivative(
        self, qd: Array, joint_transforms: Array
    ) -> Tuple[Array, Array]:
        return self._frame_jacobian_and_derivative(
            qd,
            joint_transforms,
            self.hind_left_foot_offset,
            self.hind_left_foot_parent_chain,
        )

    def hind_right_foot_jacobian(self, q: Array) -> Array:
        """Hind right foot Jacobian (w.r.t world), [Jv; Jw], shape (6, nv)"""
        transforms = self.joint_to_world_transforms(q)
        return self._hind_right_foot_jacobian(transforms)

    def _hind_right_foot_jacobian(self, joint_transforms: Array):
        return self._frame_jacobian(
            joint_transforms,
            self.hind_right_foot_offset,
            self.hind_right_foot_parent_chain,
        )

    def hind_right_foot_jacobian_and_derivative(
        self, q: Array, qd: Array
    ) -> Tuple[Array, Array]:
        """Hind right foot Jacobian and its time derivative (w.r.t world), both of shape (6, nv)"""
        transforms = self.joint_to_world_transforms(q)
        return self._hind_right_foot_jacobian_and_derivative(qd, transforms)

    def _hind_right_foot_jacobian_and_derivative(
        self, qd: Array, joint_transforms: Array
    ) -> Tuple[Array, Array]:
        return self._frame_jacobian_and_derivative(
            qd,
            joint_transforms,
            self.hind_right_foot_offset,
            self.hind_right_foot_parent_chain,
        )
