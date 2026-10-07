"""Robot kinematics and dynamics"""

# Assorted TODOs:
# - Some logic in the jacobian computation (frame, link, joint) is duplicated,
#   this can likely be simplified with some helper functions
# - Also, see if we can use some of the spatial axes computation for the jacobians

from typing import Tuple, Optional
import warnings

import jax
from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from frax.utils.urdf_parser import parse_urdf
from frax.utils.linalg_utils import cholesky_spd_inverse
from frax.utils.transform_utils import (
    create_transform_numpy,
    transform_points,
    joint_transform,
)
from frax.utils.spatial_utils import (
    get_spatial_inertias,
    get_spatial_joint_axes,
    spatial_motion_cross,
    spatial_force_cross,
)
from frax.utils.rotation_utils import (
    quat_wxyz_to_rmat,
    quat_wxyz_multiply,
    quat_wxyz_conjugate,
    quat_wxyz_exp,
    quat_wxyz_log,
)

FLOATING_BASE_TYPES = (None, "quaternion", "euler")


@jax.tree_util.register_static
class Robot:
    """Robot kinematics and dynamics

    Args:
        urdf_filename (str): Path to the URDF file to load
        collision_data (Optional[dict]): Collision information. Contains info
            on body/root collision data, as well as self-collision pairs.
            See collision_utils for more detail. Defaults to None.
        joint_ordering (Optional[list[str]]): A specific joint ordering to use.
            Defaults to None (infer ordering from URDF)
        floating_base (Optional[str]): How to model a free-floating base. Defaults to None
            (fixed-base), or "quaternion"/"euler" depending on the desired rotation
            representation. Quaternion matches MuJoCo's conventions where
            q = [position, wxyz quaternion, actuated joints] and
            v = [linear velocity, body angular velocity, actuated joint velocities].
            Euler uses 6 virtual joints (3 prismatic, 3 revolute), i.e. intrinsic XYZ
            Euler angles for the rotation. With Euler, nq = nv and qdot = v, but you have
            the gimbal lock issue inherent to Euler angles
        default_configuration (Optional[ArrayLike]): The default configuration, shape (nq),
            Defaults to None (identity floating base, if used, and all actuated joints at zero)
    """

    def __init__(
        self,
        urdf_filename: str,
        collision_data: Optional[dict] = None,
        joint_ordering: Optional[list[str]] = None,
        floating_base: Optional[str] = None,
        default_configuration: Optional[ArrayLike] = None,
    ):
        if floating_base not in FLOATING_BASE_TYPES:
            raise ValueError(
                f"Invalid floating_base: {floating_base}. Options: {FLOATING_BASE_TYPES}"
            )
        data = parse_urdf(
            urdf_filename,
            joint_ordering=joint_ordering,
            add_floating_base=floating_base is not None,
        )

        assert isinstance(collision_data, dict) or collision_data is None
        if isinstance(collision_data, dict):
            collision_positions = collision_data["positions"]
            collision_radii = collision_data["radii"]
            root_collision_positions = collision_data["root_positions"]
            root_collision_radii = collision_data["root_radii"]
            root_sc_pairs = collision_data["root_sc_pairs"]
            root_sc_tols = collision_data["root_sc_tols"]
            body_sc_pairs = collision_data["body_sc_pairs"]
            body_sc_tols = collision_data["body_sc_tols"]
        else:
            collision_positions = ()
            collision_radii = ()
            root_collision_positions = ()
            root_collision_radii = ()
            root_sc_pairs = ()
            root_sc_tols = ()
            body_sc_pairs = ()
            body_sc_tols = ()
        # fmt: off
        self.nv = data["num_joints"]
        self.joint_types = np.asarray(data["joint_types"], dtype=int)
        self.joint_names = data["joint_names"]
        self.actuated_joint_lower_limits = np.asarray(data["actuated_joint_lower_limits"], dtype=float)
        self.actuated_joint_upper_limits = np.asarray(data["actuated_joint_upper_limits"], dtype=float)
        self.actuated_joint_max_forces = np.asarray(data["actuated_joint_max_forces"], dtype=float)
        self.actuated_joint_max_velocities = np.asarray(data["actuated_joint_max_velocities"], dtype=float)
        self.joint_axes = np.asarray(data["joint_axes"], dtype=float)
        self.joint_parent_frame_positions = np.asarray(data["joint_parent_frame_positions"], dtype=float)
        self.joint_parent_frame_rotations = np.asarray(data["joint_parent_frame_rotations"], dtype=float)
        self.link_masses = np.asarray(data["link_masses"], dtype=float)
        self.link_local_inertias = np.asarray(data["link_local_inertias"], dtype=float)
        self.link_local_inertia_positions = np.asarray(data["link_local_inertia_positions"], dtype=float)
        self.link_local_inertia_rotations = np.asarray(data["link_local_inertia_rotations"], dtype=float)
        self.parent_idxs = np.asarray(data["parent_idxs"], dtype=int)
        self.floating_base = floating_base
        self.includes_floating_dof = floating_base is not None  # TODO rename this
        self.collision_positions = collision_positions # RAGGED
        self.collision_radii = collision_radii # RAGGED
        self.root_collision_positions = np.asarray(root_collision_positions, dtype=float)
        self.root_collision_radii = np.asarray(root_collision_radii, dtype=float)
        self.root_sc_pairs = np.asarray(root_sc_pairs, dtype=int)
        self.root_sc_tols = np.asarray(root_sc_tols, dtype=float)
        self.body_sc_pairs = np.asarray(body_sc_pairs, dtype=int)
        self.body_sc_tols = np.asarray(body_sc_tols, dtype=float)
        # fmt: on

        self.is_quaternion_base = floating_base == "quaternion"
        # Dimensions of the floating base's configuration and velocity
        self.nv_floating = 6 if self.includes_floating_dof else 0
        self.nq_floating = 7 if self.is_quaternion_base else self.nv_floating
        self.nq = self.nv + self.nq_floating - self.nv_floating
        self.num_actuated_joints = self.nv - self.nv_floating
        for limits in (
            self.actuated_joint_lower_limits,
            self.actuated_joint_upper_limits,
            self.actuated_joint_max_forces,
            self.actuated_joint_max_velocities,
        ):
            assert len(limits) == self.num_actuated_joints
        if default_configuration is None:
            default_configuration = np.zeros(self.nq)
            if self.is_quaternion_base:
                default_configuration[3] = 1.0
        else:
            default_configuration = np.asarray(default_configuration, dtype=float)
            if default_configuration.shape != (self.nq,):
                raise ValueError(
                    f"Expected default_configuration of shape ({self.nq},), "
                    f"got {default_configuration.shape}"
                )
            q_act = default_configuration[self.nq_floating :]
            if np.any(q_act < self.actuated_joint_lower_limits) or np.any(
                q_act > self.actuated_joint_upper_limits
            ):
                raise ValueError(
                    "default_configuration's actuated joints must be within the joint limits"
                )
            if self.is_quaternion_base and not np.isclose(
                np.linalg.norm(default_configuration[3:7]), 1.0
            ):
                raise ValueError("default_configuration's quaternion must be unit norm")
        self.default_configuration = default_configuration
        if self.is_quaternion_base:
            v_idxs = [*range(3), *range(6, self.nv)]
            q_idxs = [*range(3), *range(7, self.nq)]
        else:
            v_idxs = q_idxs = range(self.nv)
        self.velocity_to_configuration_index = dict(zip(v_idxs, q_idxs))
        self.has_collision_data = len(collision_positions) > 0
        self.has_root_collision_data = len(root_collision_positions) > 0
        self.has_sc_data = len(body_sc_pairs) > 0
        self.joint_to_prev_joint_tfs = np.asarray(
            [
                create_transform_numpy(rot, trans)
                for rot, trans in zip(
                    self.joint_parent_frame_rotations, self.joint_parent_frame_positions
                )
            ]
        )
        self.link_com_to_prev_joint_tfs = np.asarray(
            [
                create_transform_numpy(rot, trans)
                for rot, trans in zip(
                    self.link_local_inertia_rotations, self.link_local_inertia_positions
                )
            ]
        )
        # Set up padded collision positions for vmapping with static shape
        (
            self.padded_collision_positions,
            self.collision_slice_indices,
        ) = self._process_collision_data(collision_positions, collision_radii)
        self.flat_collision_radii = np.asarray(
            jax.tree_util.tree_flatten(self.collision_radii)[0]
        )

        self.total_mass = np.sum(self.link_masses)
        self.inverse_total_mass = 1.0 / self.total_mass

        # Set up joint type masks
        self.prismatic_mask = np.asarray(self.joint_types, dtype=bool)
        self.revolute_mask = ~self.prismatic_mask

        self.ancestor_mask = self._compute_ancestor_mask()
        self.velocity_mask = self._compute_velocity_mask()
        self.is_pure_kinematic_chain = np.array_equal(
            self.ancestor_mask, np.tril(np.ones((self.nv, self.nv)))
        )

        self.joint_name_to_index = {name: i for i, name in enumerate(self.joint_names)}

    def _process_collision_data(
        self, positions: tuple, radii: tuple
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Helper function: Sets up a padded representation of the collision sphere positions
        for vmapping with uniform shape

        This should only be called once upon initialization

        Args:
            positions (tuple): Collision sphere locations for each link. Tuple of len (num_links)
                where each entry contains a set of points (length 3) for that link
            radii (tuple): Collision sphere radii for each link. Tuple of len (num_links)
                where each entry contains a set of radii for that link

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                padded_positions: Positions, shape (nv, max_spheres_per_link, 3)
                slice_indices: Indices of the *flattened* padded positions to select,
                    corresponding to the non-padded data
        """
        assert isinstance(positions, tuple)
        assert isinstance(radii, tuple)
        if len(positions) == 0 or len(radii) == 0:
            return (), ()
        sphere_counts = tuple(len(rs) for rs in radii)
        max_spheres_per_link = max(sphere_counts)
        padded_positions = np.zeros((self.nv, max_spheres_per_link, 3))
        for link_idx in range(self.nv):
            for sphere_idx in range(sphere_counts[link_idx]):
                padded_positions[link_idx, sphere_idx] = positions[link_idx][sphere_idx]
        # mask: (nv, max_spheres) - True for non-padded spheres
        sphere_mask = (
            np.arange(max_spheres_per_link) < np.asarray(sphere_counts)[:, None]
        )
        slice_indices = np.flatnonzero(sphere_mask.flatten())
        return padded_positions, slice_indices

    # TODO figure out if the dtype of the mask makes much difference to performance...
    # Right now it's float but switching to bool doesn't seem to change much
    def _compute_ancestor_mask(self) -> np.ndarray:
        """Computes the connectivity matrix for the tree structure.

        Returns:
            np.ndarray: Shape (nv, nv). Mask[i, j] = 1 if j is an ancestor of i
        """
        N = self.nv
        mask = np.zeros((N, N), dtype=bool)

        # Based on how we've parsed the URDF, the base (pelvis) link is assigned idx = 0
        # But it's slightly easier to compute this mask if it is assigned -1
        assert np.min(self.parent_idxs) == -1

        # Assumes topological sort (parents appear before children)
        for i in range(N):
            mask[i, i] = True  # A joint affects its own link
            parent = self.parent_idxs[i]
            while parent != -1:
                mask[i, parent] = True
                parent = self.parent_idxs[parent]

        return mask

    def _compute_velocity_mask(self) -> np.ndarray:
        """Computes the mask describing which DOFs contribute to each body's spatial velocity

        Returns:
            np.ndarray: Shape (nv, nv). Mask[i, j] = 1 if DOF j contributes to the velocity of body i
        """
        # Note: This is the same as the ancestor mask, except for a quaternion floating base
        mask = self.ancestor_mask.copy()
        if self.is_quaternion_base:
            # Translational bodies move with the linear DOFs
            mask[0:3, 0:3] = True
            # Rotational bodies move with linear and angular DOFs
            mask[3:6, 0:6] = True
        return mask

    @property
    def num_joints(self) -> int:
        """Deprecated: use `nv` (velocity dimension) or `nq` (configuration dimension)"""
        warnings.warn(
            "Robot.num_joints is deprecated. Use Robot.nv (velocity dimension) "
            + "or Robot.nq (configuration dimension) instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.nv

    def integrate(self, q: Array, v: Array, dt: float) -> Array:
        """Integrates a configuration forward in time with a constant velocity

        Args:
            q (Array): Configuration, shape (nq,)
            v (Array): Velocity, shape (nv,)
            dt (float): Timestep

        Returns:
            Array: New configuration, shape (nq,)
        """
        if not self.is_quaternion_base:
            return q + v * dt
        pos = q[:3] + v[:3] * dt
        quat = quat_wxyz_multiply(q[3:7], quat_wxyz_exp(v[3:6] * dt))
        quat = quat / jnp.linalg.norm(quat)
        q_act = q[7:] + v[6:] * dt
        return jnp.concatenate([pos, quat, q_act])

    def difference(self, q0: Array, q1: Array) -> Array:
        """Computes the velocity which takes q0 to q1 in unit time
        (inverse of integrate)

        Args:
            q0 (Array): Starting configuration, shape (nq,)
            q1 (Array): Ending configuration, shape (nq,)

        Returns:
            Array: Velocity, shape (nv,)
        """
        if not self.is_quaternion_base:
            return q1 - q0
        quat_rel = quat_wxyz_multiply(quat_wxyz_conjugate(q0[3:7]), q1[3:7])
        return jnp.concatenate(
            [q1[:3] - q0[:3], quat_wxyz_log(quat_rel), q1[7:] - q0[7:]]
        )

    def velocity_to_qdot_map(self, q: Array) -> Array:
        """Matrix E(q) mapping velocities to the time derivative of the configuration:
        q_dot = E(q) @ v

        This is useful when combining autodiff w.r.t. q with velocities, e.g.
        dh/dt = (dh/dq) @ E(q) @ v. When nq == nv, this is the identity.

        Args:
            q (Array): Configuration, shape (nq,)

        Returns:
            Array: Velocity map, shape (nq, nv)
        """
        if not self.is_quaternion_base:
            return jnp.eye(self.nv)
        w, x, y, z = q[3:7]
        # q_dot = 0.5 * quat * [0, omega_body] (quaternion multiplication)
        quat_map = 0.5 * jnp.array([[-x, -y, -z], [w, -z, y], [z, w, -x], [-y, x, w]])
        E = jnp.zeros((self.nq, self.nv))
        E = E.at[:3, :3].set(jnp.eye(3))
        E = E.at[3:7, 3:6].set(quat_map)
        E = E.at[7:, 6:].set(jnp.eye(self.num_actuated_joints))
        return E

    def joint_to_world_transforms(self, q: Array) -> Array:
        """Computes the transformation matrices for all joints (Joint frame --> world frame)

        Args:
            q (Array): Configuration vector, shape (nq,)

        Returns:
            Array: Transformation matrices, shape (nv, 4, 4)
        """
        # Note about different FK methods:
        # Jax's associative scan is O(log(N)) complexity whereas just unrolling
        # the loop is O(N). For a pure kinematic chain like a serial manipulator,
        # associative scan should be best. But for a kinematic tree like a humanoid,
        # it may be simpler to unroll the loop and rely on the parent mapping.
        if self.is_pure_kinematic_chain:
            return self._scanned_fk(q)
        return self._unrolled_fk(q)

    def _local_joint_transforms(self, q: Array) -> Array:
        """Helper function: Computes each single-DOF joint's transform in parent frame

        Note: For a quaternion floating base, this only includes the actuated joints
        """
        i_v = self.nv_floating if self.is_quaternion_base else 0
        i_q = self.nq_floating if self.is_quaternion_base else 0
        # Calculate each joint's transformation matrix
        transforms = jax.vmap(joint_transform)(
            q[i_q:], self.joint_axes[i_v:], self.joint_types[i_v:]
        )
        # Multiply the transform by its corresponding link offset
        return self.joint_to_prev_joint_tfs[i_v:] @ transforms

    def _quaternion_base_transforms(self, q: Array) -> Array:
        """Helper function: Computes the world transforms of the 6 virtual bodies of the
        quaternion floating base

        Returns:
            Array: Transformation matrices, shape (6, 4, 4)
        """
        pos = q[:3]
        R = quat_wxyz_to_rmat(q[3:7])
        bottom_row = jnp.array([[0.0, 0.0, 0.0, 1.0]])
        # All positional TFs are at the base pos with identity rotation
        T_pos = jnp.block([[jnp.eye(3), pos[:, None]], [bottom_row]])
        # All rotational TFs are at the base pos and base rot
        T_base = jnp.block([[R, pos[:, None]], [bottom_row]])
        return jnp.stack([T_pos, T_pos, T_pos, T_base, T_base, T_base])

    def _unrolled_fk(self, q: Array) -> Array:
        """Compute the forward kinematics via unrolling the loop over the joints"""
        # Compute local joint transforms in parent frame
        local_tfs = self._local_joint_transforms(q)
        # Unrolled FK loop. Assumes topological sort of parent-child relationship
        world_tfs = jnp.zeros((self.nv, 4, 4))
        start = 0
        if self.is_quaternion_base:
            start = self.nv_floating
            world_tfs = world_tfs.at[:start].set(self._quaternion_base_transforms(q))
        for i in range(start, self.nv):
            parent = self.parent_idxs[i]
            parent_tf = world_tfs[parent] if parent != -1 else jnp.eye(4)
            world_tfs = world_tfs.at[i].set(parent_tf @ local_tfs[i - start])
        return world_tfs

    def _scanned_fk(self, q: Array) -> Array:
        """Compute the forward kinematics via scanning over a pure kinematic chain"""
        assert self.is_pure_kinematic_chain
        # Compute local joint transforms in parent frame
        local_tfs = self._local_joint_transforms(q)
        # Return the cumulative product of the transformations
        world_tfs = jax.lax.associative_scan(
            jnp.matmul, local_tfs, reverse=False, axis=0
        )
        if not self.is_quaternion_base:
            return world_tfs
        base_tfs = self._quaternion_base_transforms(q)
        return jnp.concatenate([base_tfs, base_tfs[-1] @ world_tfs])

    def base_transform(self, q: Array) -> Array:
        """Transformation matrix of the floating base (w.r.t world), shape (4, 4)"""
        if not self.includes_floating_dof:
            return jnp.eye(4)
        joint_transforms = self.joint_to_world_transforms(q)
        return self._base_transform(joint_transforms)

    def _base_transform(self, joint_transforms: Array) -> Array:
        if not self.includes_floating_dof:
            return jnp.eye(4)
        # The final virtual body of the 6DOF chain holds the full pose of the base
        return joint_transforms[self.nv_floating - 1]

    def link_to_world_transforms(self, q: Array) -> Array:
        """Compute the transformation matrices for all link inertial frames (link inertial frame --> world frame)

        Args:
            q (Array): Configuration vector, shape (nq,)

        Returns:
            Array: Transformation matrices, shape (nv, 4, 4)
        """
        joint_transforms = self.joint_to_world_transforms(q)
        return self._link_to_world_transforms(joint_transforms)

    def _link_to_world_transforms(self, joint_transforms: Array) -> Array:
        """Helper function: Computes link inertial transformation matrices, given joint transforms"""
        # Multiply the transform by its corresponding link offset
        transforms = joint_transforms @ self.link_com_to_prev_joint_tfs
        return transforms

    # Note: if only the COM positions (not rotations) are needed, using this function
    # as opposed to link_to_world_transforms is a bit faster
    def link_com_positions(self, q: Array) -> Array:
        """Compute the positions of all link COMs in world frame

        Args:
            q (Array): Joint angles, shape (nq,)

        Returns:
            Array: Link COM positions in world frame, shape (num_links, 3)
        """
        joint_transforms = self.joint_to_world_transforms(q)
        return self._link_com_positions(joint_transforms)

    def _link_com_positions(self, joint_transforms: Array) -> Array:
        """Helper function: Compute the positions of all link COMs in world frame, given the joint transforms"""
        # Determine the positions of the link COMs in world frame. Shape (nv, 3)
        # Position in world frame = joint-to-world transform x position in joint frame
        homogeneous_pos = jnp.column_stack(
            [self.link_local_inertia_positions, jnp.ones(self.nv)]
        )
        return jnp.einsum("qij,qj->qi", joint_transforms, homogeneous_pos)[:, :3]

    def center_of_mass(self, q: Array) -> Array:
        """Compute the center of mass of the robot, in world frame

        Args:
            q (Array): Configuration vector, shape (nq,)

        Returns:
            Array: Position of the center of mass, shape (3,)
        """
        transforms = self.joint_to_world_transforms(q)
        return self._center_of_mass(transforms)

    def _center_of_mass(self, joint_transforms: Array) -> Array:
        """Helper function: Compute center of mass given joint transforms"""
        link_com_positions = self._link_com_positions(joint_transforms)
        # Inertially averaged position
        return (
            jnp.sum(self.link_masses.reshape(-1, 1) * link_com_positions, axis=0)
            * self.inverse_total_mass
        )

    def center_of_mass_jacobian(self, q: Array) -> Array:
        """Computes the linear Jacobian (Jv) for the motion of the COM

        Args:
            q (Array): Configuration vector, shape (nq,)

        Returns:
            Array: Jv_COM, shape (3, nv)
        """
        joint_transforms = self.joint_to_world_transforms(q)
        return self._center_of_mass_jacobian(joint_transforms)

    def _center_of_mass_jacobian(self, joint_transforms: Array) -> Array:
        """Helper function: Compute center of mass jacobian given joint transforms"""
        link_Jvs = self._link_linear_jacobians(joint_transforms)
        return self._com_jacobian_from_link_jacobians(link_Jvs)

    def _com_jacobian_from_link_jacobians(self, link_Jvs: Array) -> Array:
        """Helper function: Compute center of mass jacobian given link linear jacobians"""
        # Inertially-weighted average of the link COM jacobians
        return (
            jnp.einsum("l,ldj->dj", self.link_masses, link_Jvs)
            * self.inverse_total_mass
        )

    def _frame_transform(
        self, joint_transforms: Array, frame_transform: Array, parent_index: int
    ) -> Array:
        """Computes the transformation matrix (w.r.t world) of a frame
        attached to a link with a specified parent joint

        Args:
            joint_transforms (Array): Transformation matrices for every joint, shape (nv, 4, 4)
            frame_transform (Array): Transformation matrix of interest in its local frame, shape (4, 4)
            parent_index (int): Index of the frame's parent joint

        Returns:
            Array: Transformation matrix, shape (4, 4)
        """
        return joint_transforms[parent_index] @ frame_transform

    def _frame_jacobian(
        self, joint_transforms: Array, frame_transform: Array, parent_chain: Array
    ) -> Array:
        """Computes the jacobian of a frame attached to a link with a specified parent chain

        Args:
            joint_transforms (Array): Transformation matrices for every joint, shape (nv, 4, 4)
            frame_transform (Array): Transformation matrix of interest in its local frame, shape (4, 4)
            parent_chain (Array): Ancestor joint indices of the frame's link

        Returns:
            Array: Jacobian, shape (6, nv). The first 3 rows are the linear Jacobian,
                and the last 3 rows are the angular Jacobian
        """
        # Get transform of the frame w.r.t the root
        frame_to_root_tf = self._frame_transform(
            joint_transforms, frame_transform, parent_chain[-1]
        )
        frame_pos = frame_to_root_tf[:3, 3]

        # Positions of all parent joints in root frame. Shape (nv, 3)
        parent_pos = joint_transforms[parent_chain, :3, 3]

        # Axes of all parent joints in root frame. Shape (nv, 3)
        parent_axes = (
            joint_transforms[parent_chain, :3, :3]
            @ self.joint_axes[parent_chain, :, jnp.newaxis]
        ).squeeze(axis=2)

        # Position of frame, with respect to joint j. Shape (nv, 3).
        frame_wrt_joints = frame_pos[jnp.newaxis, :] - parent_pos

        # Cross products between joint axis j and frame position with respect to joint j.
        # Shape (nv, 3)
        lever_arms = jnp.cross(parent_axes, frame_wrt_joints)

        # Linear jacobian has a prismatic contribution and revolute contribution
        # Prismatic contribution: All prismatic joints' z axes
        # Revolute contribution: Cross product of vector from revolute joints to EE
        Jv = jnp.where(
            self.revolute_mask[parent_chain, None], lever_arms, parent_axes
        ).T
        # Angular jacobian only has a contribution from revolute joints (their axes)
        Jw = jnp.where(
            self.revolute_mask[parent_chain, None],
            parent_axes,
            jnp.zeros_like(parent_axes),
        ).T
        J = jnp.vstack([Jv, Jw])
        # Fast path: if parents are all joints, then we can just return directly
        if len(parent_chain) == self.nv:  # Note: this is static
            return J
        # Otherwise, reconstruct full jacobian from parent computations
        J_full = jnp.zeros((6, self.nv)).at[:, parent_chain].set(J)
        return J_full

    # TODO: See if using more spatial algebra would simplify some of the operations here
    def _frame_jacobian_and_derivative(
        self,
        v: Array,
        joint_transforms: Array,
        frame_transform: Array,  # TODO rename to offset_transform?
        parent_chain: Array,
    ) -> Tuple[Array, Array]:
        """Computes both the jacobian of a frame attached to a link with a specified parent chain,
        and its time derivative

        Frequently, if we need Jdot, we also need J. This function is designed to reduce duplicated
        computations between J and Jdot in that case

        Args:
            v (Array): Generalized velocities, shape (nv,)
            joint_transforms (Array): Transformation matrices for every joint, shape (nv, 4, 4)
            frame_transform (Array): Transformation matrix of interest in its local frame, shape (4, 4)
            parent_chain (Array): Ancestor joint indices of the frame's link

        Returns:
            Tuple[Array, Array]:
                J (Array): Jacobian, shape (6, nv)
                Jdot (Array): Time derivative of the Jacobian, shape (6, nv)
        """
        # TODO: Create a version of _joint_jacobians that allows us to just compute it for the parent chain?
        joint_Jvs, joint_Jws = self._joint_jacobians(joint_transforms)
        parent_vels = joint_Jvs[parent_chain] @ v
        parent_ang_vels = joint_Jws[parent_chain] @ v

        # NOTE: Many of these operations below are similar to the frame_jacobian function
        # For more documentation, refer to the comments in that function

        parent_axes = (
            joint_transforms[parent_chain, :3, :3]
            @ self.joint_axes[parent_chain, :, jnp.newaxis]
        ).squeeze(axis=2)
        parent_axes_dot = jnp.cross(parent_ang_vels, parent_axes)
        parent_pos = joint_transforms[parent_chain, :3, 3]

        frame_to_root_tf = self._frame_transform(
            joint_transforms, frame_transform, parent_chain[-1]
        )
        frame_pos = frame_to_root_tf[:3, 3]
        frame_pos_wrt_parents = frame_pos[jnp.newaxis, :] - parent_pos
        lever_arms = jnp.cross(parent_axes, frame_pos_wrt_parents)
        Jv = jnp.where(
            self.revolute_mask[parent_chain, None], lever_arms, parent_axes
        ).T
        Jw = jnp.where(
            self.revolute_mask[parent_chain, None],
            parent_axes,
            jnp.zeros_like(parent_axes),
        ).T
        J = jnp.vstack([Jv, Jw])

        frame_vel = Jv @ v[parent_chain]
        frame_vel_wrt_parents = frame_vel[jnp.newaxis, :] - parent_vels

        lever_arms_dot = jnp.cross(parent_axes_dot, frame_pos_wrt_parents) + jnp.cross(
            parent_axes, frame_vel_wrt_parents
        )
        Jv_dot = jnp.where(
            self.revolute_mask[parent_chain, None], lever_arms_dot, parent_axes_dot
        ).T
        Jw_dot = jnp.where(
            self.revolute_mask[parent_chain, None],
            parent_axes_dot,
            jnp.zeros_like(parent_axes_dot),
        ).T
        J_dot = jnp.vstack([Jv_dot, Jw_dot])

        # Fast path: if parents are all joints, then we can just return directly
        if len(parent_chain) == self.nv:  # Note: this is static
            return J, J_dot
        # Otherwise, reconstruct full Jacobian and derivative from parent computations
        J_full = jnp.zeros((6, self.nv)).at[:, parent_chain].set(J)
        J_dot_full = jnp.zeros((6, self.nv)).at[:, parent_chain].set(J_dot)
        return J_full, J_dot_full

    def _manipulability_index_helper(self, J_full: Array, chain_idxs: Array) -> float:
        """Helper function to compute the manipulability indices for all hands and feet"""
        J_reduced = J_full[:, chain_idxs]
        sigmas = jax.lax.linalg.svd(J_reduced, compute_uv=False)
        return jnp.prod(sigmas)

    def link_collision_data(self, q: Array) -> Tuple[Array, Array]:
        """Compute collision data for all links given the joint configuration

        Args:
            q (Array): Configuration vector, shape (nq,)

        Returns:
            Tuple[Array, Array]:
                positions (Array): Positions of the collision spheres in world frame,
                    shape (num_collision_spheres, 3)
                radii (Array): Radii of the collision spheres, shape (num_collision_spheres,)
        """
        if not self.has_collision_data:
            return jnp.array([]), jnp.array([])
        joint_transforms = self.joint_to_world_transforms(q)
        return self._link_collision_data(joint_transforms)

    def _link_collision_data(self, joint_transforms: Array) -> Tuple[Array, Array]:
        """Helper function: Compute the collision data for all links given the joint transforms"""
        positions = self._link_collision_positions(joint_transforms)
        radii = self.flat_collision_radii
        return positions, radii

    def link_collision_positions(self, q: Array) -> Array:
        """Compute the positions of all collision spheres in world frame

        Args:
            q (Array): Configuration vector, shape (nq,)

        Returns:
            Array: Collision positions, shape (num_collision_spheres, 3)
        """
        if not self.has_collision_data:
            return jnp.array([])
        joint_transforms = self.joint_to_world_transforms(q)
        return self._link_collision_positions(joint_transforms)

    def _link_collision_positions(self, joint_transforms: Array) -> Array:
        """Helper function: Compute all collision positions given joint transforms"""
        # Compute collision body positions in world frame
        # Shape (nv, max_spheres, 3)
        transformed_pts_padded = jax.vmap(transform_points)(
            joint_transforms, self.padded_collision_positions
        )
        # Flatten and select only the non-padded collision data
        # Flat points shape (nv * max_spheres, 3)
        all_pts_flat = transformed_pts_padded.reshape(-1, 3)
        pts_unpadded = all_pts_flat[self.collision_slice_indices]
        return pts_unpadded

    def self_collision_distances(self, q: Array) -> Array:
        if not self.has_collision_data or not self.has_sc_data:
            return jnp.array([])
        joint_transforms = self.joint_to_world_transforms(q)
        return self._self_collision_distances(joint_transforms)

    def _self_collision_distances(self, joint_transforms: Array) -> Array:
        positions, radii = self._link_collision_data(joint_transforms)
        return self._self_collision_distances_from_link_data(positions, radii)

    def _self_collision_distances_from_link_data(
        self, positions: Array, radii: Array
    ) -> Array:
        # Compute distances between spheres on different links of the body
        # Note: just use the spheres of the full-body collision model associated
        # with the self-collision model (typically a subset)
        pairs = self.body_sc_pairs
        idxs_a = pairs[:, 0]
        idxs_b = pairs[:, 1]
        pos_a = positions[idxs_a]
        pos_b = positions[idxs_b]
        rad_a = radii[idxs_a]
        rad_b = radii[idxs_b]
        center_deltas = pos_b - pos_a
        tols = self.body_sc_tols
        body_to_body_dists = (
            jnp.linalg.norm(center_deltas, axis=-1) - rad_a - rad_b - tols
        )
        if not self.has_root_collision_data:
            return body_to_body_dists
        # Compute distances to the fixed-to-world root
        # (note that these spheres on the root do not require FK)
        root_sc_pairs = self.root_sc_pairs
        root_idxs = root_sc_pairs[:, 0]
        body_idxs = root_sc_pairs[:, 1]
        root_pos = self.root_collision_positions[root_idxs]
        body_pos = positions[body_idxs]
        root_rad = self.root_collision_radii[root_idxs]
        body_rad = radii[body_idxs]
        root_center_deltas = body_pos - root_pos
        root_tols = self.root_sc_tols
        root_to_body_dists = (
            jnp.linalg.norm(root_center_deltas, axis=-1)
            - root_rad
            - body_rad
            - root_tols
        )
        return jnp.concatenate([body_to_body_dists, root_to_body_dists])

    def _joint_jacobians(self, joint_transforms: Array) -> Tuple[Array, Array]:
        """Helper function: Compute an array containing the linear (Jv) and angular (Jw)
        jacobians for every joint origin (w.r.t the world)

        Args:
            joint_transforms (Array): Transformation matrices for every joint, shape (nv, 4, 4)

        Returns:
            Tuple[Array, Array]:
                Jv_joints (Array): Linear jacobians for every joint, shape (nv, 3, nv)
                Jw_joints (Array): Angular jacobians for every joint, shape (nv, 3, nv)
        """
        # Positions of all joints in world frame. Shape (nv, 3)
        joint_pos = joint_transforms[:, :3, 3]

        # Axes of all joints in world frame. Shape (nv, 3)
        joint_axes_world_frame = jnp.einsum(
            "qij,qj->qi", joint_transforms[:, :3, :3], self.joint_axes
        )

        # Positions of joint origin i, with respect to joint j. Shape (nv, nv, 3).
        joint_origin_wrt_joints = (
            joint_pos[:, jnp.newaxis, :] - joint_pos[jnp.newaxis, :, :]
        )

        # Cross products between joint axis j and joint origin i's position with respect to joint j.
        # Shape (nv, nv, 3)
        lever_arms = jnp.cross(joint_axes_world_frame, joint_origin_wrt_joints)

        # The velocity mask zeros out the contributions from any joint that is not an ancestor
        # of the joint of interest
        Jv_joints = self.velocity_mask[:, None, :] * jnp.where(
            self.revolute_mask[:, None], lever_arms, joint_axes_world_frame[None, :]
        ).transpose(0, 2, 1)

        # Angular jacobian only has a contribution from revolute joints (their axes)
        Jw_joints = self.velocity_mask[:, None, :] * jnp.where(
            self.revolute_mask[:, None],
            joint_axes_world_frame[None, :],
            jnp.zeros_like(joint_axes_world_frame[None, :]),
        ).transpose(0, 2, 1)
        return Jv_joints, Jw_joints

    def _link_jacobians_from_joint_jacobians(
        self, joint_transforms: Array, joint_Jvs: Array, joint_Jws: Array
    ) -> Tuple[Array, Array]:
        """Helper function: Compute the linear and angular jacobians for every link,
        using the precomputed transforms and jacobians for the joints

        Args:
            joint_transforms (Array): Transformation matrices for every joint, shape (nv, 4, 4)
            joint_Jvs (Array): Linear jacobians for every joint, shape (nv, 3, nv)
            joint_Jws (Array): Angular jacobians for every joint, shape (nv, 3, nv)

        Returns:
            Tuple[Array, Array]:
                link_Jvs (Array): Linear jacobians for every link, shape (num_links, 3, nv)
                link_Jws (Array): Angular jacobians for every link, shape (num_links, 3, nv)
        """
        # Determine the positions of the link COMs in world frame. Shape (nv, 3)
        link_com_pos = self._link_com_positions(joint_transforms)

        # Positions of all joints in world frame. Shape (nv, 3)
        joint_pos = joint_transforms[:, :3, 3]

        # Shift the linear jacobians from the joint origin to the link COM
        # Jv_com = Jv_joint + Jw x (p_com - p_joint)
        # TODO: this is a bit of an array broadcasting mess right now
        r = link_com_pos - joint_pos
        link_Jvs = joint_Jvs + jnp.cross(
            joint_Jws.transpose(0, 2, 1), r[:, jnp.newaxis, :]
        ).transpose(0, 2, 1)

        # Angular vel is the same for all points on a link, so the link_Jws = joint_Jws
        return link_Jvs, joint_Jws

    def _link_linear_jacobians(self, joint_transforms: Array) -> Array:
        """Helper function: Compute an array containing the linear jacobians Jv for every link

        Args:
            joint_transforms (Array): Transformation matrices for every joint, shape (nv, 4, 4)

        Returns:
            Array: Linear jacobians for every link, shape (num_links, 3, nv)
        """
        # Determine the positions of the link COMs in world frame. Shape (nv, 3)
        link_com_pos = self._link_com_positions(joint_transforms)

        # Positions of all joints in world frame. Shape (nv, 3)
        joint_pos = joint_transforms[:, :3, 3]

        # Axes of all joints in world frame. Shape (nv, 3)
        joint_axes_world_frame = jnp.einsum(
            "qij,qj->qi", joint_transforms[:, :3, :3], self.joint_axes
        )

        # Positions of link COM i, with respect to joint j. Shape (nv, nv, 3).
        link_com_wrt_joints = (
            link_com_pos[:, jnp.newaxis, :] - joint_pos[jnp.newaxis, :, :]
        )

        # Cross products between joint axis j and link COM i's position with respect to joint j.
        # Shape (nv, nv, 3)
        lever_arms = jnp.cross(joint_axes_world_frame, link_com_wrt_joints)

        # The velocity mask zeros out the contributions from any joint that is not an ancestor
        # of the link of interest
        return self.velocity_mask[:, None, :] * jnp.where(
            self.revolute_mask[:, None], lever_arms, joint_axes_world_frame[None, :]
        ).transpose(0, 2, 1)

    def _link_angular_jacobians(self, joint_transforms: Array) -> Array:
        """Helper function: Compute an array containing the angular jacobians Jw for every link

        Args:
            transforms (Array): Transformation matrices for every joint, shape (nv, 4, 4)

        Returns:
            Array: Angular jacobians for every link, shape (num_links, 3, nv)
        """
        # Axes of all joints in world frame. Shape (nv, 3)
        joint_axes_world_frame = jnp.einsum(
            "qij,qj->qi", joint_transforms[:, :3, :3], self.joint_axes
        )
        # Apply the revolute mask to the world-frame joint axes (only revolute joints contribute
        # to the angular jacobian) and then mask out the non-ancestor joints
        return jnp.einsum(
            "lj,j,jd->ldj",
            self.velocity_mask,
            self.revolute_mask,
            joint_axes_world_frame,
        )

    def mass_matrix(self, q: Array) -> Array:
        """Compute the mass matrix for a given joint configuration

        Args:
            q (Array): Array of joint angles, shape (nq,)

        Returns:
            Array: The mass matrix, shape (nv, nv)
        """
        joint_transforms = self.joint_to_world_transforms(q)
        return self._mass_matrix(joint_transforms)

    def _mass_matrix(self, joint_transforms: Array) -> Array:
        """Helper function: Compute mass matrix given joint transforms"""
        spatial_axes, spatial_inertias = self._spatial_axes_and_inertias(
            joint_transforms
        )
        return self._crba_from_spatial_data(spatial_axes, spatial_inertias)

    def mass_matrix_inverse(self, M: Array) -> Array:
        """Compute the inverse of the mass matrix

        Args:
            M (Array): Mass matrix, shape (nv, nv)

        Returns:
            Array: Inverse of the mass matrix, shape (nv, nv)
        """
        return cholesky_spd_inverse(M)

    def gravity_vector(self, q: Array) -> Array:
        """Compute the gravity vector for a given joint configuration

        Args:
            q (Array): Array of joint angles, shape (nq,)

        Returns:
            Array: The gravity vector, shape (nv,)
        """
        joint_transforms = self.joint_to_world_transforms(q)
        return self._gravity_vector(joint_transforms)

    def _gravity_vector(self, joint_transforms: Array) -> Array:
        """Helper function: Compute gravity vector given joint transforms"""
        method = "jacobian"  # Options: "jacobian", "rnea"
        if method == "jacobian":
            # Compute the linear jacobians for every link inertial frame
            # and form the gravity vector from these
            link_Jvs = self._link_linear_jacobians(joint_transforms)
            return self._gravity_vector_from_jacobians(link_Jvs)
        else:
            # Use the recursive newton euler algorithm to compute the gravity vector
            g_accel = jnp.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0])
            spatial_axes, spatial_inertias = self._spatial_axes_and_inertias(
                joint_transforms
            )
            return self._rnea_from_spatial_data(
                spatial_axes,
                spatial_inertias,
                v=None,
                a=None,
                gravity_accel=g_accel,
                F_ext=None,
            )

    def _gravity_vector_from_jacobians(self, link_Jvs: Array) -> Array:
        """Helper function: Compute gravity vector given link linear jacobians"""

        # The gravity vector can be computed as follows:
        # G = -1 * sum_{over all links i}(Jvi.T @ (m_i * g_vector))
        # If we know that g_vector only has a z component, we can simplify the computation

        # TODO: make gravity an input? And parse the z-axis assumption automatically

        assume_gravity_acts_only_in_z = True
        if assume_gravity_acts_only_in_z:
            g = -9.81
            mg = g * self.link_masses
            return -mg @ link_Jvs[:, 2, :]
        else:
            g = jnp.array([0.0, 0.0, -9.81])
            return -jnp.einsum("l, ldj, d -> j", self.link_masses, link_Jvs, g)

    def centrifugal_coriolis_vector(self, q: Array, v: Array) -> Array:
        """Compute the centrifugal and coriolis vector for a given joint configuration

        Args:
            q (Array): Array of joint angles, shape (nq,)
            v (Array): Array of Generalized velocities, shape (nv,)

        Returns:
            Array: The centrifugal and coriolis vector, shape (nv,)
        """
        joint_transforms = self.joint_to_world_transforms(q)
        return self._centrifugal_coriolis_vector(v, joint_transforms)

    def _centrifugal_coriolis_vector(self, v: Array, joint_transforms: Array) -> Array:
        """Helper function: Computes the centrifugal/coriolis vector given the joint transforms"""
        spatial_axes, spatial_inertias = self._spatial_axes_and_inertias(
            joint_transforms
        )
        return self._rnea_from_spatial_data(
            spatial_axes, spatial_inertias, v, a=None, gravity_accel=None, F_ext=None
        )

    def nonlinear_bias(self, q: Array, v: Array) -> Array:
        """Compute the nonlinear bias vector (Centrifugal/Coriolis + Gravity) in a single pass
        ```
        b(q, v) = c(q, v) + g(q),
        ```

        Args:
            q (Array): Configuration vector, shape (nq,)
            v (Array): Generalized velocities, shape (nv,)

        Returns:
            Array: The nonlinear bias vector, shape (nv,)
        """
        joint_transforms = self.joint_to_world_transforms(q)
        return self._nonlinear_bias(v, joint_transforms)

    def _nonlinear_bias(self, v: Array, joint_transforms: Array) -> Array:
        """Helper function: Computes the nonlinear bias (c + g) given the joint transforms"""
        g_accel = jnp.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0])
        spatial_axes, spatial_inertias = self._spatial_axes_and_inertias(
            joint_transforms
        )
        return self._rnea_from_spatial_data(
            spatial_axes,
            spatial_inertias,
            v=v,
            a=None,
            gravity_accel=g_accel,
            F_ext=None,
        )

    def rnea(
        self,
        q: Array,
        v: Optional[Array],
        a: Optional[Array],
        gravity_accel: Optional[Array],
        F_ext: Optional[Array],
    ) -> Array:
        """Recursive Newton-Euler Algorithm (vectorized form)

        Args:
            q (Array): Configuration vector, shape (nq,)
            v (Optional[Array]): Generalized velocities, shape (nv,). None if not considering
                joint velocities (as is done to compute gravity)
            a (Optional[Array]): Generalized accelerations, shape (nv,). This is currently not used
                for most methods and can be set to None.
            gravity_accel (Optional[Array]): Spatial acceleration from gravity, shape (6,). None if
                not considering gravity (as is done to compute centrifugal/coriolis)
            F_ext (Optional[Array]): External wrenches on each link (expressed in the root/world frame),
                shape (nv, 6). This is currently not used for most methods and can be set to None.

        Returns:
            Array: Joint torques, shape (nv,)
        """
        joint_transforms = self.joint_to_world_transforms(q)
        spatial_axes, spatial_inertias = self._spatial_axes_and_inertias(
            joint_transforms
        )
        return self._rnea_from_spatial_data(
            spatial_axes, spatial_inertias, v, a, gravity_accel, F_ext
        )

    def _rnea_from_spatial_data(
        self,
        spatial_axes: Array,
        spatial_inertias: Array,
        v: Optional[Array],
        a: Optional[Array],
        gravity_accel: Optional[Array],
        F_ext: Optional[Array],
    ) -> Array:
        """Helper function: Computes RNEA given precomputed spatial axes and inertias"""

        # FORWARD PASS

        spatial_accel = jnp.zeros((self.nv, 6))
        if gravity_accel is not None:
            spatial_accel += gravity_accel[None, :]

        if v is not None:
            s_v = spatial_axes * v[:, None]  # Helper
            # Spatial velocities for every link, summed over contributions from ancestors
            spatial_vel = self.velocity_mask @ s_v
            # Spatial accelerations for every link, summed over contributions from ancestors
            spatial_accel += self.velocity_mask @ spatial_motion_cross(spatial_vel, s_v)
        else:
            spatial_vel = jnp.zeros((self.nv, 6))

        if a is not None:
            spatial_accel += self.velocity_mask @ (spatial_axes * a[:, None])

        # Newton-Euler (part 1): I * a term
        link_forces = jnp.einsum("ijk,ik->ij", spatial_inertias, spatial_accel)

        # Newton-Euler (part 2): v x I * v term
        if v is not None:
            Iv = jnp.einsum("ijk,ik->ij", spatial_inertias, spatial_vel)
            link_forces += spatial_force_cross(spatial_vel, Iv)

        if F_ext is not None:
            link_forces -= F_ext

        # BACKWARD PASS

        # Sum link forces back towards the root based on ancestor relationship
        net_forces = self.ancestor_mask.T @ link_forces

        # Project forces back onto the joint axes to yield the torques
        return jnp.einsum("ij,ij->i", spatial_axes, net_forces)

    def crba(self, q: Array) -> Array:
        """Composite Rigid Body Algorithm (vectorized form)

        Args:
            q (Array): Configuration vector, shape (nq,)

        Returns:
            Array: Mass matrix, shape (nv, nv)
        """
        joint_transforms = self.joint_to_world_transforms(q)
        spatial_axes, spatial_inertias = self._spatial_axes_and_inertias(
            joint_transforms
        )
        return self._crba_from_spatial_data(spatial_axes, spatial_inertias)

    def _crba_from_spatial_data(
        self, spatial_axes: Array, spatial_inertias: Array
    ) -> Array:
        """Helper function: Computes CRBA given precomputed spatial axes and inertias"""

        # BACKWARD PASS

        # Sum inertias torwards the root based on ancestor relationship
        composite_inertias = jnp.einsum(
            "ij,jkl->ikl", self.ancestor_mask.T, spatial_inertias
        )

        # Compute all potential inertial coupling between all pairs of joints...
        M_all = jnp.einsum(
            "ij,ijk,lk->il", spatial_axes, composite_inertias, spatial_axes
        )
        # ... then mask out the terms that don't have a parent/child relationship
        M_lower = self.ancestor_mask * M_all

        # Symmetrize the mass matrix from the lower triangular portion
        return M_lower + jnp.tril(M_lower, k=-1).T

    def _spatial_axes_and_inertias(
        self, joint_transforms: Array
    ) -> Tuple[Array, Array]:
        """Helper function for CRBA and RNEA: Computes the spatial joint axes and link inertias from FK

        Args:
            joint_transforms (Array): Transformation matrices for every joint, shape (nv, 4, 4)

        Returns:
            Tuple[Array, Array]:
                spatial_axes (Array): shape (nv, 6)
                spatial_inertias (Array): shape (nv, 6, 6)
        """
        spatial_axes = get_spatial_joint_axes(
            joint_transforms, self.joint_axes, self.revolute_mask
        )
        link_transforms = self._link_to_world_transforms(joint_transforms)
        spatial_inertias = get_spatial_inertias(
            self.link_masses, self.link_local_inertias, link_transforms
        )
        return spatial_axes, spatial_inertias

    def forward_dynamics(
        self, q: Array, v: Array, tau: Array, fext: Optional[Array]
    ) -> Array:
        """Compute the joint acceleration resulting from an applied torque (and optionally,
        any external forces acting on the links), given the joint state

        Note: gravity is assumed always applied (for now)

        Args:
            q (Array): Configuration vector, shape (nq,)
            v (Array): Generalized velocities, shape (nv,)
            tau (Array): Joint torques, shape (nv,)
            fext (Optional[Array]): External wrenches on each link (expressed in the root/world frame),
                shape (nv, 6). Set to None if no external forces are applied

        Returns:
            Array: Joint accelerations, shape (nv,)
        """
        joint_transforms = self.joint_to_world_transforms(q)
        return self._forward_dynamics(joint_transforms, v, tau, fext)

    def _forward_dynamics(
        self,
        joint_transforms: Array,
        v: Array,
        tau: Array,
        fext: Optional[Array],
    ) -> Array:
        """Helper function: Computes the forward dynamics from FK"""
        # Perform a single evaluation of the spatial axes/inertias
        # and use in both CRBA and RNEA
        spatial_axes, spatial_inertias = self._spatial_axes_and_inertias(
            joint_transforms
        )
        M = self._crba_from_spatial_data(spatial_axes, spatial_inertias)
        g_accel = jnp.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0])
        bias = self._rnea_from_spatial_data(
            spatial_axes,
            spatial_inertias,
            v=v,
            gravity_accel=g_accel,
            a=None,
            F_ext=fext,
        )
        # TODO: Decide if it's better to use a cho_factor + cho_solve combo here
        return jsp.linalg.solve(M, tau - bias, assume_a="pos")
