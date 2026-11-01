"""Differential IK, retargeting, and global IK on frax robots

Requires the `ik` extra: `pip install "frax[ik]"`
"""

from frax.ik.constraints import (
    DEFAULT_PENALTY,
    BarrierConstraint,
    EqualityConstraint,
    FloorConstraint,
    InequalityConstraint,
    JointLimits,
    ObstacleConstraint,
    ResidualEquality,
    SelfCollisionConstraint,
    SphereObstacles,
)
from frax.ik.global_ik import GlobalIKResult, GlobalIKSolver, sample_configurations
from frax.ik.parallel import device_vmap
from frax.ik.retargeter import OfflineRetargeter, OnlineRetargeter, Retargeter
from frax.ik.solver import DiffIKSolver, SolverInfo, SolverState
from frax.ik.tasks import (
    EETask,
    FrameTask,
    JointTask,
    ResidualTask,
    Task,
    rotation_error,
)
from frax.ik.utils import load_bvh, resample_poses, save_motion, smooth_poses

__all__ = [
    "DEFAULT_PENALTY",
    "BarrierConstraint",
    "DiffIKSolver",
    "EETask",
    "EqualityConstraint",
    "FloorConstraint",
    "FrameTask",
    "GlobalIKResult",
    "GlobalIKSolver",
    "InequalityConstraint",
    "JointLimits",
    "JointTask",
    "ObstacleConstraint",
    "OfflineRetargeter",
    "OnlineRetargeter",
    "ResidualEquality",
    "ResidualTask",
    "Retargeter",
    "SelfCollisionConstraint",
    "SolverInfo",
    "SolverState",
    "SphereObstacles",
    "Task",
    "device_vmap",
    "load_bvh",
    "resample_poses",
    "rotation_error",
    "sample_configurations",
    "save_motion",
    "smooth_poses",
]
