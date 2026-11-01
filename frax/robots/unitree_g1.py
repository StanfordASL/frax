import numpy as np

from frax.assets import G1_ASSETS_DIR
from frax.core.humanoid import Humanoid
from frax.utils.collision_utils import bubblify_to_mine, link_pairs_to_sc_data

floating_root_urdf = G1_ASSETS_DIR / "floating_g1_29dof_rev_1_0.urdf"
fixed_root_urdf = G1_ASSETS_DIR / "g1_29dof_rev_1_0.urdf"
# Left/right symmetric, with the foot spheres resting on the sole
collision_model_file = G1_ASSETS_DIR / "bubblify/g1_29dof_rev_1_0_spherized.yml"
root_link_name = "pelvis"

joint_to_child_mapping = {
    "left_hip_pitch_joint": "left_hip_pitch_link",
    "left_hip_roll_joint": "left_hip_roll_link",
    "left_hip_yaw_joint": "left_hip_yaw_link",
    "left_knee_joint": "left_knee_link",
    "left_ankle_pitch_joint": "left_ankle_pitch_link",
    "left_ankle_roll_joint": "left_ankle_roll_link",
    "right_hip_pitch_joint": "right_hip_pitch_link",
    "right_hip_roll_joint": "right_hip_roll_link",
    "right_hip_yaw_joint": "right_hip_yaw_link",
    "right_knee_joint": "right_knee_link",
    "right_ankle_pitch_joint": "right_ankle_pitch_link",
    "right_ankle_roll_joint": "right_ankle_roll_link",
    "waist_yaw_joint": "waist_yaw_link",
    "waist_roll_joint": "waist_roll_link",
    "waist_pitch_joint": "torso_link",
    "left_shoulder_pitch_joint": "left_shoulder_pitch_link",
    "left_shoulder_roll_joint": "left_shoulder_roll_link",
    "left_shoulder_yaw_joint": "left_shoulder_yaw_link",
    "left_elbow_joint": "left_elbow_link",
    "left_wrist_roll_joint": "left_wrist_roll_link",
    "left_wrist_pitch_joint": "left_wrist_pitch_link",
    "left_wrist_yaw_joint": "left_wrist_yaw_link",
    "right_shoulder_pitch_joint": "right_shoulder_pitch_link",
    "right_shoulder_roll_joint": "right_shoulder_roll_link",
    "right_shoulder_yaw_joint": "right_shoulder_yaw_link",
    "right_elbow_joint": "right_elbow_link",
    "right_wrist_roll_joint": "right_wrist_roll_link",
    "right_wrist_pitch_joint": "right_wrist_pitch_link",
    "right_wrist_yaw_joint": "right_wrist_yaw_link",
}
fixed_root_joint_ordering = tuple(joint_to_child_mapping.keys())
fixed_root_link_ordering = tuple(joint_to_child_mapping.values())

left_hand_parent_joint_name = "left_wrist_yaw_joint"
right_hand_parent_joint_name = "right_wrist_yaw_joint"
left_foot_parent_joint_name = "left_ankle_roll_joint"
right_foot_parent_joint_name = "right_ankle_roll_joint"

# Self-collision is defined by pairs of links: every sphere on one link against
# every sphere on the other, so editing the spheres doesn't break the pairs.
# Pairs are given for the left side, and mirrored to the right. Links that always
# touch (neighbors, the shoulders and the torso, the hips and the pelvis) are left out
_UPPER_ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw")
_FOREARM = ("elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")  # and hand
_THIGH = ("hip_pitch", "hip_roll", "hip_yaw", "knee")
_LEG = ("hip_yaw", "knee", "ankle_roll")  # thigh, shin, foot


def _links(side, names):
    return [f"{side}_{name}_link" for name in names]


def _mirror(link: str) -> str:
    if link.startswith("left_"):
        return "right_" + link[len("left_") :]
    if link.startswith("right_"):
        return "left_" + link[len("right_") :]
    return link


def _symmetric(left_pairs) -> tuple:
    """Link pairs, and their mirror images, without duplicates"""
    return tuple(
        dict.fromkeys(
            tuple(sorted(pair))
            for a, b in left_pairs
            for pair in ((a, b), (_mirror(a), _mirror(b)))
        )
    )


# Upper body: the forearm and hand against the other arm, the torso, and the
# pelvis, and the upper arm against the torso
_upper_body_sc_pairs = (
    [
        (a, b)
        for a in _links("left", _FOREARM)
        for b in _links("right", _UPPER_ARM + _FOREARM)
    ]
    + [(a, b) for a in _links("left", _FOREARM) for b in ("torso_link", "pelvis")]
    + [("left_shoulder_yaw_link", "torso_link")]
)
# Lower body: the forearms and hands against the thighs, the upper arm against its
# own thigh, and the legs against each other
_lower_body_sc_pairs = (
    [(a, b) for a in _links("left", _FOREARM) for b in _links("left", _THIGH)]
    + [(a, b) for a in _links("left", _FOREARM) for b in _links("right", _THIGH)]
    + [("left_shoulder_yaw_link", b) for b in _links("left", ("hip_roll", "hip_yaw"))]
    + [(a, b) for a in _links("left", _LEG) for b in _links("right", _LEG)]
)
self_collision_link_pairs = _symmetric(_upper_body_sc_pairs + _lower_body_sc_pairs)


def g1_sc_data() -> tuple:
    """Self-collision data (sphere pairs) for `bubblify_to_mine`, from
    self_collision_link_pairs and the collision model"""
    return link_pairs_to_sc_data(collision_model_file, self_collision_link_pairs)


# EE offsets
# Note: feet offsets are designed to be the exact midpoint of the four foot contact geoms
# and hand offsets are positioned roughly in the center of the hand
left_hand_offset = np.block(
    [[np.eye(3), np.array([0.1, 0, 0]).reshape(-1, 1)], [0.0, 0.0, 0.0, 1.0]]
)
right_hand_offset = left_hand_offset  # Symmetric
left_foot_offset = np.block(
    [[np.eye(3), np.array([0.035, 0, -0.03]).reshape(-1, 1)], [0.0, 0.0, 0.0, 1.0]]
)
right_foot_offset = left_foot_offset  # Symmetric

# Pelvis height when standing with all joints at zero (from Unitree's MJCF)
default_base_height = 0.793


def load_fixed_root_g1() -> Humanoid:
    return Humanoid(
        fixed_root_urdf,
        left_hand_parent_joint_name,
        right_hand_parent_joint_name,
        left_foot_parent_joint_name,
        right_foot_parent_joint_name,
        left_hand_ee_offset=left_hand_offset,
        right_hand_ee_offset=right_hand_offset,
        left_foot_ee_offset=left_foot_offset,
        right_foot_ee_offset=right_foot_offset,
        floating_base=None,
        joint_ordering=fixed_root_joint_ordering,
        collision_data=bubblify_to_mine(
            collision_model_file,
            joint_to_child_mapping,
            root_link_name=root_link_name,
            add_floating_base=False,
            sc_data=g1_sc_data(),
            verbose=False,
        ),
    )


def load_g1(floating_base: str = "quaternion") -> Humanoid:
    """Load the Unitree G1 humanoid with a free-floating base

    Args:
        floating_base (str, optional): Floating base representation, "euler" or "quaternion".
            See Robot for details. Defaults to "quaternion".
    """
    assert floating_base in ("euler", "quaternion")
    orientation = (
        [1.0, 0.0, 0.0, 0.0] if floating_base == "quaternion" else [0.0, 0.0, 0.0]
    )
    default_q = np.concatenate(
        [
            [0.0, 0.0, default_base_height],
            orientation,
            np.zeros(len(fixed_root_joint_ordering)),
        ]
    )
    return Humanoid(
        fixed_root_urdf,
        left_hand_parent_joint_name,
        right_hand_parent_joint_name,
        left_foot_parent_joint_name,
        right_foot_parent_joint_name,
        left_hand_ee_offset=left_hand_offset,
        right_hand_ee_offset=right_hand_offset,
        left_foot_ee_offset=left_foot_offset,
        right_foot_ee_offset=right_foot_offset,
        joint_ordering=fixed_root_joint_ordering,
        floating_base=floating_base,
        default_configuration=default_q,
        collision_data=bubblify_to_mine(
            collision_model_file,
            joint_to_child_mapping,
            root_link_name=root_link_name,
            add_floating_base=True,
            sc_data=g1_sc_data(),
            verbose=False,
        ),
    )


def test_g1():
    # Quick validation that the humanoid class works
    print("\nTesting Unitree G1:")
    robot = load_g1()
    q = robot.default_configuration
    v = 0.1 * np.ones(robot.nv)
    transforms = robot.joint_to_world_transforms(q)
    M = robot._mass_matrix(transforms)
    c = robot._centrifugal_coriolis_vector(v, transforms)
    g = robot._gravity_vector(transforms)
    p_com = robot._center_of_mass(transforms)
    J_com = robot._center_of_mass_jacobian(transforms)
    J_rh = robot._right_hand_jacobian(transforms)
    coll_pos, coll_rad = robot._link_collision_data(transforms)
    mu_rh = robot._right_hand_manipulability_index(transforms)
    np.set_printoptions(suppress=True, precision=3, linewidth=300, threshold=1e5)
    print(f"\nMass Matrix:\n{M}")
    print(f"\nCentrifugal/Coriolis Vector:\n{c}")
    print(f"\nGravity Vector:\n{g}")
    print(f"\nAncestor mask: \n{robot.ancestor_mask}")
    print(f"\nCenter of mass position: {p_com}")
    print(f"\nCOM Jacobian: \n{J_com}")
    print(f"\nRight hand Jacobian: \n{J_rh}")
    print(f"\nCollision positions: \n{coll_pos}")
    print(f"\nCollision radii: \n{coll_rad}")
    print(f"\nRight hand manipulability index: {mu_rh}")


if __name__ == "__main__":
    test_g1()
