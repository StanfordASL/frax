import numpy as np

from frax.assets import G1_ASSETS_DIR
from frax.core.humanoid import Humanoid
from frax.utils.collision_utils import bubblify_to_mine

floating_root_urdf = G1_ASSETS_DIR / "floating_g1_29dof_rev_1_0.urdf"
fixed_root_urdf = G1_ASSETS_DIR / "g1_29dof_rev_1_0.urdf"
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

# NOTE: This self-collision model is still somewhat approximate,
# but it will cover most cases from a typical teleoperation scenario

# Terms in the collision model where you can replace "left" with "right"
# and vice versa
asymmetric_sc_terms = (
    # LEFT ARM TO RIGHT ARM
    ("left_wrist_yaw_link", 0, "right_wrist_yaw_link", 0, 0.0),
    ("left_wrist_yaw_link", 0, "right_wrist_pitch_link", 0, 0.0),
    ("left_wrist_yaw_link", 0, "right_wrist_roll_link", 0, 0.0),
    ("left_wrist_yaw_link", 0, "right_elbow_link", 0, 0.0),
    ("left_wrist_pitch_link", 0, "right_wrist_pitch_link", 0, 0.0),
    ("left_wrist_pitch_link", 0, "right_wrist_roll_link", 0, 0.0),
    ("left_wrist_pitch_link", 0, "right_elbow_link", 0, 0.0),
    ("left_wrist_roll_link", 0, "right_wrist_roll_link", 0, 0.0),
    ("left_wrist_roll_link", 0, "right_elbow_link", 0, 0.0),
    ("left_elbow_link", 0, "right_elbow_link", 0, 0.0),
    # LEFT ARM TO LEFT LEG
    # Hand to upper leg
    ("left_wrist_yaw_link", 0, "left_hip_pitch_link", 0, 0.0),
    ("left_wrist_yaw_link", 0, "left_hip_roll_link", 0, 0.0),
    ("left_wrist_yaw_link", 0, "left_hip_roll_link", 1, 0.0),
    ("left_wrist_yaw_link", 0, "left_hip_yaw_link", 0, 0.0),
    ("left_wrist_yaw_link", 0, "left_hip_yaw_link", 1, 0.0),
    # Forearm to upper leg
    ("left_wrist_pitch_link", 0, "left_hip_pitch_link", 0, 0.0),
    ("left_wrist_pitch_link", 0, "left_hip_roll_link", 0, 0.0),
    ("left_wrist_pitch_link", 0, "left_hip_roll_link", 1, 0.0),
    ("left_wrist_pitch_link", 0, "left_hip_yaw_link", 0, 0.0),
    ("left_wrist_pitch_link", 0, "left_hip_yaw_link", 1, 0.0),
    ("left_wrist_roll_link", 0, "left_hip_pitch_link", 0, 0.0),
    ("left_wrist_roll_link", 0, "left_hip_roll_link", 0, 0.0),
    ("left_wrist_roll_link", 0, "left_hip_roll_link", 1, 0.0),
    ("left_wrist_roll_link", 0, "left_hip_yaw_link", 0, 0.0),
    ("left_wrist_roll_link", 0, "left_hip_yaw_link", 1, 0.0),
    # LEFT ARM TO RIGHT LEG
    # Hand to upper leg
    ("left_wrist_yaw_link", 0, "right_hip_pitch_link", 0, 0.0),
    ("left_wrist_yaw_link", 0, "right_hip_roll_link", 0, 0.0),
    ("left_wrist_yaw_link", 0, "right_hip_roll_link", 1, 0.0),
    ("left_wrist_yaw_link", 0, "right_hip_yaw_link", 0, 0.0),
    ("left_wrist_yaw_link", 0, "right_hip_yaw_link", 1, 0.0),
    # LEFT ARM TO PELVIS
    ("left_wrist_yaw_link", 0, "pelvis", 0, 0.0),
    ("left_wrist_pitch_link", 0, "pelvis", 0, 0.0),
    ("left_wrist_roll_link", 0, "pelvis", 0, 0.0),
    # LEFT ARM TO TORSO
    # Forearm to torso
    ("left_wrist_yaw_link", 0, "torso_link", 0, 0.0),
    ("left_wrist_yaw_link", 0, "torso_link", 1, 0.0),
    ("left_wrist_yaw_link", 0, "torso_link", 2, 0.0),
    ("left_wrist_pitch_link", 0, "torso_link", 0, 0.0),
    ("left_wrist_pitch_link", 0, "torso_link", 1, 0.0),
    ("left_wrist_pitch_link", 0, "torso_link", 2, 0.0),
    ("left_wrist_roll_link", 0, "torso_link", 0, 0.0),
    ("left_wrist_roll_link", 0, "torso_link", 1, 0.0),
    ("left_wrist_roll_link", 0, "torso_link", 2, 0.0),
    # Elbow to torso
    ("left_shoulder_yaw_link", 0, "torso_link", 2, 0.0),
    # LEFT LEG TO RIGHT LEG
    ("left_ankle_roll_link", 1, "right_knee_link", 0, 0.0),
    ("left_ankle_roll_link", 1, "right_knee_link", 1, 0.0),
    ("left_ankle_roll_link", 1, "right_knee_link", 2, 0.0),
)

# Terms in the collision model where there is a self-collision pair
# between a matching sphere on the left and right sides
symmetric_sc_terms = (
    ("left_wrist_yaw_link", 0, "right_wrist_yaw_link", 0, 0.0),
    ("left_elbow_link", 0, "right_elbow_link", 0, 0.0),
    ("left_ankle_roll_link", 1, "right_ankle_roll_link", 1, 0.0),
    ("left_hip_yaw_link", 0, "right_hip_yaw_link", 0, 0.0),
    ("left_ankle_roll_link", 1, "right_ankle_roll_link", 1, 0.0),
)


# NOTE: this approach assumes that the collision model is symmetric
# As currently constructed, the model for the G1 is symmetric, so this works
def symmetrize(entries):
    """Helper function: 'symmetrizes' the asymmetric collision pairs across the left/right sides"""

    # Swap "left" with "right" and vice versa if indicated in the link name
    # Otherwise, leave as-is (for things like torso/pelvis links)
    def change_side(name):
        if name.startswith("left"):
            name = "right" + name.lstrip("left")
        elif name.startswith("right"):
            name = "left" + name.lstrip("right")
        return name

    # Go through the list of asymmetric collision pairs, swap left <=> right,
    # and merge the existing and newly-constructed pairs
    entries = list(entries)
    new_entries = []
    for entry in entries:
        new_entries.append(entry)
        first_name = change_side(entry[0])
        second_name = change_side(entry[2])
        new_entry = (first_name, entry[1], second_name, entry[3], entry[4])
        new_entries.append(new_entry)
    return tuple(new_entries)


# Final self-collision data
g1_sc_data = tuple(list(symmetrize(asymmetric_sc_terms)) + list(symmetric_sc_terms))


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
            sc_data=g1_sc_data,
            verbose=False,
        ),
    )


def load_g1(floating_base: str = "euler") -> Humanoid:
    """Load the Unitree G1 humanoid with a free-floating base

    Args:
        floating_base (str, optional): Floating base representation, "euler" or "quaternion".
            See Robot for details. Defaults to "euler".
    """
    assert floating_base in ("euler", "quaternion")
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
        collision_data=bubblify_to_mine(
            collision_model_file,
            joint_to_child_mapping,
            root_link_name=root_link_name,
            add_floating_base=True,
            sc_data=g1_sc_data,
            verbose=False,
        ),
    )


def test_g1():
    # Quick validation that the humanoid class works
    print("\nTesting Unitree G1:")
    robot = load_g1()
    q = robot.neutral_configuration()
    qd = 0.1 * np.ones(robot.nv)
    transforms = robot.joint_to_world_transforms(q)
    M = robot._mass_matrix(transforms)
    c = robot._centrifugal_coriolis_vector(qd, transforms)
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
