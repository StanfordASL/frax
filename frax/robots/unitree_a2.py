import numpy as np

from frax.assets import A2_ASSETS_DIR
from frax.core.quadruped import Quadruped
from frax.utils.collision_utils import bubblify_to_mine


urdf = A2_ASSETS_DIR / "a2.urdf"
# TODO: Add a spherized collision model
collision_model_file = A2_ASSETS_DIR / "bubblify/a2_spherized.yml"
root_link_name = "base_link"

joint_to_child_mapping = {
    "FL_hip_joint": "FL_hip",
    "FL_thigh_joint": "FL_thigh",
    "FL_calf_joint": "FL_calf",
    "RL_hip_joint": "RL_hip",
    "RL_thigh_joint": "RL_thigh",
    "RL_calf_joint": "RL_calf",
    "FR_hip_joint": "FR_hip",
    "FR_thigh_joint": "FR_thigh",
    "FR_calf_joint": "FR_calf",
    "RR_hip_joint": "RR_hip",
    "RR_thigh_joint": "RR_thigh",
    "RR_calf_joint": "RR_calf",
}
joint_ordering = tuple(joint_to_child_mapping.keys())
link_ordering = tuple(joint_to_child_mapping.values())

front_left_foot_parent_joint_name = "FL_calf_joint"
front_right_foot_parent_joint_name = "FR_calf_joint"
hind_left_foot_parent_joint_name = "RL_calf_joint"
hind_right_foot_parent_joint_name = "RR_calf_joint"

# Foot offsets are at the center of the spherical foot (the fixed foot joint's origin)
foot_offset = np.block(
    [[np.eye(3), np.array([0.0, 0.0, -0.275]).reshape(-1, 1)], [0.0, 0.0, 0.0, 1.0]]
)

default_q_act = np.array([0., 0.8, -1.5, 0., 0.8, -1.5, 0., 0.8, -1.5, 0., 0.8, -1.5])
default_base_height = 0.434


def load_a2(floating_base: str = "quaternion") -> Quadruped:
    """Load the Unitree A2 quadruped with a free-floating base

    Args:
        floating_base (str, optional): Floating base representation, "euler" or "quaternion".
            See Robot for details. Defaults to "quaternion".
    """
    assert floating_base in ("euler", "quaternion")
    orientation = [1.0, 0.0, 0.0, 0.0] if floating_base == "quaternion" else [0.0, 0.0, 0.0]
    default_q = np.concatenate([[0.0, 0.0, default_base_height], orientation, default_q_act])
    return Quadruped(
        urdf,
        front_left_foot_parent_joint_name,
        front_right_foot_parent_joint_name,
        hind_left_foot_parent_joint_name,
        hind_right_foot_parent_joint_name,
        front_left_foot_offset=foot_offset,
        front_right_foot_offset=foot_offset,
        hind_left_foot_offset=foot_offset,
        hind_right_foot_offset=foot_offset,
        joint_ordering=joint_ordering,
        floating_base=floating_base,
        collision_data=bubblify_to_mine(
            collision_model_file,
            joint_to_child_mapping,
            root_link_name=root_link_name,
            add_floating_base=True,
            sc_data=None, # TODO
            verbose=False,
        ),
        default_configuration=default_q,
    )


def test_a2():
    # Quick validation that the quadruped class works
    print("\nTesting Unitree A2:")
    robot = load_a2()
    q = robot.default_configuration
    qd = 0.1 * np.ones(robot.nv)
    transforms = robot.joint_to_world_transforms(q)
    M = robot._mass_matrix(transforms)
    c = robot._centrifugal_coriolis_vector(qd, transforms)
    g = robot._gravity_vector(transforms)
    p_com = robot._center_of_mass(transforms)
    J_com = robot._center_of_mass_jacobian(transforms)
    J_fl = robot._front_left_foot_jacobian(transforms)
    np.set_printoptions(suppress=True, precision=3, linewidth=300, threshold=1e5)
    print(f"\nMass Matrix:\n{M}")
    print(f"\nCentrifugal/Coriolis Vector:\n{c}")
    print(f"\nGravity Vector:\n{g}")
    print(f"\nAncestor mask: \n{robot.ancestor_mask}")
    print(f"\nCenter of mass position: {p_com}")
    print(f"\nCOM Jacobian: \n{J_com}")
    print(f"\nFront left foot Jacobian: \n{J_fl}")


if __name__ == "__main__":
    test_a2()
