"""This URDF parser was extracted from the Genesis simulation project
and hackily modified to work with this code

This could use some significant cleanup, but it works for now

Note: I tried a few other different ways of parsing URDFs, for instance,
using pybullet's urdf parser, or urdf_parser_py. This urdfpy-based method
seemed to best handle the frame definitions and merging fixed links.
"""

from typing import Optional, Tuple

import numpy as np
from frax.utils import urdfpy


def parse_urdf(
    filename,
    joint_ordering: Optional[list[str]] = None,
    add_floating_base: bool = False,
):
    filename = str(filename)  # Make sure not Path object
    # Parse the urdf with the same logic as Genesis
    l_infos, j_infos = gs_parse_urdf(filename, merge_fixed=True, fixed_base=True)
    # This will return the links and joints in a breadth-first ordering of the kinematic tree
    # But, if we want an explicit ordering of the joints (depth-first, for instance),
    # we can reorder things
    if joint_ordering is not None:
        l_infos, j_infos = order_by_joint_names(l_infos, j_infos, joint_ordering)

    # Convert the genesis-style formatting to what my Manipulator/Humanoid classes expect
    data = genesis_to_mine(l_infos, j_infos)

    # Add 6DOF corresponding to the free-floating base if desired
    if add_floating_base:
        data = add_floating_base_to_data(data)

    return data


def add_floating_base_to_data(data: dict) -> dict:
    num_new_joints = 6
    new_joint_names = [
        "x_prismatic",
        "y_prismatic",
        "z_prismatic",
        "roll_joint",
        "pitch_joint",
        "yaw_joint",
    ]
    new_joint_types = [1, 1, 1, 0, 0, 0]  # 1 for prismatic, 0 for revolute
    new_joint_axes = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    # Expand joint data
    data["joint_names"] = new_joint_names + data["joint_names"]
    data["joint_types"] = new_joint_types + data["joint_types"]
    data["joint_axes"] = new_joint_axes + data["joint_axes"]

    # Note: The limits only apply to the actuated joints, so the floating base adds none

    data["joint_parent_frame_positions"] = [[0.0, 0.0, 0.0]] * num_new_joints + data[
        "joint_parent_frame_positions"
    ]
    data["joint_parent_frame_rotations"] = [np.eye(3).tolist()] * num_new_joints + data[
        "joint_parent_frame_rotations"
    ]

    # Expand link data
    # The original base link (pelvis) becomes a normal link (the 6th link in our new chain)
    # We add 5 dummy links before it.
    data["link_masses"] = [0.0] * 5 + [data["base_mass"]] + data["link_masses"]
    data["link_local_inertias"] = (
        [np.zeros((3, 3)).tolist()] * 5
        + [data["base_local_inertia"]]
        + data["link_local_inertias"]
    )
    data["link_local_inertia_positions"] = (
        [[0.0, 0.0, 0.0]] * 5
        + [data["base_local_inertia_position"]]
        + data["link_local_inertia_positions"]
    )
    data["link_local_inertia_rotations"] = (
        [np.eye(3).tolist()] * 5
        + [data["base_local_inertia_rotation"]]
        + data["link_local_inertia_rotations"]
    )

    # Update parent indices
    new_parent_idxs = [-1, 0, 1, 2, 3, 4]
    for p in data["parent_idxs"]:
        new_parent_idxs.append(p + 6)
    data.update({"parent_idxs": new_parent_idxs})

    # New base link (world) has 0 mass/inertia
    data["base_mass"] = 0.0
    data["base_local_inertia"] = np.zeros((3, 3)).tolist()
    data["base_local_inertia_position"] = [0.0, 0.0, 0.0]
    data["base_local_inertia_rotation"] = np.eye(3).tolist()

    data["num_joints"] += num_new_joints

    return data


def genesis_to_mine(l_infos, j_infos):
    joint_names = []
    joint_types = []
    joint_lower_limits = []
    joint_upper_limits = []
    joint_max_forces = []
    joint_max_velocities = []
    joint_axes = []
    joint_parent_frame_positions = []
    joint_parent_frame_rotations = []

    link_masses = []
    link_local_inertias = []
    link_local_inertia_positions = []
    link_local_inertia_rotations = []
    parent_idxs = []

    num_joints = len(j_infos) - 1
    num_non_base_links = len(l_infos) - 1

    for i, j_info in enumerate(j_infos):
        if i == 0:
            # Genesis parses their urdfs in a way that the base link is still assigned
            # a joint, even though it's fixed. We don't control this joint, so we'll ignore it
            assert j_info["type"] == "fixed"
            continue
        joint_names.append(j_info["name"])
        if j_info["type"] == "revolute":
            joint_types.append(0)
        elif j_info["type"] == "prismatic":
            joint_types.append(1)
        else:
            # There shouldn't be any fixed joints at this point
            raise ValueError(f"Unknown joint type: {j_info['type']}")

        joint_lower_limits.append(j_info["dofs_limit"][0, 0])
        joint_upper_limits.append(j_info["dofs_limit"][0, 1])
        joint_max_forces.append(j_info["dofs_force_range"][0, 1])
        joint_max_velocities.append(j_info["dofs_velocity_range"][0, 1])
        joint_axes.append(j_info["axis"].tolist())

    for i, l_info in enumerate(l_infos):
        if i == 0:
            # Genesis includes the base link in their list.
            # This can be ignored for a fixed-based serial manipulator,
            # but should NOT be ignored for a humanoid where the base = pelvis
            base_mass = l_info["inertial_mass"]
            base_local_inertia = l_info["inertial_i"].tolist()
            base_local_inertia_position = l_info["inertial_pos"].tolist()
            base_local_inertia_rotation = l_info["inertial_rot"].tolist()
            # Note: the base's pose with respect to the parent joint
            # is not a definable quantity, since there is no parent joint
            continue
        link_masses.append(l_info["inertial_mass"])
        link_local_inertias.append(l_info["inertial_i"].tolist())
        link_local_inertia_positions.append(l_info["inertial_pos"].tolist())
        link_local_inertia_rotations.append(l_info["inertial_rot"].tolist())
        # NOTE: Since this genesis-like parser considers the root (world or pelvis)
        # as link 0, the parent of link 1 (say, a link connected directly to the root)
        # will be index 0. But, this is NOT really what we want when we consider the
        # joint parent mapping, where parent[joint 0] should be -1 (there is no parent
        # joint for a joint connected directly to the root)
        # So, we subtract 1 to adjust for this
        parent_idxs.append(l_info["parent_idx"] - 1)
        # NOTE: The position/rotation of the link's associated joint is
        # stored in the **link** info, not the joint info
        joint_parent_frame_positions.append(l_info["pos"].tolist())
        joint_parent_frame_rotations.append(l_info["rot"].tolist())

    # Quick sanity check
    assert len(joint_types) == num_joints
    assert len(link_masses) == num_non_base_links

    data = {
        "num_joints": num_joints,
        "joint_names": joint_names,
        "joint_types": joint_types,
        "actuated_joint_lower_limits": joint_lower_limits,
        "actuated_joint_upper_limits": joint_upper_limits,
        "actuated_joint_max_forces": joint_max_forces,
        "actuated_joint_max_velocities": joint_max_velocities,
        "joint_axes": joint_axes,
        "joint_parent_frame_positions": joint_parent_frame_positions,
        "joint_parent_frame_rotations": joint_parent_frame_rotations,
        "link_masses": link_masses,
        "link_local_inertias": link_local_inertias,
        "link_local_inertia_positions": link_local_inertia_positions,
        "link_local_inertia_rotations": link_local_inertia_rotations,
        "parent_idxs": parent_idxs,
        "base_mass": base_mass,
        "base_local_inertia": base_local_inertia,
        "base_local_inertia_position": base_local_inertia_position,
        "base_local_inertia_rotation": base_local_inertia_rotation,
    }
    return data


def _order_links(l_infos, j_infos, links_g_info=None):
    # re-order links based on depth in the kinematic tree, so that parent links are always before child links
    n_links = len(l_infos)
    dict_child = {k: [] for k in range(n_links)}
    for lc in range(n_links):
        if "parent_idx" not in l_infos[lc]:
            l_infos[lc]["parent_idx"] = -1
        lp = l_infos[lc]["parent_idx"]
        if lp != -1:
            dict_child[lp].append(lc)
    ordered_links_idx = []
    n_level = 0
    stack_topology = [lc for lc in range(n_links) if l_infos[lc]["parent_idx"] == -1]

    while len(stack_topology) > 0:
        next_stack = []
        ordered_links_idx.append([])
        for link in stack_topology:
            ordered_links_idx[n_level].append(link)
            next_stack += dict_child[link]
        n_level += 1
        stack_topology = next_stack

    if not ordered_links_idx:
        # avoid case with worldbody without any body (geom directly assigned to worldbody)
        return [], [], []

    ordered_links_idx = np.concatenate(ordered_links_idx).tolist()

    for l_info in l_infos:
        if l_info["parent_idx"] >= 0:  # non-base link
            l_info["parent_idx"] = ordered_links_idx.index(l_info["parent_idx"])

    new_l_infos = [l_infos[i] for i in ordered_links_idx]
    new_j_infos = [j_infos[i] for i in ordered_links_idx]

    if links_g_info is not None:
        links_g_info = [links_g_info[i] for i in ordered_links_idx]

    return new_l_infos, new_j_infos, links_g_info


def order_by_joint_names(
    l_infos: list[dict], j_infos: list[dict], target_joint_names: list[str]
) -> Tuple[list[dict], list[dict]]:
    """Reorders link and joint info based on an explicit list of joint names.

    Args:
        l_infos (list[dict]): List of link info dicts
        j_infos (list[dict]): List of joint info dicts
        target_joint_names (list[str]): List defining the desired joint ordering (by joint name)

    Returns:
        tuple: (new_l_infos, new_j_infos)
    """
    assert isinstance(target_joint_names, (list, tuple))
    assert all(isinstance(name, str) for name in target_joint_names)

    # Map Joint Name -> Original Index
    # We assume l_infos[i] corresponds to j_infos[i]
    joint_name_to_idx = {info.get("name"): i for i, info in enumerate(j_infos)}

    target_joint_names_set = set(target_joint_names)

    # NOTE: Genesis includes a joint for the base link
    # Let's handle this explicitly even if it is not included in the joint ordering
    base_joint = j_infos[0]
    if base_joint["name"] not in target_joint_names_set:
        target_joint_names = [base_joint["name"]] + list(target_joint_names)
    else:
        # If the base joint is included in the joint ordering, it must be first
        if not base_joint["name"] == target_joint_names[0]:
            raise Exception("The root joint must be first in the ordering")

    # Make sure that all joints are included in the target list
    for existing_joint_name, index in joint_name_to_idx.items():
        if index != 0 and existing_joint_name not in target_joint_names_set:
            raise Exception(
                f"A joint in the URDF ({existing_joint_name}) is missing in the target ordering!"
            )

    # Make sure that all target joints actually exist
    for name in target_joint_names:
        if name not in joint_name_to_idx:
            raise Exception(f"Target joint {name} not found in the model!")

    # Reorder the joints and links
    ordered_links_idx = [joint_name_to_idx[name] for name in target_joint_names]
    new_l_infos = [l_infos[i] for i in ordered_links_idx]
    new_j_infos = [j_infos[i] for i in ordered_links_idx]

    # Update parent indices to reflect the new order
    old_to_new_map = {
        old_idx: new_idx for new_idx, old_idx in enumerate(ordered_links_idx)
    }
    for l_info in new_l_infos:
        # parent_idx = -1  for root joint/link (pelvis for humanoid)
        if "parent_idx" in l_info and l_info["parent_idx"] != -1:
            old_parent_idx = l_info["parent_idx"]

            if old_parent_idx in old_to_new_map:
                l_info["parent_idx"] = old_to_new_map[old_parent_idx]
            else:
                raise Exception("Somehow, parent index has gotten lost")

    return new_l_infos, new_j_infos


def gs_parse_urdf(filename, merge_fixed=True, links_to_keep=[], fixed_base=True):
    robot = urdfpy.URDF.load(filename)

    # merge links connected by fixed joints
    if merge_fixed:
        robot = merge_fixed_links(robot, links_to_keep)

    link_name_to_idx = dict()
    for idx, link in enumerate(robot.links):
        link_name_to_idx[link.name] = idx

    # Note that each link corresponds to one joint
    n_links = len(robot.links)
    assert n_links == len(robot.joints) + 1
    l_infos = [dict() for _ in range(n_links)]
    j_infos = [dict() for _ in range(n_links)]

    for i in range(n_links):
        link = robot.links[i]
        l_info = l_infos[i]
        l_info["name"] = link.name

        # we compute urdf's invweight later
        # l_info["invweight"] = -1.0

        if link.inertial is None:
            l_info["inertial_pos"] = np.zeros(3)
            l_info["inertial_rot"] = np.eye(3)
            l_info["inertial_i"] = None
            l_info["inertial_mass"] = None

        else:
            l_info["inertial_pos"] = link.inertial.origin[:3, 3]
            l_info["inertial_rot"] = link.inertial.origin[:3, :3]
            l_info["inertial_i"] = link.inertial.inertia
            l_info["inertial_mass"] = link.inertial.mass

        # NOTE: Ignore geoms for now, just interested in kinematics and inertia data
        # l_info["g_infos"] = list()

        # for geom in link.collisions + link.visuals:
        #     geom_is_col = not isinstance(geom, urdfpy.Visual)
        #     if isinstance(geom.geometry.geometry, urdfpy.Mesh):
        #         # One asset (.obj) can contain multiple meshes. Each mesh is one RigidGeom in genesis.
        #         for tmesh in geom.geometry.meshes:
        #             scale = morph.scale
        #             if geom.geometry.geometry.scale is not None:
        #                 scale *= geom.geometry.geometry.scale

        #             mesh = gs.Mesh.from_trimesh(
        #                 tmesh,
        #                 scale=scale,
        #                 convexify=geom_is_col and morph.convexify,
        #                 surface=gs.surfaces.Collision() if geom_is_col else surface,
        #                 metadata={
        #                     "mesh_path": urdfpy.utils.get_filename(
        #                         os.path.dirname(path), geom.geometry.geometry.filename
        #                     )
        #                 },
        #             )

        #             if not geom_is_col and (
        #                 morph.prioritize_urdf_material or not tmesh.visual.defined
        #             ):
        #                 if (
        #                     geom.material is not None
        #                     and geom.material.color is not None
        #                 ):
        #                     mesh.set_color(geom.material.color)

        #             geom_type = gs.GEOM_TYPE.MESH

        #             g_info = {
        #                 "type": geom_type,
        #                 "data": None,
        #                 "pos": geom.origin[:3, 3].copy(),
        #                 "quat": gu.R_to_quat(geom.origin[:3, :3]),
        #                 "mesh": mesh,
        #                 "is_col": geom_is_col,
        #             }
        #             l_info["g_infos"].append(g_info)
        #     else:
        #         # Each geometry primitive is one RigidGeom in genesis.
        #         if isinstance(geom.geometry.geometry, urdfpy.Box):
        #             tmesh = trimesh.creation.box(extents=geom.geometry.geometry.size)
        #             geom_type = gs.GEOM_TYPE.BOX
        #             geom_data = np.array(geom.geometry.geometry.size)

        #         elif isinstance(geom.geometry.geometry, urdfpy.Cylinder):
        #             tmesh = trimesh.creation.cylinder(
        #                 radius=geom.geometry.geometry.radius,
        #                 height=geom.geometry.geometry.length,
        #             )
        #             geom_type = gs.GEOM_TYPE.CYLINDER
        #             geom_data = None

        #         elif isinstance(geom.geometry.geometry, urdfpy.Sphere):
        #             if geom_is_col:
        #                 tmesh = trimesh.creation.icosphere(
        #                     radius=geom.geometry.geometry.radius, subdivisions=2
        #                 )
        #             else:
        #                 tmesh = trimesh.creation.icosphere(
        #                     radius=geom.geometry.geometry.radius
        #                 )
        #             geom_type = gs.GEOM_TYPE.SPHERE
        #             geom_data = np.array([geom.geometry.geometry.radius])

        #         mesh = gs.Mesh.from_trimesh(
        #             tmesh,
        #             scale=morph.scale,
        #             surface=gs.surfaces.Collision() if geom_is_col else surface,
        #             convexify=True,
        #         )

        #         if not geom_is_col:
        #             if geom.material is not None and geom.material.color is not None:
        #                 mesh.set_color(geom.material.color)

        #         g_info = {
        #             "type": geom_type,
        #             "data": geom_data,
        #             "pos": geom.origin[:3, 3],
        #             "quat": gu.R_to_quat(geom.origin[:3, :3]),
        #             "mesh": mesh,
        #             "is_col": geom_is_col,
        #         }
        #         l_info["g_infos"].append(g_info)

    #########################  non-base joints and links #########################
    for joint in robot.joints:
        idx = link_name_to_idx[joint.child]
        l_info = l_infos[idx]
        j_info = j_infos[idx]

        j_info["name"] = joint.name
        j_info["pos"] = np.zeros(3)
        j_info["rot"] = np.eye(3)

        j_info["axis"] = joint.axis

        l_info["parent_idx"] = link_name_to_idx[joint.parent]
        l_info["pos"] = joint.origin[:3, 3]
        l_info["rot"] = joint.origin[:3, :3]

        if joint.joint_type == "fixed":
            j_info["dofs_motion_ang"] = np.zeros((0, 3))
            j_info["dofs_motion_vel"] = np.zeros((0, 3))
            j_info["dofs_limit"] = np.zeros((0, 2))
            j_info["dofs_stiffness"] = np.zeros((0))

            j_info["type"] = "fixed"
            j_info["n_qs"] = 0
            j_info["n_dofs"] = 0
            j_info["init_qpos"] = np.zeros(0)

        elif joint.joint_type == "revolute":
            j_info["dofs_motion_ang"] = np.array([joint.axis])
            j_info["dofs_motion_vel"] = np.zeros((1, 3))
            j_info["dofs_limit"] = np.array(
                [
                    [
                        joint.limit.lower if joint.limit.lower is not None else -np.inf,
                        joint.limit.upper if joint.limit.upper is not None else np.inf,
                    ]
                ]
            )
            j_info["dofs_stiffness"] = np.array([0.0])

            j_info["type"] = "revolute"
            j_info["n_qs"] = 1
            j_info["n_dofs"] = 1
            j_info["init_qpos"] = np.zeros(1)

        elif joint.joint_type == "continuous":
            j_info["dofs_motion_ang"] = np.array([joint.axis])
            j_info["dofs_motion_vel"] = np.zeros((1, 3))
            j_info["dofs_limit"] = np.array([[-np.inf, np.inf]])
            j_info["dofs_stiffness"] = np.array([0.0])

            j_info["type"] = "revolute"
            j_info["n_qs"] = 1
            j_info["n_dofs"] = 1
            j_info["init_qpos"] = np.zeros(1)

        elif joint.joint_type == "prismatic":
            j_info["dofs_motion_ang"] = np.zeros((1, 3))
            j_info["dofs_motion_vel"] = np.array([joint.axis])
            j_info["dofs_limit"] = np.array(
                [
                    [
                        joint.limit.lower if joint.limit.lower is not None else -np.inf,
                        joint.limit.upper if joint.limit.upper is not None else np.inf,
                    ]
                ]
            )
            j_info["dofs_stiffness"] = np.array([0.0])

            j_info["type"] = "prismatic"
            j_info["n_qs"] = 1
            j_info["n_dofs"] = 1
            j_info["init_qpos"] = np.zeros(1)

        elif joint.joint_type == "floating":
            raise NotImplementedError("Floating joint not supported yet")
            # j_info["dofs_motion_ang"] = np.eye(6, 3, -3)
            # j_info["dofs_motion_vel"] = np.eye(6, 3)
            # j_info["dofs_limit"] = np.tile([-np.inf, np.inf], (6, 1))
            # j_info["dofs_stiffness"] = np.zeros(6)

            # j_info["type"] = "free"
            # j_info["n_qs"] = 7
            # j_info["n_dofs"] = 6
            # j_info["init_qpos"] = np.concatenate([gu.zero_pos(), gu.identity_quat()])

        else:
            raise Exception(f"Unsupported URDF joint type: {joint.joint_type}")

        # TODO: parse these

        # j_info["dofs_invweight"] = gu.default_dofs_invweight(j_info["n_dofs"])
        # j_info["dofs_sol_params"] = gu.default_solver_params(j_info["n_dofs"])
        # j_info["dofs_kp"] = gu.default_dofs_kp(j_info["n_dofs"])
        # j_info["dofs_kv"] = gu.default_dofs_kv(j_info["n_dofs"])
        # j_info["dofs_force_range"] = gu.default_dofs_force_range(j_info["n_dofs"])

        # NOTE: This is the stand-in default value from genesis
        j_info["dofs_force_range"] = np.tile([[-100, 100]], [j_info["n_dofs"], 1])

        # if joint.joint_type in ["floating", "fixed"]:
        #     j_info["dofs_damping"] = gu.free_dofs_damping(j_info["n_dofs"])
        #     j_info["dofs_armature"] = gu.free_dofs_armature(j_info["n_dofs"])
        # else:
        #     j_info["dofs_damping"] = gu.default_dofs_damping(j_info["n_dofs"])
        #     j_info["dofs_armature"] = gu.default_dofs_armature(j_info["n_dofs"])

        # if joint.safety_controller is not None:
        #     if joint.safety_controller.k_position is not None:
        #         j_info["dofs_kp"] = np.tile(
        #             joint.safety_controller.k_position, j_info["n_dofs"]
        #         )
        #     if joint.safety_controller.k_velocity is not None:
        #         j_info["dofs_kv"] = np.tile(
        #             joint.safety_controller.k_velocity, j_info["n_dofs"]
        #         )

        # NEW: Velocity limit handling
        # Matching the way Genesis handled it (though, I'm not a fan)
        j_info["dofs_velocity_range"] = np.tile([[-100, 100]], [j_info["n_dofs"], 1])

        if joint.limit is not None:
            if joint.limit.effort is not None:
                j_info["dofs_force_range"] = (
                    j_info["dofs_force_range"]
                    / np.abs(j_info["dofs_force_range"])
                    * joint.limit.effort
                )
            # NEW: Velocity limit handling
            if joint.limit.velocity is not None:
                j_info["dofs_velocity_range"] = (
                    j_info["dofs_velocity_range"]
                    / np.abs(j_info["dofs_velocity_range"])
                    * joint.limit.velocity
                )

    l_infos, j_infos, _ = _order_links(l_infos, j_infos)
    ######################### first joint and base link #########################
    j_info = j_infos[0]
    l_info = l_infos[0]

    j_info["pos"] = np.zeros(3)
    j_info["rot"] = np.eye(3)
    j_info["name"] = f"joint_{l_info['name']}"

    # Genesis parses the base link as having an associated joint even though this is fixed
    j_info["axis"] = np.zeros(3)

    l_info["pos"] = np.zeros(3)
    l_info["rot"] = np.eye(3)

    if not fixed_base:
        raise NotImplementedError("Base link must be fixed for now")
        # j_info["dofs_motion_ang"] = np.eye(6, 3, -3)
        # j_info["dofs_motion_vel"] = np.eye(6, 3)
        # j_info["dofs_limit"] = np.tile([-np.inf, np.inf], (6, 1))
        # j_info["dofs_stiffness"] = np.zeros(6)

        # j_info["type"] = "free"
        # j_info["n_qs"] = 7
        # j_info["n_dofs"] = 6
        # j_info["init_qpos"] = np.concatenate([gu.zero_pos(), gu.identity_quat()])
    else:
        j_info["dofs_motion_ang"] = np.zeros((0, 3))
        j_info["dofs_motion_vel"] = np.zeros((0, 3))
        j_info["dofs_limit"] = np.zeros((0, 2))
        j_info["dofs_stiffness"] = np.zeros((0))

        j_info["type"] = "fixed"
        j_info["n_qs"] = 0
        j_info["n_dofs"] = 0
        j_info["init_qpos"] = np.zeros(0)

    # j_info["dofs_invweight"] = gu.default_dofs_invweight(j_info["n_dofs"])
    # j_info["dofs_damping"] = gu.free_dofs_damping(j_info["n_dofs"])
    # j_info["dofs_armature"] = gu.free_dofs_armature(j_info["n_dofs"])
    # j_info["dofs_sol_params"] = gu.default_solver_params(j_info["n_dofs"])

    # j_info["dofs_kp"] = gu.default_dofs_kp(j_info["n_dofs"])
    # j_info["dofs_kv"] = gu.default_dofs_kv(j_info["n_dofs"])
    # j_info["dofs_force_range"] = gu.default_dofs_force_range(j_info["n_dofs"])

    # from IPython import embed; embed()
    # apply scale
    # for l_info in l_infos:
    #     l_info["pos"] *= morph.scale
    #     l_info["inertial_pos"] *= morph.scale

    #     if l_info["inertial_mass"] is not None:
    #         l_info["inertial_mass"] *= morph.scale**3
    #     if l_info["inertial_i"] is not None:
    #         l_info["inertial_i"] *= morph.scale**5

    #     for g_info in l_info["g_infos"]:
    #         g_info["pos"] *= morph.scale

    #         # TODO: parse friction
    #         g_info["friction"] = gu.default_friction()
    #         g_info["sol_params"] = gu.default_solver_params(n=1)[0]

    # for j_info in j_infos:
    #     j_info["pos"] *= morph.scale

    return l_infos, j_infos


def merge_fixed_links(robot, links_to_keep):
    links = robot.links.copy()
    joints = robot.joints.copy()
    link_name_to_idx = {link.name: idx for idx, link in enumerate(links)}
    original_to_merged = {}

    while True:
        fixed_joint_found = False
        for joint in joints:
            if joint.joint_type == "fixed" and joint.child not in links_to_keep:
                parent_name = joint.parent
                child_name = joint.child

                if parent_name in original_to_merged:
                    parent_name = original_to_merged[parent_name]
                if child_name in original_to_merged:
                    child_name = original_to_merged[child_name]

                parent_idx = link_name_to_idx.get(parent_name)
                child_idx = link_name_to_idx.get(child_name)

                if parent_idx is None or child_idx is None:
                    continue

                parent_link = links[parent_idx]
                child_link = links[child_idx]

                if parent_link.name not in original_to_merged:
                    original_to_merged[parent_link.name] = parent_link.name
                original_to_merged[child_link.name] = original_to_merged[
                    parent_link.name
                ]

                update_subtree(links, joints, child_link.name, joint.origin)
                merge_inertia(parent_link, child_link)
                # parent_link.visuals.extend(child_link.visuals)
                parent_link.collisions.extend(child_link.collisions)

                links.pop(child_idx)
                joints.remove(joint)

                link_name_to_idx = {link.name: idx for idx, link in enumerate(links)}

                fixed_joint_found = True
                break

        if not fixed_joint_found:
            break

    for joint in joints:
        if joint.parent in original_to_merged:
            joint.parent = original_to_merged[joint.parent]
        if joint.child in original_to_merged:
            joint.child = original_to_merged[joint.child]

    return urdfpy.URDF(
        robot.name,
        links=links,
        joints=joints,
        # materials=robot.materials,
    )


def translate_inertia(I, m, dist):
    """Translate inertia tensor I by dist for a body with mass m."""
    dist = np.array(dist)
    dist_squared = np.dot(dist, dist)
    identity_matrix = np.eye(3)
    translation_matrix = m * (dist_squared * identity_matrix - np.outer(dist, dist))
    return I + translation_matrix


def rotate_inertia(I, R):
    """Rotate inertia tensor I by rotation matrix R."""
    return R @ I @ R.T


def merge_inertia(link1, link2):
    """Combine two links with fixed joint."""
    if link2.inertial is None:
        return

    if link1.inertial is None:
        link1.inertial = link2.inertial
        return

    m1 = link1.inertial.mass
    m2 = link2.inertial.mass

    com1 = link1.inertial.origin[:3, 3]
    com2 = link2.inertial.origin[:3, 3]

    R1 = link1.inertial.origin[:3, :3]
    R2 = link2.inertial.origin[:3, :3]

    combined_mass = m1 + m2
    if combined_mass > 0:
        combined_com = (m1 * com1 + m2 * com2) / combined_mass
    else:
        combined_com = com1

    # Rotate and translate inertia tensors to the new center of mass
    inertia1_rotated = rotate_inertia(link1.inertial.inertia, R1)
    inertia2_rotated = rotate_inertia(link2.inertial.inertia, R2)

    inertia1_new = translate_inertia(inertia1_rotated, m1, combined_com - com1)
    inertia2_new = translate_inertia(inertia2_rotated, m2, combined_com - com2)

    # Combine the inertia tensors
    combined_inertia = inertia1_new + inertia2_new

    # Set the properties of the combined link
    link1.inertial.mass = combined_mass
    link1.inertial.origin[:3, 3] = combined_com
    link1.inertial.origin[:3, :3] = np.eye(
        3
    )  # Reset rotation to identity since it's now aligned
    link1.inertial.inertia = combined_inertia


def transform_inertial(inertial, transform):
    if inertial is None:
        return None

    new_origin = transform @ inertial

    return urdfpy.Inertial(
        origin=new_origin, mass=inertial.mass, inertia=inertial.inertia
    )


def update_subtree(links, joints, root_name, transform):
    current_name = root_name
    current_idx = next(
        (idx for idx, link in enumerate(links) if link.name == current_name), None
    )
    if current_idx is None:
        return
    current_link = links[current_idx]

    # Apply the transformation to the current link
    if current_link.inertial is not None:
        current_link.inertial.origin = transform @ current_link.inertial.origin

    # for geom in current_link.visuals:
    #     geom.origin = transform @ geom.origin

    for geom in current_link.collisions:
        geom.origin = transform @ geom.origin

    for joint in joints:
        if joint.parent == current_name:
            joint.origin = transform @ joint.origin
