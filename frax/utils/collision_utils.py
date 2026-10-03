"""Collision utils"""

from typing import Optional, Tuple, Union
from pathlib import Path
import yaml


def bubblify_to_mine(
    filepath: Union[str, Path],
    child_map: dict[str, str],
    root_link_name: str,
    add_floating_base: bool,
    sc_data: Optional[Tuple[Tuple[str, int, str, int, float]]],
    verbose: bool = True,
) -> dict[str, tuple]:
    """Converts a bubblify-generated spherized collision model to my expected format

    Args:
        filepath (Union[str, Path]): Path to the bubblify YAML file
        child_map (dict[str, str]): Dictonary: parent joint name => child link name
        root_link_name (str): Name of the root link in the URDF (for instance, "pelvis")
        add_floating_base (bool): Whether to account for a 6DOF floating base
        sc_data (Optional[Tuple[tuple]]): Self-collision data. Has the following format:
            ([first link], [first link sphere idx], [second link], [second link sphere idx], [tol])
            for each self-collision pair we care about. None if no self-collision data available.
        verbose (bool, optional): Whether to print info about the loaded data + warn if
            there are links with no collision. Defaults to True.

    Returns:
        dict[str, tuple]:
            Dictionary with entries (all tuples)
            - "positions": Positions of the spheres in each of the robot's joint frames.
                Length = nv. positions[i] has length = num_spheres_for_joint_i
            - "radii": Radii of the spheres associated with each robot joint.
                Length = nv. radii[i] has length = num_spheres_for_joint_i
            - "root_positions": Sphere positions for the fixed-to-world root in robot base frame.
            - "root_radii": Sphere radii for the fixed-to-world root
    """
    with open(filepath, "r") as file:
        data = yaml.safe_load(file)
    all_spheres = data["collision_spheres"]

    if root_link_name not in all_spheres:
        raise ValueError(f"Root link name ({root_link_name}) not found")

    # Assumes that the parent/child mapping has proper ordering
    joint_ordering = list(child_map.keys())

    if add_floating_base:
        # Add 6 virtual links
        joint_ordering = [
            "x_prismatic",
            "y_prismatic",
            "z_prismatic",
            "roll_joint",
            "pitch_joint",
            "yaw_joint",
        ] + list(joint_ordering)
        child_map = {
            "x_prismatic": "x_link",
            "y_prismatic": "y_link",
            "z_prismatic": "z_link",
            "roll_joint": "roll_link",
            "pitch_joint": "pitch_link",
            "yaw_joint": root_link_name,
        } | child_map
        # Note: the new root link will be a virtual link "fixed" to the world
        # It's fine if there is no collision data associated with this
        root_link_name = "x_link"

    # Handle the fixed root link
    try:
        root_link_spheres = all_spheres[root_link_name]
        root_link_positions = tuple(tuple(s["center"]) for s in root_link_spheres)
        root_link_radii = tuple(s["radius"] for s in root_link_spheres)
    except KeyError:
        if verbose:
            print(f"No collision data for link: {root_link_name}")
        root_link_positions = ()
        root_link_radii = ()

    # Handle all moving links
    all_positions = []
    all_radii = []
    for joint_name in joint_ordering:
        child_link_name = child_map[joint_name]
        try:
            child_link_spheres = all_spheres[child_link_name]
            positions = tuple(tuple(s["center"]) for s in child_link_spheres)
            radii = tuple(s["radius"] for s in child_link_spheres)
        except KeyError:
            if verbose:
                print(f"No collision data for link: {child_link_name}")
            positions = ()
            radii = ()
        all_positions.append(positions)
        all_radii.append(radii)

    collision_data = dict(
        positions=tuple(all_positions),
        radii=tuple(all_radii),
        root_positions=root_link_positions,
        root_radii=root_link_radii,
    )

    ####### SELF COLLISION #########

    if sc_data is None:
        empty_sc_data = dict(
            root_sc_pairs=(),
            root_sc_tols=(),
            body_sc_pairs=(),
            body_sc_tols=(),
        )
        return collision_data | empty_sc_data

    # NOTE: we are currently assuming 1 child link per joint
    link_ordering = [child_map[joint] for joint in joint_ordering]
    spheres_per_joint = tuple(len(r) for r in all_radii)

    def get_full_idx(link_name, idx_within_link, spheres_per_link):
        link_idx = link_ordering.index(link_name)
        return idx_within_link + sum(spheres_per_link[:link_idx])

    root_pairs = []
    body_pairs = []
    root_tols = []
    body_tols = []

    for entry in sc_data:
        (
            first_link_name,
            first_link_sphere_idx,
            second_link_name,
            second_link_sphere_idx,
            tol,
        ) = entry
        assert first_link_name != second_link_name
        # Make sure that if we have a root/body pair, that the root comes first
        if second_link_name == root_link_name:
            (
                first_link_name,
                first_link_sphere_idx,
                second_link_name,
                second_link_sphere_idx,
            ) = (
                second_link_name,
                second_link_sphere_idx,
                first_link_name,
                first_link_sphere_idx,
            )
        # Determine the correct indices of the spheres within the full collision model
        # (i.e. not just with respect to the current link)
        if first_link_name == root_link_name:
            idx_0 = first_link_sphere_idx
        else:
            idx_0 = get_full_idx(
                first_link_name, first_link_sphere_idx, spheres_per_joint
            )
        idx_1 = get_full_idx(
            second_link_name, second_link_sphere_idx, spheres_per_joint
        )

        # Collision pair between [robot root] <=> [robot body]
        if first_link_name == root_link_name:
            root_pairs.append((idx_0, idx_1))
            root_tols.append(tol)
        # Collision pair between [robot body] <=> [robot body]
        else:
            body_pairs.append((idx_0, idx_1))
            body_tols.append(tol)

    return collision_data | dict(
        root_sc_pairs=tuple(root_pairs),
        root_sc_tols=tuple(root_tols),
        body_sc_pairs=tuple(body_pairs),
        body_sc_tols=tuple(body_tols),
    )
