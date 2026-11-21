import xml.etree.ElementTree as ET

from frax.assets import FRANKA_ASSETS_DIR


def _inertials_and_kinematics(urdf):
    root = ET.parse(FRANKA_ASSETS_DIR / urdf).getroot()
    inertials = {
        link.get("name"): ET.tostring(link.find("inertial"))
        for link in root.iter("link")
        if link.find("inertial") is not None
    }
    # Everything defining the kinematic tree (joint limits are allowed to differ)
    kinematics = {
        joint.get("name"): (joint.get("type"),)
        + tuple(
            None if joint.find(tag) is None else ET.tostring(joint.find(tag))
            for tag in ("parent", "child", "origin", "axis")
        )
        for joint in root.iter("joint")
    }
    return inertials, kinematics


def test_fer_and_fr3_match():
    """The FER and FR3 share kinematics, and the FER uses the FR3 inertials
    (franka_description's FER link4 inertia is physically invalid:
    https://github.com/frankarobotics/franka_description/issues/3)"""
    fer_inertials, fer_kinematics = _inertials_and_kinematics("fer.urdf")
    fr3_inertials, fr3_kinematics = _inertials_and_kinematics("fr3.urdf")
    assert fer_inertials == fr3_inertials
    assert fer_kinematics == fr3_kinematics
