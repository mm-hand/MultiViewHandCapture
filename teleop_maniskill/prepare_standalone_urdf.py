#!/usr/bin/env python3
"""Generate a primitive-only standalone NERO + capture-MMHand URDF.

The kinematic specification is embedded in this module. Generation and
runtime therefore require neither a sibling checkout nor mesh files elsewhere
in MultiViewHandCapture. Geometry is intentionally lightweight: it is a visual
and collision proxy, while every arm/hand joint keeps the production name,
origin, axis, limit, and hand mount used by the teleoperation stack.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_URDF = HERE / "assets" / "standalone_nero_mmhand.urdf"
ACTIVE_TYPES = {"revolute", "continuous", "prismatic"}

ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
HAND_JOINT_NAMES = (
    "Little_MCP_AA",
    "Little_MCP_FE",
    "finger_4_distal_phalanx_1_PIP_Joint",
    "finger_4_fingertip_1_DIP_Joint",
    "Ring_MCP_AA",
    "Ring_MCP_FE",
    "finger_3_distal_phalanx_1_PIP_Joint",
    "finger_3_fingertip_1_DIP_Joint",
    "Middle_MCP_AA",
    "Middle_MCP_FE",
    "finger_2_distal_phalanx_1_PIP_Joint",
    "finger_2_fingertip_1_DIP_Joint",
    "Index_MCP_AA",
    "Index_MCP_FE",
    "finger_1_distal_phalanx_1_PIP_Joint",
    "finger_1_fingertip_1_DIP_Joint",
    "Thumb_MCP_AA",
    "Thumb_MCP_FE",
    "mmhand_thumb_1_finger_7_distal_phalanx_1_PIP_Joint",
    "mmhand_thumb_1_finger_7_fingertip_1_DIP_Joint",
    "Thumb_CMC",
)

NERO_LINK_NAMES = (
    "world",
    "base_link",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "link7",
    "gripper_flange",
)
HAND_LINK_NAMES = (
    "capture_hand__base_link",
    "capture_hand__palm_1",
    "capture_hand__palm_1_finger_base_1",
    "capture_hand__palm_1_abduction_adduction_link_1",
    "capture_hand__palm_1_finger_base_2",
    "capture_hand__palm_1_finger_base_3",
    "capture_hand__palm_1_finger_base_4",
    "capture_hand__palm_1_abduction_adduction_link_2",
    "capture_hand__palm_1_abduction_adduction_link_3",
    "capture_hand__palm_1_abduction_adduction_link_4",
    "capture_hand__mmhand_thumb_1_thumb_base_1",
    "capture_hand__mmhand_thumb_1_thumb_abduction_adduction_link_1",
    "capture_hand__mmhand_thumb_1_finger_7_proximal_phalanx_1",
    "capture_hand__mmhand_thumb_1_finger_7_distal_phalanx_1",
    "capture_hand__mmhand_thumb_1_finger_7_fingertip_1",
    "capture_hand__finger_1_proximal_phalanx_1",
    "capture_hand__finger_1_distal_phalanx_1",
    "capture_hand__finger_1_fingertip_1",
    "capture_hand__finger_2_proximal_phalanx_1",
    "capture_hand__finger_2_distal_phalanx_1",
    "capture_hand__finger_2_fingertip_1",
    "capture_hand__finger_3_proximal_phalanx_1",
    "capture_hand__finger_3_distal_phalanx_1",
    "capture_hand__finger_3_fingertip_1",
    "capture_hand__finger_4_proximal_phalanx_1",
    "capture_hand__finger_4_distal_phalanx_1",
    "capture_hand__finger_4_fingertip_1",
    "capture_hand__1-tip_Link",
    "capture_hand__2-tip_Link",
    "capture_hand__3-tip_Link",
    "capture_hand__4-tip_Link",
    "capture_hand__5-tip_Link",
)


class JointSpec(NamedTuple):
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: str
    rpy: str = "0 0 0"
    axis: str | None = None
    lower: str | None = None
    upper: str | None = None
    effort: str = "100"
    velocity: str = "100"


# This is the locally embedded NERO kinematic template.  Values are copied
# verbatim from the NERO model previously used by the teleoperation demo.
NERO_JOINTS = (
    JointSpec("world_to_base_link", "fixed", "world", "base_link", "0 0 0"),
    JointSpec(
        "joint1", "revolute", "base_link", "link1", "0 0 0.138",
        axis="0 0 1", lower="-2.70526", upper="2.70526", velocity="5",
    ),
    JointSpec(
        "joint2", "revolute", "link1", "link2", "0 0 0",
        "1.5707963  3.1415926 0", "0 0 1", "-1.74", "1.74",
        velocity="5",
    ),
    JointSpec(
        "joint3", "revolute", "link2", "link3", "0 -0.31 0",
        "-1.5707963  0 3.1415926926", "0 0 1", "-2.75", "2.75",
        velocity="5",
    ),
    JointSpec(
        "joint4", "revolute", "link3", "link4", "0 0 0",
        "-1.5707963  0 3.1415926926", "0 0 1", "-1.01", "2.14",
        velocity="5",
    ),
    JointSpec(
        "joint5", "revolute", "link4", "link5", "0 -0.27001 0",
        "1.5707963  -1.5707963  0", "0 0 1", "-2.75", "2.75",
        velocity="5",
    ),
    JointSpec(
        "joint6", "revolute", "link5", "link6", "0 0 0",
        "1.5707963  -1.5707963  0", "0 0 1", "-0.73", "0.95",
        velocity="5",
    ),
    JointSpec(
        "joint7", "revolute", "link6", "link7", "0 -0.0235 0",
        "1.5707963 0 0", "0 0 1", "-1.5707963", "1.5707963",
        velocity="5",
    ),
    JointSpec(
        "gripper_flange_joint", "fixed", "link7", "gripper_flange",
        "0.031 0 -0.0235", "-1.5708 0 -1.5708",
    ),
)

# The active entries are deliberately in capture J00...J20 order.  Fixed
# entries preserve the source capture URDF's complete link-frame tree.
HAND_JOINTS = (
    JointSpec(
        "Little_MCP_AA", "revolute",
        "capture_hand__palm_1_finger_base_4",
        "capture_hand__palm_1_abduction_adduction_link_4",
        "0.098962 -0.040332 -0.023497", "0.0 -0.0 0.0",
        "-0.0 -0.0 -1.0", "0.0", "0.942478",
    ),
    JointSpec(
        "Little_MCP_FE", "revolute",
        "capture_hand__palm_1_abduction_adduction_link_4",
        "capture_hand__finger_4_proximal_phalanx_1",
        "0.010207 0.007601 -0.0114", "0.0 -0.0 0.0",
        "-0.587785 0.809017 0.0", "-0.261799", "1.570796",
    ),
    JointSpec(
        "finger_4_distal_phalanx_1_PIP_Joint", "revolute",
        "capture_hand__finger_4_proximal_phalanx_1",
        "capture_hand__finger_4_distal_phalanx_1",
        "0.031738 0.022892 -6e-05", "0.0 -0.0 0.0",
        "-0.587785 0.809017 0.0", "-0.15708", "1.553343",
    ),
    JointSpec(
        "finger_4_fingertip_1_DIP_Joint", "revolute",
        "capture_hand__finger_4_distal_phalanx_1",
        "capture_hand__finger_4_fingertip_1",
        "0.02598 0.013579 -5e-06", "0.0 -0.0 0.0",
        "-0.587785 0.809017 -0.0", "-0.226893", "1.762783",
    ),
    JointSpec(
        "Ring_MCP_AA", "revolute",
        "capture_hand__palm_1_finger_base_3",
        "capture_hand__palm_1_abduction_adduction_link_3",
        "0.098962 -0.040332 -0.023497", "0.0 -0.0 0.0",
        "-0.0 -0.0 -1.0", "0.0", "1.064651",
    ),
    JointSpec(
        "Ring_MCP_FE", "revolute",
        "capture_hand__palm_1_abduction_adduction_link_3",
        "capture_hand__finger_3_proximal_phalanx_1",
        "0.010207 0.007601 -0.0114", "0.0 -0.0 0.0",
        "-0.587785 0.809017 0.0", "-0.261799", "1.570796",
    ),
    JointSpec(
        "finger_3_distal_phalanx_1_PIP_Joint", "revolute",
        "capture_hand__finger_3_proximal_phalanx_1",
        "capture_hand__finger_3_distal_phalanx_1",
        "0.031738 0.022892 -6e-05", "0.0 -0.0 0.0",
        "-0.587785 0.809017 0.0", "-0.15708", "1.553343",
    ),
    JointSpec(
        "finger_3_fingertip_1_DIP_Joint", "revolute",
        "capture_hand__finger_3_distal_phalanx_1",
        "capture_hand__finger_3_fingertip_1",
        "0.02598 0.013579 -5e-06", "0.0 -0.0 0.0",
        "-0.587785 0.809017 -0.0", "-0.226893", "1.762783",
    ),
    JointSpec(
        "Middle_MCP_AA", "revolute",
        "capture_hand__palm_1_finger_base_2",
        "capture_hand__palm_1_abduction_adduction_link_2",
        "0.098962 -0.040332 -0.023497", "0.0 -0.0 0.0",
        "-0.0 0.0 -1.0", "0.0", "1.029744",
    ),
    JointSpec(
        "Middle_MCP_FE", "revolute",
        "capture_hand__palm_1_abduction_adduction_link_2",
        "capture_hand__finger_2_proximal_phalanx_1",
        "0.01083 0.006683 -0.0114", "0.0 -0.0 0.0",
        "-0.515038 0.857167 -0.0", "-0.244346", "1.570796",
    ),
    JointSpec(
        "finger_2_distal_phalanx_1_PIP_Joint", "revolute",
        "capture_hand__finger_2_proximal_phalanx_1",
        "capture_hand__finger_2_distal_phalanx_1",
        "0.033613 0.020039 -6e-05", "0.0 -0.0 0.0",
        "-0.515038 0.857167 -0.0", "-0.15708", "1.553343",
    ),
    JointSpec(
        "finger_2_fingertip_1_DIP_Joint", "revolute",
        "capture_hand__finger_2_distal_phalanx_1",
        "capture_hand__finger_2_fingertip_1",
        "0.027065 0.011263 -5e-06", "0.0 -0.0 0.0",
        "-0.515038 0.857167 -0.0", "-0.226893", "1.762783",
    ),
    JointSpec(
        "Index_MCP_AA", "revolute",
        "capture_hand__palm_1_finger_base_1",
        "capture_hand__palm_1_abduction_adduction_link_1",
        "0.098962 -0.040332 -0.023497", "0.0 -0.0 0.0",
        "-0.0 0.0 -1.0", "0.0", "1.012291",
    ),
    JointSpec(
        "Index_MCP_FE", "revolute",
        "capture_hand__palm_1_abduction_adduction_link_1",
        "capture_hand__finger_1_proximal_phalanx_1",
        "0.011655 0.00511 -0.0114", "0.0 -0.0 0.0",
        "-0.390731 0.920505 -0.0", "-0.244346", "1.570796",
    ),
    JointSpec(
        "finger_1_distal_phalanx_1_PIP_Joint", "revolute",
        "capture_hand__finger_1_proximal_phalanx_1",
        "capture_hand__finger_1_distal_phalanx_1",
        "0.036074 0.015166 -6e-05", "0.0 -0.0 0.0",
        "-0.390731 0.920505 -0.0", "-0.15708", "1.553343",
    ),
    JointSpec(
        "finger_1_fingertip_1_DIP_Joint", "revolute",
        "capture_hand__finger_1_distal_phalanx_1",
        "capture_hand__finger_1_fingertip_1",
        "0.028369 0.007387 -5e-06", "0.0 -0.0 0.0",
        "-0.390731 0.920505 -0.0", "-0.226893", "1.762783",
    ),
    JointSpec(
        "Thumb_MCP_AA", "revolute",
        "capture_hand__mmhand_thumb_1_thumb_base_1",
        "capture_hand__mmhand_thumb_1_thumb_abduction_adduction_link_1",
        "0.000906 -0.012002 -0.045452", "0.0 -0.0 0.0",
        "-0.070801 0.928071 -0.365612", "-1.2", "1",
    ),
    JointSpec(
        "Thumb_MCP_FE", "revolute",
        "capture_hand__mmhand_thumb_1_thumb_abduction_adduction_link_1",
        "capture_hand__mmhand_thumb_1_finger_7_proximal_phalanx_1",
        "0.005694 0.007003 -0.014507", "0.0 -0.0 0.0",
        "0.862725 0.240955 0.444574", "-0.331613", "1.570796",
    ),
    JointSpec(
        "mmhand_thumb_1_finger_7_distal_phalanx_1_PIP_Joint",
        "revolute",
        "capture_hand__mmhand_thumb_1_finger_7_proximal_phalanx_1",
        "capture_hand__mmhand_thumb_1_finger_7_distal_phalanx_1",
        "0.019099 -0.007442 -0.033335", "0.0 -0.0 0.0",
        "0.862725 0.240955 0.444574", "-0.15708", "1.553343",
    ),
    JointSpec(
        "mmhand_thumb_1_finger_7_fingertip_1_DIP_Joint",
        "revolute",
        "capture_hand__mmhand_thumb_1_finger_7_distal_phalanx_1",
        "capture_hand__mmhand_thumb_1_finger_7_fingertip_1",
        "0.010549 -0.006561 -0.026553", "0.0 -0.0 0.0",
        "0.862725 0.240955 0.444574", "-0.226893", "1.762783",
    ),
    JointSpec(
        "Thumb_CMC", "revolute", "capture_hand__palm_1",
        "capture_hand__mmhand_thumb_1_thumb_base_1",
        "-0.052098 -0.000861 0.0088", "0.0 -0.0 0.0",
        "0.0 0.0 -1.0", "0.0", "1.51",
    ),
    JointSpec(
        "capture_hand__palm_1_fixed_joint", "fixed",
        "capture_hand__base_link", "capture_hand__palm_1",
        "0.099 -0.040324 -0.033543", "0.0 -0.0 0.0",
    ),
    JointSpec(
        "capture_hand__palm_1_finger_base_1_fixed_joint", "fixed",
        "capture_hand__palm_1", "capture_hand__palm_1_finger_base_1",
        "-0.1039 0.040332 0.023697", "0.0 -0.0 0.0",
    ),
    JointSpec(
        "capture_hand__palm_1_finger_base_2_fixed_joint", "fixed",
        "capture_hand__palm_1", "capture_hand__palm_1_finger_base_2",
        "-0.1036 0.064332 0.023697", "0.0 -0.0 0.0",
    ),
    JointSpec(
        "capture_hand__palm_1_finger_base_3_fixed_joint", "fixed",
        "capture_hand__palm_1", "capture_hand__palm_1_finger_base_3",
        "-0.1039 0.088332 0.023697", "0.0 -0.0 0.0",
    ),
    JointSpec(
        "capture_hand__palm_1_finger_base_4_fixed_joint", "fixed",
        "capture_hand__palm_1", "capture_hand__palm_1_finger_base_4",
        "-0.119644 0.112222 0.023665", "0.0 -0.0 0.0",
    ),
    JointSpec(
        "capture_hand__1-tip", "fixed",
        "capture_hand__finger_1_fingertip_1", "capture_hand__1-tip_Link",
        "0.023001459 0.010424364 -0.004367931",
    ),
    JointSpec(
        "capture_hand__2-tip", "fixed",
        "capture_hand__finger_2_fingertip_1", "capture_hand__2-tip_Link",
        "0.021343147 0.013505685 -0.004311705",
    ),
    JointSpec(
        "capture_hand__3-tip", "fixed",
        "capture_hand__finger_3_fingertip_1", "capture_hand__3-tip_Link",
        "0.020106421 0.015325961 -0.004336191",
    ),
    JointSpec(
        "capture_hand__4-tip", "fixed",
        "capture_hand__finger_4_fingertip_1", "capture_hand__4-tip_Link",
        "0.020105907 0.015325727 -0.004336522",
    ),
    JointSpec(
        "capture_hand__5-tip", "fixed",
        "capture_hand__mmhand_thumb_1_finger_7_fingertip_1",
        "capture_hand__5-tip_Link",
        "0.012366569 -0.000590601 -0.022397383",
    ),
)

MOUNT_JOINT = JointSpec(
    "nero_mmhand_mount", "fixed", "gripper_flange",
    "capture_hand__base_link", "0 0 0.00600000005",
    "0 -1.57079621 -3.14159265",
)


class Shape(NamedTuple):
    kind: str
    dimensions: tuple[float, ...]
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)


NERO_SHAPES = {
    "base_link": Shape("cylinder", (0.09, 0.13), (0.0, 0.0, 0.065)),
    "link1": Shape("cylinder", (0.06, 0.10), (0.0, 0.0, 0.01)),
    "link2": Shape(
        "cylinder", (0.045, 0.31), (0.0, -0.155, 0.0),
        (0.0, math.pi / 2.0, -math.pi / 2.0),
    ),
    "link3": Shape("sphere", (0.052,)),
    "link4": Shape(
        "cylinder", (0.042, 0.27001), (0.0, -0.135005, 0.0),
        (0.0, math.pi / 2.0, -math.pi / 2.0),
    ),
    "link5": Shape("sphere", (0.047,)),
    "link6": Shape(
        "cylinder", (0.032, 0.06), (0.0, -0.012, 0.0),
        (0.0, math.pi / 2.0, -math.pi / 2.0),
    ),
    "link7": Shape("box", (0.07, 0.07, 0.07)),
    "gripper_flange": Shape("cylinder", (0.042, 0.025)),
}


def _numbers(text: str) -> tuple[float, float, float]:
    values = tuple(float(value) for value in text.split())
    if len(values) != 3:
        raise ValueError(f"expected three numbers, got {text!r}")
    return values


def _segment_shape(vector: tuple[float, float, float], radius: float) -> Shape:
    x, y, z = vector
    length = math.sqrt(x * x + y * y + z * z)
    if length <= 1e-9:
        return Shape("sphere", (radius,))
    yaw = math.atan2(y, x)
    pitch = math.atan2(math.hypot(x, y), z)
    return Shape(
        "cylinder",
        (radius, length),
        (x * 0.5, y * 0.5, z * 0.5),
        (0.0, pitch, yaw),
    )


def _hand_shape(link_name: str) -> Shape | None:
    if link_name == "capture_hand__base_link":
        return Shape("cylinder", (0.033, 0.045))
    if link_name == "capture_hand__palm_1":
        return Shape("box", (0.082, 0.108, 0.026), (-0.03, 0.035, 0.0))
    if link_name.endswith("-tip_Link"):
        return Shape("sphere", (0.0075,))
    if "_finger_base_" in link_name:
        return None

    outgoing = [joint for joint in HAND_JOINTS if joint.parent == link_name]
    if not outgoing:
        return Shape("sphere", (0.007,))
    # Multi-child palm geometry is handled above; every other visible hand
    # segment has one kinematically relevant child.
    vector = _numbers(outgoing[0].xyz)
    if "abduction_adduction" in link_name:
        radius = 0.009
    elif "thumb_base" in link_name:
        radius = 0.012
    elif "proximal" in link_name:
        radius = 0.009
    elif "distal" in link_name:
        radius = 0.008
    elif "fingertip" in link_name:
        radius = 0.007
    else:
        radius = 0.008
    return _segment_shape(vector, radius)


def _format(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def _add_geometry(parent: ET.Element, shape: Shape) -> None:
    geometry = ET.SubElement(parent, "geometry")
    if shape.kind == "box":
        ET.SubElement(geometry, "box", {"size": _format(shape.dimensions)})
    elif shape.kind == "cylinder":
        radius, length = shape.dimensions
        ET.SubElement(
            geometry,
            "cylinder",
            {"radius": f"{radius:.9g}", "length": f"{length:.9g}"},
        )
    elif shape.kind == "sphere":
        ET.SubElement(
            geometry, "sphere", {"radius": f"{shape.dimensions[0]:.9g}"}
        )
    else:
        raise ValueError(f"unknown primitive kind: {shape.kind}")


def _add_link(root: ET.Element, name: str) -> None:
    link = ET.SubElement(root, "link", {"name": name})
    if name == "world":
        return

    is_hand = name.startswith("capture_hand__")
    shape = _hand_shape(name) if is_hand else NERO_SHAPES.get(name)
    mass = 0.012 if is_hand else 0.7
    inertia = 1e-5 if is_hand else 2e-3
    if name == "capture_hand__palm_1":
        mass, inertia = 0.12, 2e-4
    elif name == "base_link":
        mass, inertia = 2.0, 1e-2

    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": f"{mass:.9g}"})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": f"{inertia:.9g}",
            "iyy": f"{inertia:.9g}",
            "izz": f"{inertia:.9g}",
            "ixy": "0",
            "ixz": "0",
            "iyz": "0",
        },
    )
    if shape is None:
        return

    material_name = "hand" if is_hand else "arm"
    if name.endswith("-tip_Link"):
        material_name = "tip"
    for tag in ("visual", "collision"):
        element = ET.SubElement(link, tag)
        ET.SubElement(
            element,
            "origin",
            {"xyz": _format(shape.xyz), "rpy": _format(shape.rpy)},
        )
        _add_geometry(element, shape)
        if tag == "visual":
            ET.SubElement(element, "material", {"name": material_name})


def _add_joint(root: ET.Element, spec: JointSpec) -> None:
    joint = ET.SubElement(
        root, "joint", {"name": spec.name, "type": spec.joint_type}
    )
    ET.SubElement(joint, "parent", {"link": spec.parent})
    ET.SubElement(joint, "child", {"link": spec.child})
    ET.SubElement(joint, "origin", {"xyz": spec.xyz, "rpy": spec.rpy})
    if spec.axis is not None:
        ET.SubElement(joint, "axis", {"xyz": spec.axis})
    if spec.joint_type in ACTIVE_TYPES:
        if spec.lower is None or spec.upper is None:
            raise ValueError(f"active joint {spec.name} has no limits")
        ET.SubElement(
            joint,
            "limit",
            {
                "lower": spec.lower,
                "upper": spec.upper,
                "effort": spec.effort,
                "velocity": spec.velocity,
            },
        )


def build_tree() -> ET.ElementTree:
    root = ET.Element("robot", {"name": "standalone_nero_capture_mmhand"})
    for name, rgba in (
        ("arm", "0.16 0.31 0.52 1"),
        ("hand", "0.92 0.52 0.16 1"),
        ("tip", "0.96 0.82 0.26 1"),
    ):
        material = ET.SubElement(root, "material", {"name": name})
        ET.SubElement(material, "color", {"rgba": rgba})

    for link_name in (*NERO_LINK_NAMES, *HAND_LINK_NAMES):
        _add_link(root, link_name)
    for joint in (*NERO_JOINTS, *HAND_JOINTS, MOUNT_JOINT):
        _add_joint(root, joint)
    return ET.ElementTree(root)


def _joint_signature(joint: ET.Element) -> tuple[object, ...]:
    origin = joint.find("origin")
    axis = joint.find("axis")
    limit = joint.find("limit")
    return (
        joint.get("type"),
        joint.find("parent").get("link"),
        joint.find("child").get("link"),
        origin.get("xyz"),
        origin.get("rpy"),
        None if axis is None else axis.get("xyz"),
        None if limit is None else limit.get("lower"),
        None if limit is None else limit.get("upper"),
    )


def _spec_signature(spec: JointSpec) -> tuple[object, ...]:
    return (
        spec.joint_type,
        spec.parent,
        spec.child,
        spec.xyz,
        spec.rpy,
        spec.axis,
        spec.lower,
        spec.upper,
    )


def validate_tree(tree: ET.ElementTree) -> dict[str, object]:
    root = tree.getroot()
    links = root.findall("link")
    joints = root.findall("joint")
    link_names = [str(link.get("name")) for link in links]
    joint_names = [str(joint.get("name")) for joint in joints]
    errors: list[str] = []

    if len(link_names) != 42 or len(set(link_names)) != 42:
        errors.append(f"expected 42 unique links, got {len(set(link_names))}")
    if len(joint_names) != 41 or len(set(joint_names)) != 41:
        errors.append(f"expected 41 unique joints, got {len(set(joint_names))}")

    link_set = set(link_names)
    children: list[str] = []
    for joint in joints:
        parent = str(joint.find("parent").get("link"))
        child = str(joint.find("child").get("link"))
        if parent not in link_set or child not in link_set:
            errors.append(
                f"{joint.get('name')} references unknown links {parent}->{child}"
            )
        children.append(child)
    if len(children) != len(set(children)):
        errors.append("a link has more than one parent joint")
    if link_set - set(children) != {"world"}:
        errors.append(f"invalid roots: {sorted(link_set - set(children))}")

    active = [
        str(joint.get("name"))
        for joint in joints
        if joint.get("type") in ACTIVE_TYPES
    ]
    expected_active = [*ARM_JOINT_NAMES, *HAND_JOINT_NAMES]
    if active != expected_active:
        errors.append(f"active-joint order mismatch: {active}")

    expected_joints = {
        spec.name: spec for spec in (*NERO_JOINTS, *HAND_JOINTS, MOUNT_JOINT)
    }
    for joint in joints:
        name = str(joint.get("name"))
        spec = expected_joints.get(name)
        if spec is None or _joint_signature(joint) != _spec_signature(spec):
            errors.append(f"kinematic signature mismatch for {name}")

    mesh_nodes = root.findall(".//mesh")
    if mesh_nodes:
        errors.append(f"expected no mesh elements, found {len(mesh_nodes)}")
    primitive_tags = {"box", "cylinder", "sphere"}
    for owner in root.findall(".//visual") + root.findall(".//collision"):
        geometry = owner.find("geometry")
        shapes = [] if geometry is None else list(geometry)
        if len(shapes) != 1 or shapes[0].tag not in primitive_tags:
            errors.append("visual/collision is not exactly one URDF primitive")
    serialized = ET.tostring(root, encoding="unicode")
    if "package://" in serialized or "filename=" in serialized:
        errors.append("external asset reference remains")

    if errors:
        raise ValueError(
            "standalone URDF validation failed:\n  - " + "\n  - ".join(errors)
        )
    return {
        "links": len(links),
        "joints": len(joints),
        "active_joints": len(active),
        "arm_dof": len(ARM_JOINT_NAMES),
        "hand_dof": len(HAND_JOINT_NAMES),
        "meshes": 0,
        "root": "world",
    }


def _serialize(tree: ET.ElementTree) -> bytes:
    ET.indent(tree, space="  ")
    buffer = io.BytesIO()
    tree.write(buffer, encoding="utf-8", xml_declaration=True)
    return buffer.getvalue()


def _write_if_changed(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_bytes() == content:
        return "unchanged"
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    temporary.chmod(0o644)
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return "written"


def ensure_standalone_urdf(output: str | Path = DEFAULT_OUTPUT_URDF) -> Path:
    output_path = Path(output).expanduser().resolve()
    tree = build_tree()
    validate_tree(tree)
    _write_if_changed(output_path, _serialize(tree))
    validate_tree(ET.parse(output_path))
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_URDF)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the existing output without rewriting it.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output = args.output.expanduser().resolve()
        status = "valid"
        if args.check_only:
            validation = validate_tree(ET.parse(output))
        else:
            before = output.read_bytes() if output.is_file() else None
            output = ensure_standalone_urdf(output)
            status = "unchanged" if before == output.read_bytes() else "written"
            validation = validate_tree(ET.parse(output))
        print(
            json.dumps(
                {"status": status, "output": str(output), **validation},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (ET.ParseError, OSError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
