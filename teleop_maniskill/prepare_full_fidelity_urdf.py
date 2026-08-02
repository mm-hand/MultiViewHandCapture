#!/usr/bin/env python3
"""Build and validate the portable, mesh-based NERO + capture-MMHand URDF.

This is a maintenance importer, not a runtime dependency.  ``build`` reads an
upstream NERO combined URDF and the capture project's MMHand URDF once, then
rewrites every mesh reference into the local ``assets/robot`` bundle.  The
checked-in generated URDF and meshes are sufficient at runtime; neither source
repository is consulted by ``validate`` or by the simulator.

The importer deliberately copies, rather than re-derives, link inertials,
joint frames, axes, limits, mesh origins, and mesh scales.  Only these changes
are made:

* MMHand link names are prefixed to avoid NERO's ``base_link`` collision.
* MMHand fixed-joint names are prefixed; the 21 active J00--J20 names remain
  byte-for-byte unchanged.
* mesh filenames become portable paths relative to the generated URDF.
* four conservative MMHand display materials replace the source's single
  silver material.  Geometry and kinematics are not changed.
* capture inertias that are singular because their values were rounded to
  zero receive a 1e-9 kg m^2 diagonal regularizer required by SAPIEN/PhysX.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from typing import Iterable


HERE = Path(__file__).resolve().parent
ROBOT_ASSET_DIR = HERE / "assets" / "robot"
DEFAULT_OUTPUT_URDF = ROBOT_ASSET_DIR / "nero_capture_mmhand.urdf"
DEFAULT_MANIFEST = ROBOT_ASSET_DIR / "MANIFEST.json"

ACTIVE_TYPES = {"revolute", "continuous", "prismatic"}
ARM_LINK_NAMES = (
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
ARM_JOINT_NAMES = (
    "world_to_base_link",
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
    "gripper_flange_joint",
)
ARM_ACTIVE_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
HAND_PREFIX = "capture_hand__"
MOUNT_JOINT_NAME = "nero_capture_mmhand_mount"
MOUNT_XYZ = "0 0 0.00600000005"
MOUNT_RPY = "0 -1.57079621 -3.14159265"

# Capture J00--J20 contract.  The tuple is ordered for UDP/controller use;
# SAPIEN's raw articulation order is intentionally not assumed anywhere.
HAND_ACTIVE_JOINT_NAMES = (
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

HAND_MATERIALS = (
    ("capture_hand_palm_dark", "0.16 0.18 0.22 1.0"),
    ("capture_hand_finger_base", "0.38 0.41 0.46 1.0"),
    ("capture_hand_silver", "0.72 0.75 0.80 1.0"),
    ("capture_hand_fingertip_dark", "0.045 0.050 0.060 1.0"),
)


class AssetError(RuntimeError):
    """Raised when an imported or generated robot asset breaks its contract."""


def _required_child(parent: ET.Element, tag: str) -> ET.Element:
    child = parent.find(tag)
    if child is None:
        raise AssetError(f"<{parent.tag}> is missing required <{tag}>")
    return child


def _source_member(root: ET.Element, tag: str, name: str) -> ET.Element:
    member = root.find(f"{tag}[@name='{name}']")
    if member is None:
        raise AssetError(f"source robot is missing {tag} {name!r}")
    return copy.deepcopy(member)


def _material_for_hand_link(source_name: str) -> str:
    lowered = source_name.lower()
    if source_name in {"base_link", "palm_1"}:
        return "capture_hand_palm_dark"
    if "fingertip" in lowered:
        return "capture_hand_fingertip_dark"
    if (
        "finger_base" in lowered
        or "abduction_adduction" in lowered
        or "thumb_base" in lowered
    ):
        return "capture_hand_finger_base"
    return "capture_hand_silver"


def _rewrite_arm_meshes(element: ET.Element) -> None:
    for owner in ("visual", "collision"):
        for mesh in element.findall(f"{owner}/geometry/mesh"):
            source_name = mesh.get("filename")
            if not source_name:
                raise AssetError("NERO mesh without a filename")
            basename = PurePosixPath(source_name).name
            if owner == "visual":
                mesh.set("filename", f"meshes/nero/visual/{basename}")
            else:
                mesh.set("filename", f"meshes/nero/collision/{basename}")


def _rewrite_hand_link(link: ET.Element) -> None:
    source_name = link.get("name")
    if not source_name:
        raise AssetError("MMHand link without a name")
    link.set("name", HAND_PREFIX + source_name)

    # SAPIEN 3 rejects a non-positive-definite inertia before an articulation
    # builder is returned.  Eleven source links have a rounded zero eigenvalue
    # and five more have an all-zero matrix.  A 1e-9 diagonal regularizer is
    # three orders below the smallest non-zero source entry and preserves the
    # source mass, COM, principal directions, and practical dynamics.
    inertia = link.find("inertial/inertia")
    if inertia is not None and not _inertia_is_positive_definite(inertia):
        for attribute in ("ixx", "iyy", "izz"):
            value = float(inertia.get(attribute, "0")) + 1e-9
            inertia.set(attribute, f"{value:.12g}")

    for mesh in link.findall(".//mesh"):
        source_filename = mesh.get("filename")
        if not source_filename:
            raise AssetError(f"MMHand link {source_name!r} has an unnamed mesh")
        mesh.set(
            "filename", f"meshes/mmhand/{PurePosixPath(source_filename).name}"
        )

    material_name = _material_for_hand_link(source_name)
    for visual in link.findall("visual"):
        material = visual.find("material")
        if material is None:
            material = ET.SubElement(visual, "material")
        material.attrib.clear()
        material.set("name", material_name)


def _inertia_is_positive_definite(inertia: ET.Element) -> bool:
    """Test a symmetric 3x3 inertia with Sylvester's criterion."""

    ixx = float(inertia.get("ixx", "0"))
    ixy = float(inertia.get("ixy", "0"))
    ixz = float(inertia.get("ixz", "0"))
    iyy = float(inertia.get("iyy", "0"))
    iyz = float(inertia.get("iyz", "0"))
    izz = float(inertia.get("izz", "0"))
    leading2 = ixx * iyy - ixy * ixy
    determinant = (
        ixx * (iyy * izz - iyz * iyz)
        - ixy * (ixy * izz - iyz * ixz)
        + ixz * (ixy * iyz - iyy * ixz)
    )
    return ixx > 0.0 and leading2 > 0.0 and determinant > 0.0


def _rewrite_hand_joint(joint: ET.Element) -> None:
    source_name = joint.get("name")
    if not source_name:
        raise AssetError("MMHand joint without a name")
    if joint.get("type") not in ACTIVE_TYPES:
        joint.set("name", HAND_PREFIX + source_name)
    _required_child(joint, "parent").set(
        "link", HAND_PREFIX + str(_required_child(joint, "parent").get("link"))
    )
    _required_child(joint, "child").set(
        "link", HAND_PREFIX + str(_required_child(joint, "child").get("link"))
    )


def build_tree(nero_source: Path, hand_source: Path) -> ET.ElementTree:
    """Combine source models while preserving all physical model fields."""

    nero_root = ET.parse(nero_source).getroot()
    hand_root = ET.parse(hand_source).getroot()
    output_root = ET.Element("robot", {"name": "nero_capture_mmhand"})
    output_root.append(
        ET.Comment(
            " Modified portable combination: NERO arm + capture MMHand; "
            "see ASSET_PROVENANCE.md. "
        )
    )

    for name, rgba in HAND_MATERIALS:
        material = ET.SubElement(output_root, "material", {"name": name})
        ET.SubElement(material, "color", {"rgba": rgba})

    # Keep the source order stable because it makes diffs and manual URDF
    # inspection much easier.  DAE visuals retain their embedded materials.
    for name in ARM_LINK_NAMES:
        link = _source_member(nero_root, "link", name)
        _rewrite_arm_meshes(link)
        output_root.append(link)
        child_joint = next(
            (
                joint_name
                for joint_name in ARM_JOINT_NAMES
                if (
                    (_source_member(nero_root, "joint", joint_name).find("child"))
                    is not None
                    and _source_member(nero_root, "joint", joint_name)
                    .find("child")
                    .get("link")
                    == name
                )
            ),
            None,
        )
        if child_joint is not None:
            output_root.append(_source_member(nero_root, "joint", child_joint))

    source_active_hand = {
        str(joint.get("name"))
        for joint in hand_root.findall("joint")
        if joint.get("type") in ACTIVE_TYPES
    }
    if source_active_hand != set(HAND_ACTIVE_JOINT_NAMES):
        raise AssetError(
            "capture MMHand active-joint contract changed: "
            f"missing={sorted(set(HAND_ACTIVE_JOINT_NAMES) - source_active_hand)}, "
            f"extra={sorted(source_active_hand - set(HAND_ACTIVE_JOINT_NAMES))}"
        )

    for source_link in hand_root.findall("link"):
        link = copy.deepcopy(source_link)
        _rewrite_hand_link(link)
        output_root.append(link)
    for source_joint in hand_root.findall("joint"):
        joint = copy.deepcopy(source_joint)
        _rewrite_hand_joint(joint)
        output_root.append(joint)

    mount = ET.SubElement(
        output_root, "joint", {"name": MOUNT_JOINT_NAME, "type": "fixed"}
    )
    ET.SubElement(mount, "parent", {"link": "gripper_flange"})
    ET.SubElement(mount, "child", {"link": HAND_PREFIX + "base_link"})
    ET.SubElement(mount, "origin", {"xyz": MOUNT_XYZ, "rpy": MOUNT_RPY})

    mujoco = nero_root.find("mujoco")
    if mujoco is not None:
        output_root.append(copy.deepcopy(mujoco))

    ET.indent(output_root, space="  ")
    tree = ET.ElementTree(output_root)
    validate_tree(tree, DEFAULT_OUTPUT_URDF, check_files=False)
    return tree


def _atomic_write_tree(tree: ET.ElementTree, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{output.name}.", dir=output.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        tree.write(stream, encoding="utf-8", xml_declaration=True)
        stream.write(b"\n")
    temporary.chmod(0o644)
    os.replace(temporary, output)


def _mesh_files(root: ET.Element, urdf_path: Path) -> list[Path]:
    asset_root = urdf_path.parent.resolve()
    files: list[Path] = []
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename:
            raise AssetError("generated mesh without filename")
        pure = PurePosixPath(filename)
        if pure.is_absolute() or ".." in pure.parts or ":" in filename:
            raise AssetError(f"non-portable mesh path: {filename!r}")
        resolved = (asset_root / Path(*pure.parts)).resolve()
        if not resolved.is_relative_to(asset_root):
            raise AssetError(f"mesh escaped robot asset directory: {filename!r}")
        files.append(resolved)
    return files


def validate_tree(
    tree: ET.ElementTree,
    urdf_path: Path = DEFAULT_OUTPUT_URDF,
    *,
    check_files: bool = True,
) -> dict[str, int]:
    """Validate the portable asset's structural and filesystem contract."""

    root = tree.getroot()
    links = root.findall("link")
    joints = root.findall("joint")
    active = [joint for joint in joints if joint.get("type") in ACTIVE_TYPES]
    active_names = [str(joint.get("name")) for joint in active]
    expected_active = set(ARM_ACTIVE_JOINT_NAMES) | set(HAND_ACTIVE_JOINT_NAMES)
    if len(links) != 42:
        raise AssetError(f"expected 42 links, found {len(links)}")
    if len(joints) != 41:
        raise AssetError(f"expected 41 joints, found {len(joints)}")
    if len(active) != 28 or set(active_names) != expected_active:
        raise AssetError(
            "expected NERO 7 + capture MMHand 21 active joints; "
            f"found {active_names}"
        )

    link_names = [str(link.get("name")) for link in links]
    if len(set(link_names)) != len(link_names):
        raise AssetError("duplicate link name in generated URDF")
    joint_names = [str(joint.get("name")) for joint in joints]
    if len(set(joint_names)) != len(joint_names):
        raise AssetError("duplicate joint name in generated URDF")

    mount = root.find(f"joint[@name='{MOUNT_JOINT_NAME}']")
    if mount is None:
        raise AssetError("MMHand mount joint is missing")
    if _required_child(mount, "parent").get("link") != "gripper_flange":
        raise AssetError("MMHand mount parent changed")
    if _required_child(mount, "child").get("link") != HAND_PREFIX + "base_link":
        raise AssetError("MMHand mount child changed")
    origin = _required_child(mount, "origin")
    if origin.get("xyz") != MOUNT_XYZ or origin.get("rpy") != MOUNT_RPY:
        raise AssetError("MMHand mount transform changed")

    meshes = _mesh_files(root, urdf_path)
    # NERO contributes 9 visual + 9 collision meshes.  The capture hand
    # source deliberately equips all 27 visible parts with visuals but only
    # 12 contact-relevant parts with collision, for 57 total references.
    if len(meshes) != 57:
        raise AssetError(f"expected 57 visual/collision mesh references, found {len(meshes)}")
    if check_files:
        missing = sorted({str(path) for path in meshes if not path.is_file()})
        if missing:
            raise AssetError("missing local meshes:\n" + "\n".join(missing))

    return {
        "links": len(links),
        "joints": len(joints),
        "active_joints": len(active),
        "mesh_references": len(meshes),
        "unique_mesh_files": len(set(meshes)),
    }


def validate_urdf(path: Path = DEFAULT_OUTPUT_URDF) -> dict[str, int]:
    return validate_tree(ET.parse(path), path.resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(
    urdf_path: Path = DEFAULT_OUTPUT_URDF,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> None:
    """Record byte sizes and hashes for every runtime robot asset."""

    root = ET.parse(urdf_path).getroot()
    mesh_paths = set(_mesh_files(root, urdf_path.resolve()))
    included = mesh_paths | {
        urdf_path.resolve(),
        (ROBOT_ASSET_DIR / "ASSET_PROVENANCE.md").resolve(),
        (ROBOT_ASSET_DIR / "LICENSE-UltraDexGrasp-Apache-2.0.txt").resolve(),
        (ROBOT_ASSET_DIR / "LICENSE-MMHand-dex-retargeting-MIT.txt").resolve(),
    }
    entries = []
    for path in sorted(included):
        if not path.is_file():
            continue
        entries.append(
            {
                "path": path.relative_to(ROBOT_ASSET_DIR.resolve()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    payload = {
        "schema": "teleop.robot-assets.manifest.v1",
        "generated_asset": urdf_path.name,
        "runtime_files": entries,
        "total_bytes": sum(entry["bytes"] for entry in entries),
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _build_command(args: argparse.Namespace) -> int:
    tree = build_tree(args.nero_source.resolve(), args.hand_source.resolve())
    _atomic_write_tree(tree, args.output.resolve())
    result = validate_urdf(args.output.resolve())
    if args.output.resolve() == DEFAULT_OUTPUT_URDF.resolve():
        write_manifest(args.output.resolve(), args.manifest.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {args.output.resolve()}")
    return 0


def _validate_command(args: argparse.Namespace) -> int:
    result = validate_urdf(args.urdf.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="import the two source URDFs")
    build.add_argument("--nero-source", type=Path, required=True)
    build.add_argument("--hand-source", type=Path, required=True)
    build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_URDF)
    build.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    build.set_defaults(handler=_build_command)

    validate = subparsers.add_parser(
        "validate", help="validate the self-contained generated asset"
    )
    validate.add_argument("--urdf", type=Path, default=DEFAULT_OUTPUT_URDF)
    validate.set_defaults(handler=_validate_command)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
