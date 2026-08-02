"""Regression tests for the primitive-only standalone teleop robot."""

from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from teleop_maniskill.prepare_standalone_urdf import (
    ACTIVE_TYPES,
    ARM_JOINT_NAMES,
    DEFAULT_OUTPUT_URDF,
    HAND_JOINT_NAMES,
    MOUNT_JOINT,
    ensure_standalone_urdf,
    validate_tree,
)


class TestStandaloneUrdf(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = ensure_standalone_urdf()
        cls.root = ET.parse(cls.path).getroot()

    def test_tree_and_dof_contract(self) -> None:
        result = validate_tree(ET.ElementTree(self.root))
        self.assertEqual(result["links"], 42)
        self.assertEqual(result["joints"], 41)
        self.assertEqual(result["active_joints"], 28)

        active = [
            str(joint.get("name"))
            for joint in self.root.findall("joint")
            if joint.get("type") in ACTIVE_TYPES
        ]
        self.assertEqual(active, [*ARM_JOINT_NAMES, *HAND_JOINT_NAMES])

    def test_mount_is_embedded_exactly(self) -> None:
        mount = self.root.find(f"joint[@name='{MOUNT_JOINT.name}']")
        self.assertIsNotNone(mount)
        self.assertEqual(mount.find("parent").get("link"), MOUNT_JOINT.parent)
        self.assertEqual(mount.find("child").get("link"), MOUNT_JOINT.child)
        self.assertEqual(mount.find("origin").get("xyz"), MOUNT_JOINT.xyz)
        self.assertEqual(mount.find("origin").get("rpy"), MOUNT_JOINT.rpy)

    def test_all_geometry_is_local_urdf_primitives(self) -> None:
        self.assertEqual(self.root.findall(".//mesh"), [])
        xml = ET.tostring(self.root, encoding="unicode")
        self.assertNotIn("filename=", xml)
        self.assertNotIn("package://", xml)
        for owner in (
            self.root.findall(".//visual")
            + self.root.findall(".//collision")
        ):
            geometry = owner.find("geometry")
            self.assertIsNotNone(geometry)
            self.assertEqual(len(geometry), 1)
            self.assertIn(geometry[0].tag, {"box", "cylinder", "sphere"})

    def test_generation_is_idempotent_and_portable(self) -> None:
        before = DEFAULT_OUTPUT_URDF.stat()
        self.assertEqual(ensure_standalone_urdf(), DEFAULT_OUTPUT_URDF.resolve())
        after = DEFAULT_OUTPUT_URDF.stat()
        self.assertEqual(before.st_mtime_ns, after.st_mtime_ns)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "robot.urdf"
            ensure_standalone_urdf(output)
            self.assertEqual(
                output.read_bytes(), DEFAULT_OUTPUT_URDF.read_bytes()
            )

    def test_sapien_physics_loader_sees_28_dof(self) -> None:
        try:
            import sapien
        except ImportError:
            self.skipTest("SAPIEN is not installed in this Python environment")

        # Loading visual shapes initializes Vulkan in SAPIEN 3.  Strip only
        # visuals in a temporary copy so this structural physics test also
        # runs on headless CI; the checked-in asset retains all visuals.
        tree = ET.parse(self.path)
        root = tree.getroot()
        for link in root.findall("link"):
            for visual in list(link.findall("visual")):
                link.remove(visual)
        for material in list(root.findall("material")):
            root.remove(material)
        with tempfile.TemporaryDirectory() as directory:
            physics_path = Path(directory) / "physics_only.urdf"
            tree.write(physics_path, encoding="utf-8", xml_declaration=True)
            scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
            robot = scene.create_urdf_loader().load(str(physics_path))
            self.assertIsNotNone(robot)
            self.assertEqual(robot.dof, 28)
            self.assertEqual(
                {joint.name for joint in robot.get_active_joints()},
                set(ARM_JOINT_NAMES) | set(HAND_JOINT_NAMES),
            )


if __name__ == "__main__":
    unittest.main()
