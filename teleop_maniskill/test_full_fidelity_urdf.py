"""Regression tests for the checked-in full-mesh NERO + capture MMHand."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import unittest
import xml.etree.ElementTree as ET

from .prepare_full_fidelity_urdf import (
    ARM_ACTIVE_JOINT_NAMES,
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_URDF,
    HAND_ACTIVE_JOINT_NAMES,
    validate_urdf,
)


class TestFullFidelityUrdf(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = DEFAULT_OUTPUT_URDF.resolve()
        cls.root = ET.parse(cls.path).getroot()

    def test_structure_and_mesh_contract(self) -> None:
        result = validate_urdf(self.path)
        self.assertEqual(
            result,
            {
                "links": 42,
                "joints": 41,
                "active_joints": 28,
                "mesh_references": 57,
                "unique_mesh_files": 45,
            },
        )
        active = {
            joint.get("name")
            for joint in self.root.findall("joint")
            if joint.get("type") in {"revolute", "continuous", "prismatic"}
        }
        self.assertEqual(
            active,
            set(ARM_ACTIVE_JOINT_NAMES) | set(HAND_ACTIVE_JOINT_NAMES),
        )

    def test_every_mesh_is_a_real_project_local_relative_file(self) -> None:
        asset_root = self.path.parent
        suffixes = set()
        for mesh in self.root.findall(".//mesh"):
            filename = mesh.get("filename")
            self.assertIsNotNone(filename)
            pure = PurePosixPath(filename)
            self.assertFalse(pure.is_absolute())
            self.assertNotIn("..", pure.parts)
            self.assertNotIn(":", filename)
            resolved = (asset_root / Path(*pure.parts)).resolve()
            self.assertTrue(resolved.is_relative_to(asset_root))
            self.assertTrue(resolved.is_file(), resolved)
            suffixes.add(resolved.suffix.lower())
        self.assertEqual(suffixes, {".dae", ".stl"})

    def test_arm_keeps_dae_materials_and_hand_has_named_materials(self) -> None:
        for link_name in ("base_link", *(f"link{i}" for i in range(1, 8))):
            link = self.root.find(f"link[@name='{link_name}']")
            self.assertIsNotNone(link)
            for visual in link.findall("visual"):
                self.assertIsNone(
                    visual.find("material"),
                    "a URDF material would overwrite the DAE's multipart CAD materials",
                )
        hand_visuals = [
            visual
            for link in self.root.findall("link")
            if str(link.get("name")).startswith("capture_hand__")
            for visual in link.findall("visual")
        ]
        self.assertTrue(hand_visuals)
        self.assertTrue(all(visual.find("material") is not None for visual in hand_visuals))

    def test_manifest_hashes_all_runtime_assets(self) -> None:
        payload = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema"], "teleop.robot-assets.manifest.v1")
        total = 0
        for entry in payload["runtime_files"]:
            path = (DEFAULT_MANIFEST.parent / entry["path"]).resolve()
            self.assertTrue(path.is_relative_to(DEFAULT_MANIFEST.parent.resolve()))
            data = path.read_bytes()
            self.assertEqual(len(data), entry["bytes"])
            self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"])
            total += len(data)
        self.assertEqual(total, payload["total_bytes"])


if __name__ == "__main__":
    unittest.main()
