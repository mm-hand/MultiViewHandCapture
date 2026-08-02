"""Regression tests for the bundled OBJ cases."""

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np
import trimesh

from .assets.objects.generate_builtin_objects import (
    make_box,
    make_can,
    write_obj,
)


OBJECT_ROOT = Path(__file__).resolve().parent / "assets/objects"
GENERATED_EXPECTED = {
    "can": (make_can, 194, 384, [0.76, 0.76, 1.0], 1),
    "box": (make_box, 8, 12, [0.85, 0.65, 1.0], 1),
}
CUP_SHA256 = "2ca38c4ad436015a0bcf7e94dad7289547178472ac6ce122067459613acde0ca"


class TestObjectAssets(unittest.TestCase):
    def test_obj_geometry_is_finite_watertight_and_expected_size(self) -> None:
        expected = {
            "cup": (2000, 4000, [1.46813989, 1.11682928, 0.84699846], 1),
            **{
                name: values[1:]
                for name, values in GENERATED_EXPECTED.items()
            },
        }
        for name, (vertices, faces, extents, components) in expected.items():
            with self.subTest(case=name):
                path = OBJECT_ROOT / name / "mesh/simplified.obj"
                self.assertTrue(path.is_file())
                mesh = trimesh.load(path, force="mesh", process=False)
                self.assertIsInstance(mesh, trimesh.Trimesh)
                self.assertEqual(len(mesh.vertices), vertices)
                self.assertEqual(len(mesh.faces), faces)
                self.assertTrue(np.isfinite(mesh.vertices).all())
                np.testing.assert_allclose(mesh.extents, extents, atol=1e-8)
                self.assertTrue(mesh.is_watertight)
                self.assertEqual(len(mesh.split(only_watertight=False)), components)

    def test_generator_reproduces_checked_in_obj_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            for name, (factory, *_expected) in GENERATED_EXPECTED.items():
                with self.subTest(case=name):
                    generated = output_root / f"{name}.obj"
                    write_obj(generated, factory(), name)
                    checked_in = OBJECT_ROOT / name / "mesh/simplified.obj"
                    self.assertEqual(generated.read_bytes(), checked_in.read_bytes())

    def test_cup_is_the_unmodified_upstream_mesh(self) -> None:
        import hashlib

        cup = OBJECT_ROOT / "cup/mesh/simplified.obj"
        self.assertEqual(hashlib.sha256(cup.read_bytes()).hexdigest(), CUP_SHA256)


if __name__ == "__main__":
    unittest.main()
