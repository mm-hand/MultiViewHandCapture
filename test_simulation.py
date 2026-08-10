import importlib.util
import types
import unittest
from pathlib import Path

import numpy as np

from config import ROBOT_JOINT_NAMES


@unittest.skipUnless(importlib.util.find_spec("sapien"), "SAPIEN is optional")
class SimulationTests(unittest.TestCase):
    def setUp(self):
        from simulation import GraspSimulation

        self.simulation = GraspSimulation(headless=True)

    def tearDown(self):
        self.simulation.close()

    def test_latest_urdf_and_joint_contract(self):
        import simulation

        self.assertEqual(type(self.simulation).__module__, "simulation.grasp")
        self.assertEqual(
            {joint.name for joint in self.simulation.hand.active_joints},
            set(ROBOT_JOINT_NAMES),
        )
        self.assertFalse((Path(simulation.__file__).parent / "assets").exists())

    def test_contact_material_and_object_damping(self):
        import sapien

        simulation = self.simulation
        self.assertAlmostEqual(simulation.hand_material.get_static_friction(), 2.0)
        self.assertAlmostEqual(simulation.hand_material.get_dynamic_friction(), 1.0)
        self.assertAlmostEqual(simulation.object_material.get_static_friction(), 0.3)
        self.assertAlmostEqual(simulation.object_material.get_dynamic_friction(), 0.3)
        self.assertAlmostEqual(simulation.object_body.get_linear_damping(), 2.0)
        self.assertAlmostEqual(simulation.object_body.get_angular_damping(), 2.0)

        scene_config = sapien.physx.get_scene_config()
        self.assertTrue(scene_config.enable_pcm)
        self.assertTrue(scene_config.enable_tgs)
        self.assertTrue(scene_config.enable_friction_every_iteration)
        shape_config = sapien.physx.get_shape_config()
        self.assertAlmostEqual(shape_config.contact_offset, 0.003)
        self.assertAlmostEqual(shape_config.rest_offset, 0.0)

    def test_random_cylinder_reset_and_joint_input(self):
        simulation = self.simulation
        old = simulation.object
        simulation._reset()
        self.assertIsNot(simulation.object, old)
        self.assertNotIn(old, simulation.scene.get_entities())
        self.assertLessEqual(0.025, simulation.radius)
        self.assertLessEqual(simulation.radius, 0.035)
        self.assertLessEqual(0.080, simulation.height)
        self.assertLessEqual(simulation.height, 0.120)
        self.assertAlmostEqual(simulation.object.pose.p[2], simulation.height / 2)
        self.assertTrue(simulation.update(np.zeros(21)))
        self.assertTrue(simulation.update(None))

        for bad in (np.zeros(20), np.full(21, np.nan)):
            with self.assertRaises(ValueError):
                simulation.update(bad)

        values = iter((0.025, 0.080, 0.035, 0.120))
        simulation.rng = type("Rng", (), {"uniform": lambda self, *_: next(values)})()
        simulation._reset()
        self.assertEqual((simulation.radius, simulation.height), (0.025, 0.080))
        simulation._reset()
        self.assertEqual((simulation.radius, simulation.height), (0.035, 0.120))

    def test_joint_limits_are_applied_by_name(self):
        simulation = self.simulation
        simulation.update(np.full(21, 100.0))
        for name, expected in zip(ROBOT_JOINT_NAMES, simulation.upper):
            np.testing.assert_allclose(
                simulation.joints[name].get_drive_target(), [expected], atol=1e-7
            )

    def test_keyboard_mapping(self):
        import sapien

        simulation = self.simulation

        class Window:
            def __init__(self, key):
                self.key = key

            def key_down(self, key):
                return key == self.key

        simulation.viewer = types.SimpleNamespace(closed=False, close=lambda: None)
        for key, sign in (("j", -1), ("u", 1)):
            simulation.viewer.window = Window(key)
            simulation.hand.pose = sapien.Pose([0, 0, 0.2])
            simulation._move(0.1)
            self.assertGreater(sign * (simulation.hand.pose.p[2] - 0.2), 0)

        for negative, positive, index in (("k", "i", 1), ("l", "o", 2), ("m", "p", 3)):
            for key, sign in ((negative, -1), (positive, 1)):
                simulation.viewer.window = Window(key)
                simulation.hand.pose = sapien.Pose([0, 0, 0.2])
                simulation._move(0.1)
                self.assertGreater(sign * simulation.hand.pose.q[index], 0)
        simulation.viewer = None


if __name__ == "__main__":
    unittest.main()
