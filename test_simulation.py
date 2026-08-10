import importlib.util
import types
import unittest
from pathlib import Path
from unittest.mock import patch

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
        from simulation import grasp

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
        body_config = sapien.physx.get_body_config()
        self.assertEqual(body_config.solver_position_iterations, 25)
        self.assertEqual(body_config.solver_velocity_iterations, 4)
        self.assertAlmostEqual(body_config.sleep_threshold, 0.005)

        shape_count = 0
        for link in simulation.hand.links:
            for shape in link.collision_shapes:
                shape_count += 1
                material = shape.get_physical_material()
                self.assertAlmostEqual(material.get_static_friction(), 2.0)
                self.assertAlmostEqual(material.get_dynamic_friction(), 1.0)
                self.assertNotEqual(
                    shape.get_collision_groups()[2] & (1 << grasp._SELF_COLLISION_BIT),
                    0,
                )
        self.assertGreater(shape_count, 0)
        self.assertEqual(simulation.hand_collision_shape_count, shape_count)

        for joint in simulation.hand.active_joints:
            self.assertAlmostEqual(joint.get_friction(), grasp._JOINT_FRICTION)
            self.assertAlmostEqual(joint.stiffness, grasp._DRIVE_STIFFNESS)
            self.assertAlmostEqual(joint.damping, grasp._DRIVE_DAMPING)
            self.assertAlmostEqual(joint.force_limit, grasp._DRIVE_FORCE_LIMIT)
            self.assertEqual(joint.get_drive_mode(), "force")
            np.testing.assert_allclose(joint.get_drive_velocity_target(), [0.0])

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

    def test_joint_target_is_limit_clipped_and_rate_limited_by_name(self):
        from simulation import grasp

        simulation = self.simulation
        self.assertAlmostEqual(grasp._HAND_MAX_DELTA, 0.30)
        start = simulation.commanded_q.copy()
        desired = simulation.command_hand(np.full(21, 100.0))
        first = simulation._advance_hand_command()
        np.testing.assert_allclose(
            first,
            start + np.clip(desired - start, -0.30, 0.30),
            atol=1e-12,
        )
        self.assertAlmostEqual(np.max(first - start), 0.30)

        for _ in range(20):
            simulation._advance_hand_command()
        np.testing.assert_allclose(simulation.commanded_q, simulation.upper)
        for name, expected in zip(ROBOT_JOINT_NAMES, simulation.commanded_q):
            np.testing.assert_allclose(
                simulation.joints[name].get_drive_target(), [expected], atol=1e-7
            )

    def test_update_keeps_last_desired_pose_and_advances_at_20_hz(self):
        simulation = self.simulation
        simulation.last_time = 10.0
        target = np.full(21, 1.0)

        with patch("simulation.grasp.time.monotonic", return_value=10.05):
            self.assertTrue(simulation.update(target))
        first = simulation.commanded_q.copy()
        np.testing.assert_allclose(
            first,
            simulation.neutral_command + np.clip(
                simulation.desired_q - simulation.neutral_command,
                -0.30,
                0.30,
            ),
            atol=1e-12,
        )

        # No new retarget output keeps the desired pose and continues the
        # bounded transition rather than resetting or following contact error.
        desired = simulation.desired_q.copy()
        with patch("simulation.grasp.time.monotonic", return_value=10.10):
            self.assertTrue(simulation.update(None))
        np.testing.assert_allclose(simulation.desired_q, desired)
        self.assertGreater(np.max(simulation.commanded_q - first), 0.0)

    def test_reset_reapplies_stiff_hold_and_clears_hand_state(self):
        from simulation import grasp

        simulation = self.simulation
        simulation.command_hand(np.full(21, 0.8))
        simulation._advance_hand_command()
        self.assertGreater(np.max(np.abs(simulation.commanded_q)), 0.0)

        simulation._reset()
        np.testing.assert_allclose(simulation.commanded_q, simulation.neutral_command)
        np.testing.assert_allclose(simulation.desired_q, simulation.neutral_command)
        np.testing.assert_allclose(simulation.hand.get_qpos(), simulation.neutral_qpos)
        np.testing.assert_allclose(simulation.hand.get_qvel(), np.zeros(21))

        before = simulation.hand.get_qpos().copy()
        simulation.step_physics(100)
        drift = float(np.max(np.abs(simulation.hand.get_qpos() - before)))
        self.assertLess(drift, 1e-3)

        for joint in simulation.hand.active_joints:
            self.assertAlmostEqual(joint.stiffness, grasp._DRIVE_STIFFNESS)
            self.assertAlmostEqual(joint.damping, grasp._DRIVE_DAMPING)
            self.assertAlmostEqual(joint.force_limit, grasp._DRIVE_FORCE_LIMIT)
            expected = simulation.neutral_command[
                ROBOT_JOINT_NAMES.index(joint.name)
            ]
            np.testing.assert_allclose(joint.get_drive_target(), [expected], atol=1e-8)
        for link in simulation.hand.links:
            for shape in link.collision_shapes:
                material = shape.get_physical_material()
                self.assertAlmostEqual(material.get_static_friction(), 2.0)
                self.assertAlmostEqual(material.get_dynamic_friction(), 1.0)
                self.assertNotEqual(
                    shape.get_collision_groups()[2] & (1 << grasp._SELF_COLLISION_BIT),
                    0,
                )

    def test_displaced_thumb_returns_to_persistent_drive_target(self):
        simulation = self.simulation
        name = "Thumb_MCP_FE"
        index = simulation.joint_index[name]
        target = simulation.neutral_command[ROBOT_JOINT_NAMES.index(name)]
        displaced = simulation.hand.get_qpos().copy()
        displaced[index] = target + 0.20
        simulation.hand.set_qpos(displaced)
        simulation.hand.set_qvel(np.zeros(21))
        before = abs(float(simulation.hand.get_qpos()[index] - target))

        simulation.step_physics(30)

        after = abs(float(simulation.hand.get_qpos()[index] - target))
        self.assertLess(after, before)
        np.testing.assert_allclose(
            simulation.joints[name].get_drive_target(),
            [target],
            atol=1e-8,
        )

    def test_fixed_thumb_drive_target_pushes_cylinder_without_being_overwritten(self):
        import sapien

        simulation = self.simulation
        simulation.rng = type(
            "MidpointRng",
            (),
            {"uniform": lambda self, low, high: (low + high) / 2},
        )()
        simulation._reset()
        thumb = next(
            link for link in simulation.hand.links
            if link.name == "mmhand_thumb_1_finger_7_fingertip_1"
        )
        shape_poses = [
            thumb.entity_pose * shape.local_pose
            for shape in thumb.collision_shapes
        ]
        thumb_surface_x = max(
            float(pose.p[0]) + max(
                float(getattr(shape, "radius", 0.0) or 0.0),
                float(getattr(shape, "half_length", 0.0) or 0.0),
            )
            for pose, shape in zip(shape_poses, thumb.collision_shapes)
        )
        simulation.object.pose = sapien.Pose([
            thumb_surface_x + simulation.radius + 0.002,
            float(shape_poses[-1].p[1]),
            simulation.height / 2,
        ])
        simulation.object_body.disable_gravity = True
        object_start = np.asarray(simulation.object.pose.p, float).copy()
        target_before = simulation.commanded_q.copy()

        try:
            for _ in range(12):
                pose = simulation.hand.pose
                simulation.hand.pose = sapien.Pose(
                    np.asarray(pose.p, float) + [0.006, 0.0, 0.0],
                    pose.q,
                )
                simulation.step_physics(5)

            object_delta_x = float(simulation.object.pose.p[0] - object_start[0])
            self.assertGreater(object_delta_x, 0.035)
            np.testing.assert_allclose(simulation.commanded_q, target_before)
            for name, expected in zip(
                ROBOT_JOINT_NAMES[16:21],
                simulation.commanded_q[16:21],
            ):
                np.testing.assert_allclose(
                    simulation.joints[name].get_drive_target(),
                    [expected],
                    atol=1e-8,
                )
        finally:
            simulation.object_body.disable_gravity = False

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
