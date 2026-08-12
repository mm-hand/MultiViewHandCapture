import importlib.util
import time
from types import SimpleNamespace
import unittest

import numpy as np

from config import ROBOT_JOINT_NAMES


class _HoldingSimulation:
    def update(self, _command):
        return True

    def close(self):
        pass


class _ClosingSimulation:
    def update(self, _command):
        return False

    def close(self):
        pass


class _FailingSimulation:
    def update(self, _command):
        raise RuntimeError("fake render failure")

    def close(self):
        pass


class SimulationProcessTests(unittest.TestCase):
    def test_spawn_failure_preserves_original_error(self):
        from simulation.process import GraspSimulationProcess

        with self.assertRaises(Exception) as caught:
            GraspSimulationProcess(_factory=lambda: None)
        self.assertNotIsInstance(caught.exception, AssertionError)

    def test_non_blocking_command_validation_and_idempotent_close(self):
        from simulation.process import GraspSimulationProcess

        simulation = GraspSimulationProcess(
            update_hz=100, _factory=_HoldingSimulation
        )
        try:
            self.assertTrue(simulation.update(np.zeros(21)))
            for bad in (np.zeros(20), np.full(21, np.nan)):
                with self.assertRaises(ValueError):
                    simulation.update(bad)
        finally:
            simulation.close()
            simulation.close()

    def test_child_close_is_reported(self):
        from simulation.process import GraspSimulationProcess

        simulation = GraspSimulationProcess(
            update_hz=100, _factory=_ClosingSimulation
        )
        try:
            deadline = time.monotonic() + 2.0
            while simulation.update() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(simulation.update())
        finally:
            simulation.close()

    def test_child_error_includes_remote_traceback(self):
        from simulation.process import GraspSimulationProcess

        simulation = GraspSimulationProcess(
            update_hz=100, _factory=_FailingSimulation
        )
        try:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                try:
                    simulation.update()
                except RuntimeError as error:
                    self.assertIn("fake render failure", str(error))
                    self.assertIn("test_simulation.py", str(error))
                    break
                time.sleep(0.01)
            else:
                self.fail("child error was not propagated")
        finally:
            simulation.close()


@unittest.skipUnless(importlib.util.find_spec("sapien"), "SAPIEN is optional")
class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from simulation.grasp import GraspSimulation
        cls.simulation = GraspSimulation(headless=True)

    @classmethod
    def tearDownClass(cls):
        cls.simulation.close()

    def test_robot_and_reset_contract(self):
        simulation = self.simulation
        self.assertEqual(
            {joint.name for joint in simulation.hand.active_joints},
            set(ROBOT_JOINT_NAMES),
        )
        old = simulation.object
        simulation._reset()
        self.assertIsNot(simulation.object, old)
        np.testing.assert_allclose(simulation.commanded_q, simulation.neutral_command)
        np.testing.assert_allclose(simulation.desired_q, simulation.neutral_command)

    def test_input_limits_and_rate(self):
        simulation = self.simulation
        for bad in (np.zeros(20), np.full(21, np.nan)):
            with self.assertRaises(ValueError):
                simulation.command_hand(bad)
        start = simulation.commanded_q.copy()
        desired = simulation.command_hand(np.full(21, 100.0))
        actual = simulation._advance_hand_command()
        np.testing.assert_allclose(actual, start + np.clip(desired - start, -.30, .30))
        self.assertTrue(np.all((actual >= simulation.lower) & (actual <= simulation.upper)))

    def test_drive_targets_persist(self):
        simulation = self.simulation
        target = simulation.command_hand(np.full(21, .4))
        for _ in range(3):
            simulation._advance_hand_command()
            simulation.step_physics(1)
        for name, expected in zip(ROBOT_JOINT_NAMES, target):
            np.testing.assert_allclose(
                simulation.joints[name].get_drive_target(), [expected], atol=1e-7
            )

    def test_contacts_match_hand_links_by_name(self):
        simulation = self.simulation
        # SAPIEN may return a fresh Python wrapper for a contact link instead
        # of the same wrapper present in articulation.links.
        fingertip = SimpleNamespace(name="finger_1_fingertip_1")
        point = SimpleNamespace(impulse=np.array([0.0, 0.0, 0.1]))
        contact = SimpleNamespace(
            bodies=(
                SimpleNamespace(entity=simulation.object),
                SimpleNamespace(entity=fingertip),
            ),
            points=(point,),
        )
        original_scene = simulation.scene
        simulation.scene = SimpleNamespace(get_contacts=lambda: (contact,))
        try:
            self.assertEqual(simulation._contacts(), (1, 1))
        finally:
            simulation.scene = original_scene

    def test_free_root_tracks_target_with_physical_velocity(self):
        import sapien

        simulation = self.simulation
        simulation._reset_hand()
        start = np.asarray(simulation.hand.pose.p, float).copy()
        simulation.target_pose = sapien.Pose(
            start + np.array([0.0, 0.0, 0.01]),
            simulation.target_pose.q,
        )
        linear, angular = simulation._apply_root_velocity()
        self.assertGreater(linear[2], 0.0)
        np.testing.assert_allclose(angular, 0.0, atol=1e-12)
        simulation.step_physics()
        self.assertGreater(simulation.hand.pose.p[2], start[2])
        self.assertGreater(simulation.hand.root_linear_velocity[2], 0.0)
        simulation._reset_hand()


if __name__ == "__main__":
    unittest.main()
