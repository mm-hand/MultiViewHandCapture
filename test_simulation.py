import importlib.util
import unittest

import numpy as np

from config import ROBOT_JOINT_NAMES


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


if __name__ == "__main__":
    unittest.main()
