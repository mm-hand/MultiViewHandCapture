import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from hand_core import (
    FINGER_CHAINS,
    Kinematics,
    OneEuro,
    StableHandedness,
    StereoProcessor,
    apply_angles,
    enforce_lengths,
    extract_angles,
    geometry_error,
    split_stereo,
    swap_handedness,
)
from config import BONE_TOLERANCE, ROBOT_TO_CONTRACT
from retarget import Retargeter, RobotModel


def straight_hand(handedness):
    points = np.zeros((21, 3), float)
    xs = [25, 0, -20, -35]
    if handedness == "Left":
        xs = [-x for x in xs]
    for mcp, x, y in zip((5, 9, 13, 17), xs, (45, 50, 45, 38)):
        points[mcp : mcp + 4] = [(x, y, 0), (x, y + 30, 0), (x, y + 50, 0), (x, y + 65, 0)]
    side = -1 if handedness == "Left" else 1
    points[1], points[2] = (30 * side, 15, 0), (45 * side, 25, 0)
    direction = points[2] - points[1]
    direction /= np.linalg.norm(direction)
    points[3], points[4] = points[2] + 20 * direction, points[2] + 35 * direction
    return points


class CoreTests(unittest.TestCase):
    def test_unmirrored_handedness_and_hysteresis(self):
        self.assertEqual(swap_handedness("Left"), "Right")
        stable = StableHandedness(3)
        self.assertEqual(stable.update("Right"), "Right")
        self.assertEqual(stable.update("Left"), "Right")
        self.assertEqual(stable.update("Right"), "Right")
        self.assertEqual(stable.update("Left"), "Right")
        self.assertEqual(stable.update("Left"), "Right")
        self.assertEqual(stable.update("Left"), "Left")

    def test_angle_round_trip_for_both_hands(self):
        expected = np.array([20, 30, 25, 130, 40, 15, 70, 25, 20, 80, 35, 10, 60, 20])
        for hand in ("Left", "Right"):
            points = apply_angles(straight_hand(hand), hand, expected)
            np.testing.assert_allclose(extract_angles(points, hand), expected, atol=1e-8)

    def test_hyperextension_removed_and_lengths_preserved(self):
        points = apply_angles(straight_hand("Right"), "Right", np.full(14, -25.0))
        model = Kinematics(calibration_frames=1)
        model.update(points, "Right", 0.0)
        corrected, phase = model.update(points, "Right", 1 / 30)
        self.assertEqual(phase, "GESTURE TRACKING")
        self.assertTrue(np.all(extract_angles(corrected, "Right") >= -1e-8))
        for chain in FINGER_CHAINS:
            for start, end in zip(chain[:-1], chain[1:]):
                self.assertAlmostEqual(
                    np.linalg.norm(corrected[end] - corrected[start]),
                    model.lengths[(start, end)],
                )

    def test_one_euro_arrays_and_reset(self):
        filter_ = OneEuro(1, 0.01, 1)
        zeros, ones = np.zeros((21, 3)), np.ones((21, 3))
        np.testing.assert_array_equal(filter_(zeros, 0), zeros)
        self.assertTrue(np.all((filter_(ones, 1 / 30) > 0) & (filter_.value < 1)))
        filter_.reset()
        np.testing.assert_array_equal(filter_(ones, 1), ones)

    def test_soft_bone_length_band(self):
        points = straight_hand("Right")
        lengths = {
            edge: np.linalg.norm(points[edge[1]] - points[edge[0]])
            for chain in FINGER_CHAINS
            for edge in zip(chain[:-1], chain[1:])
        }
        corrected = enforce_lengths(points * 3, lengths)
        for (start, end), nominal in lengths.items():
            self.assertAlmostEqual(
                np.linalg.norm(corrected[end] - corrected[start]),
                nominal * (1 + BONE_TOLERANCE),
            )

    def test_geometry_gate(self):
        points = np.zeros((21, 3))
        points[:, 2] = 400
        self.assertIsNone(geometry_error(points, 1))
        self.assertEqual(geometry_error(points, 9), "reprojection")
        points[1, 2] = 2000
        self.assertEqual(geometry_error(points, 1), "depth")
        points[1] = (400, 0, 400)
        self.assertEqual(geometry_error(points, 1), "hand-size")

    def test_three_bad_frames_hold_then_hide(self):
        processor = object.__new__(StereoProcessor)
        processor.bad_frames = 0
        processor.kinematics = Kinematics(1)
        processor.last = {
            "handedness": "Right",
            "keypoint_absolute": np.zeros((21, 3)),
            "keypoint_relative": np.zeros((21, 3)),
        }
        for _ in range(3):
            result = processor._reject(processor._empty(), "detection")
            self.assertTrue(result["found"] and result["stale"])
        result = processor._reject(processor._empty(), "detection")
        self.assertFalse(result["found"])
        self.assertIsNone(processor.last)


class RetargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = RobotModel()

    def test_contract_limits_and_mapping(self):
        self.assertEqual(self.model.topic, "/raw_ik_target")
        self.assertEqual(
            self.model.layout,
            "mmhand:J00-J20:urdf_deg:structure_urdf_v2:v3:73FD45FA",
        )
        self.assertEqual(len(self.model.names), 21)
        self.assertTrue(np.all(self.model.lower <= self.model.upper))
        self.assertEqual(self.model.vectors(self.model.lower).shape, (16, 3))
        self.assertEqual(set(self.model.points(self.model.lower)), {
            "1-tip_Link", "2-tip_Link", "3-tip_Link", "4-tip_Link", "5-tip_Link",
        })
        q = np.linspace(0, 0.2, 21)
        published = np.radians(self.model.contract_degrees(q))
        np.testing.assert_allclose(published[np.asarray(ROBOT_TO_CONTRACT)], q)

    def test_analytic_vector_gradient(self):
        q = self.model.lower + 0.3 * (self.model.upper - self.model.lower)
        target = self.model.vectors(q * 0.8)
        retargeter = Retargeter(self.model)
        value, gradient, _ = retargeter.objective(q, target, q * 0.9)
        numeric, step = [], 1e-6
        for index in range(21):
            plus, minus = q.copy(), q.copy()
            plus[index] += step
            minus[index] -= step
            numeric.append(
                (
                    retargeter.objective(plus, target, q * 0.9)[0]
                    - retargeter.objective(minus, target, q * 0.9)[0]
                )
                / (2 * step)
            )
        self.assertTrue(np.isfinite(value))
        np.testing.assert_allclose(gradient, numeric, atol=1e-7)

    def test_vector_solve_and_filter_reset(self):
        human = apply_angles(
            straight_hand("Left"),
            "Left",
            np.array((20, 30, 25, 20, 40, 30, 20, 50, 30, 15, 40, 25, 10, 30)),
        )
        retargeter = Retargeter(self.model)
        target = retargeter.targets(human)
        initial = retargeter.objective(retargeter.raw_q, target)[2]
        result = retargeter.solve(human)
        self.assertTrue(result["success"])
        self.assertLess(result["rms"], initial)
        self.assertTrue(np.all(result["q"] >= self.model.lower - 1e-10))
        self.assertTrue(np.all(result["q"] <= self.model.upper + 1e-10))
        self.assertIsNotNone(retargeter.filter.value)
        retargeter.pause()
        self.assertIsNone(retargeter.filter.value)


class VideoRegressionTests(unittest.TestCase):
    def test_recordings(self):
        folder = Path(__file__).parent / "test_data"
        recordings = (("right_hand.mkv", "Right"), ("left_hand.mkv", "Left"))
        if not all((folder / name).exists() for name, _ in recordings):
            self.skipTest("local ignored recordings are not present")

        for name, expected_hand in recordings:
            with self.subTest(name=name):
                capture, processor = cv2.VideoCapture(str(folder / name)), StereoProcessor()
                labels, radii, phases, angle_history, accepted = [], [], [], [], 0
                fps = capture.get(cv2.CAP_PROP_FPS) or 30
                try:
                    frame_number = 0
                    while True:
                        ok, frame = capture.read()
                        if not ok:
                            break
                        result = processor.process(*split_stereo(frame), frame_number / fps)
                        frame_number += 1
                        if not result["found"] or result["stale"]:
                            continue
                        points = result["keypoint_absolute"]
                        accepted += 1
                        labels.append(result["handedness"])
                        phases.append(result["phase"])
                        radii.append(np.max(np.linalg.norm(points - points[0], axis=1)))
                        if result["phase"].startswith("GESTURE"):
                            angle_history.append(extract_angles(points, result["handedness"]))
                finally:
                    capture.release()
                    processor.close()

                self.assertGreater(accepted, 0.85 * frame_number)
                self.assertGreater(labels.count(expected_hand), 0.99 * len(labels))
                self.assertLessEqual(max(radii), 300)
                self.assertTrue(any(phase.startswith("GESTURE") for phase in phases))
                angles = np.asarray(angle_history)
                self.assertGreater(len(angles), 100)
                for pip, dip in ((3, 4), (6, 7), (9, 10), (12, 13)):
                    self.assertGreater(max(np.ptp(angles[:, pip]), np.ptp(angles[:, dip])), 5)
                self.assertGreaterEqual(np.min(angles), -1e-5)
                first_tracking = next(i for i, phase in enumerate(phases) if phase.startswith("GESTURE"))
                self.assertLess(abs(radii[first_tracking] - radii[first_tracking - 1]), 20)

    def test_touch_recordings(self):
        folder = Path(__file__).parent / "test_data"
        recordings = (("right_touch.mkv", "Right"), ("left_touch.mkv", "Left"))
        if not all((folder / name).exists() for name, _ in recordings):
            self.skipTest("local ignored touch recordings are not present")

        for name, expected_hand in recordings:
            with self.subTest(name=name):
                capture, processor = cv2.VideoCapture(str(folder / name)), StereoProcessor()
                baseline, output, labels = [], [], []
                retargeter = Retargeter() if expected_hand == "Left" else None
                robot_distance, robot_rms, solve_time = [], [], []
                fps = capture.get(cv2.CAP_PROP_FPS) or 30
                frame_number = 0
                try:
                    while True:
                        ok, frame = capture.read()
                        if not ok:
                            break
                        result = processor.process(*split_stereo(frame), frame_number / fps)
                        frame_number += 1
                        if (
                            not result["found"]
                            or result["stale"]
                            or not result["phase"].startswith("GESTURE")
                        ):
                            continue
                        point = processor.kinematics.points_filter.value
                        final = result["keypoint_absolute"]
                        baseline.append([np.linalg.norm(point[4] - point[i]) for i in (8, 12, 16, 20)])
                        output.append([np.linalg.norm(final[4] - final[i]) for i in (8, 12, 16, 20)])
                        labels.append(result["handedness"])
                        if retargeter is not None:
                            started = time.perf_counter()
                            robot = retargeter.solve(final)
                            solve_time.append(time.perf_counter() - started)
                            self.assertTrue(robot["success"])
                            tips, thumb = robot["points"], robot["points"]["5-tip_Link"]
                            robot_distance.append([
                                np.linalg.norm(tips[name] - thumb)
                                for name in ("1-tip_Link", "2-tip_Link", "3-tip_Link", "4-tip_Link")
                            ])
                            robot_rms.append(robot["rms"])
                        self.assertGreaterEqual(np.min(extract_angles(final, result["handedness"])), -1e-5)
                        for chain in FINGER_CHAINS:
                            for start, end in zip(chain[:-1], chain[1:]):
                                ratio = np.linalg.norm(final[end] - final[start]) / processor.kinematics.lengths[(start, end)]
                                self.assertLessEqual(abs(ratio - 1), BONE_TOLERANCE + 1e-8)
                finally:
                    capture.release()
                    processor.close()

                baseline, output = np.asarray(baseline), np.asarray(output)
                self.assertGreater(len(baseline), 0.8 * frame_number)
                self.assertGreater(labels.count(expected_hand), 0.99 * len(labels))
                for finger in range(4):
                    contact = baseline[:, finger] < 15
                    self.assertGreater(np.count_nonzero(contact), 15)
                    self.assertLessEqual(
                        np.median(output[contact, finger]),
                        np.median(baseline[contact, finger]) + 7,
                    )
                    self.assertLessEqual(
                        np.percentile(output[contact, finger], 95),
                        np.percentile(baseline[contact, finger], 95) + 12,
                    )
                if retargeter is not None:
                    robot_distance = np.asarray(robot_distance)
                    contact_medians = [
                        np.median(robot_distance[baseline[:, finger] < 15, finger])
                        for finger in range(4)
                    ]
                    self.assertLess(np.mean(contact_medians), 0.035)
                    self.assertLess(np.median(robot_rms), 0.03)
                    self.assertLess(np.median(solve_time), 0.033)
                    self.assertLess(np.percentile(solve_time, 95), 0.05)


if __name__ == "__main__":
    unittest.main()
