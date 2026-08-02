import unittest
from pathlib import Path

import cv2
import numpy as np

from hand_core import (
    FINGER_CHAINS,
    HandDetector,
    Kinematics,
    OneEuro,
    StableHandedness,
    StereoProcessor,
    apply_angles,
    enforce_lengths,
    extract_angles,
    geometry_error,
    relative_points,
    split_stereo,
)
from config import (
    ANGLE_FILTER,
    BONE_TOLERANCE,
    FINGER_PAD_AXES,
    HAND_LANDMARKER_PATH,
    MAX_DEPTH_MM,
    MMHAND_PALM_ORIGIN,
    MCP_AA_NEUTRAL_DEG,
    MP_DETECTION_CONFIDENCE,
    MP_PRESENCE_CONFIDENCE,
    MP_TRACKING_CONFIDENCE,
    POINT_FILTER,
    ROBOT_JOINT_NAMES,
    THUMB_TIP_SCALE,
)
from retarget import Retargeter, RobotModel
from track import (
    ROBOT_CAMERA_DISTANCE,
    _human_view_wxyz,
    _robot_camera_pose,
)


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


def _unit(vector):
    return vector / max(np.linalg.norm(vector), 1e-9)


class CoreTests(unittest.TestCase):
    def test_initial_palm_facing_camera_poses(self):
        camera = _unit(np.array((-0.26, 0, 0.06)) - np.array((0, 0, 0.06)))
        np.testing.assert_array_equal(camera, (-1, 0, 0))
        right_rotation = np.diag((-1, -1, 1))
        np.testing.assert_array_equal(right_rotation @ np.array((1, 0, 0)), camera)
        np.testing.assert_array_equal(right_rotation @ np.array((0, 0, 1)), (0, 0, 1))

        for handedness in ("Left", "Right"):
            points = relative_points(straight_hand(handedness))
            wxyz = _human_view_wxyz(handedness, points)
            w, x, y, z = wxyz
            rotation = np.array(
                (
                    (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
                    (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
                    (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
                )
            )
            palm_base = rotation @ (points[5] - points[17])
            normal = _unit(np.cross(_unit(points[5] - points[17]), _unit(points[9])))
            if handedness == "Left":
                normal = -normal
            self.assertAlmostEqual(palm_base[2], 0, places=8)
            self.assertGreater(np.dot(rotation @ normal, camera), 0.99)

        model = RobotModel()
        position, look_at, up = _robot_camera_pose(model)
        robot_camera = _unit(position - look_at)
        self.assertGreater(np.dot(robot_camera, -model.palm_frame[:, 0]), 0.999)
        self.assertAlmostEqual(np.linalg.norm(position - look_at), ROBOT_CAMERA_DISTANCE)
        _, pad_directions = model.fingertip_pads(np.zeros(21))
        self.assertTrue(np.all(pad_directions[1:] @ robot_camera > 0.9))
        transforms = model.fk(np.zeros(21))
        robot_base = (
            transforms["finger_1_proximal_phalanx_1"][:3, 3]
            - transforms["finger_4_proximal_phalanx_1"][:3, 3]
        )
        level_up = _unit(np.cross(-robot_camera, robot_base))
        if np.dot(level_up, model.palm_frame[:, 2]) < 0:
            level_up = -level_up
        rolled_right = _unit(np.cross(-robot_camera, up))
        self.assertAlmostEqual(
            np.degrees(np.arctan2(np.dot(level_up, rolled_right), np.dot(level_up, up))),
            15,
            places=6,
        )
        self.assertAlmostEqual(np.dot(robot_base, robot_camera), 0, places=8)

    def test_handedness_hysteresis(self):
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

    def test_filter_configuration_matches_previous_commit(self):
        self.assertEqual(POINT_FILTER, (1.0, 0.01, 1.0))
        self.assertEqual(ANGLE_FILTER, (1.0, 0.02, 1.0))

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
        self.assertIsNone(geometry_error(points, 30))
        self.assertEqual(geometry_error(points, 31), "reprojection")
        points[:, 2] = -100
        self.assertIsNone(geometry_error(points, 1))
        points[:, 2] = MAX_DEPTH_MM
        self.assertIsNone(geometry_error(points, 1))
        points[1, 2] = MAX_DEPTH_MM + 1
        self.assertEqual(geometry_error(points, 1), "depth")
        points[:, 2] = 400
        points[1] = (400, 0, 400)
        self.assertEqual(geometry_error(points, 1), "hand-size")

    def test_mediapipe_tasks_configuration_and_model(self):
        self.assertTrue(HAND_LANDMARKER_PATH.is_file())
        self.assertEqual(
            (MP_DETECTION_CONFIDENCE, MP_PRESENCE_CONFIDENCE, MP_TRACKING_CONFIDENCE),
            (0.5, 0.6, 0.6),
        )
        detector = HandDetector()
        try:
            image = np.zeros((64, 64, 3), np.uint8)
            self.assertEqual(detector.detect(image, 0.0), (None, None))
            self.assertEqual(detector.detect(image, 0.0), (None, None))
            self.assertEqual(detector.last_timestamp_ms, 1)
        finally:
            detector.close()

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

    def test_urdf_limits_and_order(self):
        self.assertEqual(self.model.names, ROBOT_JOINT_NAMES)
        self.assertEqual(len(self.model.joints), 31)
        self.assertTrue(np.all(self.model.lower <= self.model.upper))
        self.assertEqual(self.model.thumb(self.model.lower)[0].shape, (3,))
        np.testing.assert_allclose(self.model.palm_frame.T @ self.model.palm_frame, np.eye(3), atol=1e-8)
        self.assertGreater(np.linalg.det(self.model.palm_frame), 0)
        self.assertEqual(np.asarray(MMHAND_PALM_ORIGIN).shape, (3,))
        self.assertGreater(THUMB_TIP_SCALE, 0)

    def test_robot_fingertip_pad_directions(self):
        q = (self.model.lower + self.model.upper) / 2
        tips, directions = self.model.fingertip_pads(q)
        self.assertEqual(tips.shape, (5, 3))
        self.assertEqual(directions.shape, (5, 3))
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1, atol=1e-8)
        thumb_tip, thumb_normal, _ = self.model.thumb(q)
        np.testing.assert_allclose(
            (tips[0] - self.model.palm_position) @ self.model.palm_frame,
            thumb_tip,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            _unit(directions[0] @ self.model.palm_frame), thumb_normal, atol=1e-8
        )
        transforms = self.model.fk(q)
        for index, axis in enumerate(FINGER_PAD_AXES, 1):
            link = f"finger_{index}_fingertip_1"
            local_direction = transforms[link][:3, :3].T @ directions[index]
            np.testing.assert_allclose(local_direction, _unit(axis), atol=1e-8)

    def test_direct_fingers_and_thumb_solve(self):
        expected = np.array((20, 30, 25, 20, 40, 30, 20, 50, 30, 15, 40, 25, 10, 30))
        human = apply_angles(
            straight_hand("Left"), "Left", expected,
        )
        retargeter = Retargeter(self.model)
        result = retargeter.solve(relative_points(human))
        self.assertIsNotNone(result)
        self.assertTrue(np.all(result >= self.model.lower - 1e-10))
        self.assertTrue(np.all(result <= self.model.upper + 1e-10))
        np.testing.assert_allclose(
            np.degrees(result[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]]),
            expected[[11, 12, 13, 8, 9, 10, 5, 6, 7, 2, 3, 4]],
        )
        np.testing.assert_allclose(np.degrees(result[[0, 4, 8, 12]]), MCP_AA_NEUTRAL_DEG)
        retargeter.pause()
        np.testing.assert_array_equal(retargeter.q, retargeter.neutral)

    def test_thumb_targets(self):
        points = relative_points(straight_hand("Left"))
        retargeter = Retargeter(self.model)
        tip, vectors, nearest = retargeter._task(points)
        np.testing.assert_allclose(tip, points[4] * THUMB_TIP_SCALE)
        np.testing.assert_allclose(vectors, points[[8, 12, 16, 20]] - points[4])
        np.testing.assert_array_equal(nearest, np.argsort(np.linalg.norm(vectors, axis=1))[:2])


class VideoRegressionTests(unittest.TestCase):
    def test_recordings(self):
        folder = Path(__file__).parent / "test_data"
        recordings = (("right_hand.mkv", "Right"), ("left_hand.mkv", "Left"))
        if not all((folder / name).exists() for name, _ in recordings):
            self.skipTest("local ignored recordings are not present")

        for name, expected_hand in recordings:
            with self.subTest(name=name):
                capture, processor = cv2.VideoCapture(str(folder / name)), StereoProcessor()
                labels, radii, phases, angle_history = [], [], [], []
                accepted = 0
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

if __name__ == "__main__":
    unittest.main()
