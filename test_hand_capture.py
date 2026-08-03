import sys
import types
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

import numpy as np

from config import (
    ANGLE_FILTER,
    HAND_LANDMARKER_PATH,
    KEYPOINT_LAYOUT,
    KEYPOINT_TOPIC,
    MAX_DEPTH_MM,
    MCP_AA_NEUTRAL_DEG,
    MP_DETECTION_CONFIDENCE,
    MP_PRESENCE_CONFIDENCE,
    MP_TRACKING_CONFIDENCE,
    POINT_2D_FILTER,
    POINT_3D_FILTER,
    ROBOT_JOINT_NAMES,
    ROBOT_LAYOUT,
    ROBOT_TOPIC,
    URDF_PATH,
)
from hand_core import (
    FINGER_CHAINS,
    HandDetector,
    Kinematics,
    OneEuro,
    StableHandedness,
    StereoProcessor,
    apply_angles,
    extract_angles,
    geometry_error,
    relative_points,
)
from retarget import AA, Retargeter, RobotModel
from track import RosOutput, _parse_args


def straight_hand(handedness):
    points = np.zeros((21, 3), float)
    xs = [25, 0, -20, -35]
    if handedness == "Left":
        xs = [-x for x in xs]
    for mcp, x, y in zip((5, 9, 13, 17), xs, (45, 50, 45, 38)):
        points[mcp : mcp + 4] = (
            (x, y, 0),
            (x, y + 30, 0),
            (x, y + 50, 0),
            (x, y + 65, 0),
        )
    side = -1 if handedness == "Left" else 1
    points[1], points[2] = (30 * side, 15, 0), (45 * side, 25, 0)
    direction = points[2] - points[1]
    direction /= np.linalg.norm(direction)
    points[3], points[4] = points[2] + 20 * direction, points[2] + 35 * direction
    return points


class CoreTests(unittest.TestCase):
    def test_cli_only_accepts_optional_ros(self):
        self.assertFalse(_parse_args([]).ros)
        self.assertTrue(_parse_args(["--ros"]).ros)
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            _parse_args(["--mode", "points"])

    def test_unsigned_flexion_angles_for_both_hands(self):
        expected = np.array((20, 30, 25, 130, 40, 15, 70, 25, 20, 80, 35, 10, 60, 20))
        for handedness in ("Left", "Right"):
            points = apply_angles(straight_hand(handedness), handedness, expected)
            actual = extract_angles(points, handedness)
            np.testing.assert_allclose(actual, expected, atol=1e-8)
            self.assertTrue(np.all((actual >= 0) & (actual <= 180)))

            reverse = apply_angles(
                straight_hand(handedness), handedness, np.full(14, -25.0)
            )
            np.testing.assert_allclose(
                extract_angles(reverse, handedness), 25.0, atol=1e-8
            )

    def test_filters_bone_lengths_and_kinematics(self):
        self.assertEqual(POINT_2D_FILTER, (1.0, 0.01, 1.0))
        self.assertEqual(POINT_3D_FILTER, (0.5, 0.001, 1.0))
        self.assertEqual(ANGLE_FILTER, (1.0, 0.02, 1.0))
        filter_ = OneEuro(*POINT_3D_FILTER)
        zeros, ones = np.zeros((21, 3)), np.ones((21, 3))
        np.testing.assert_array_equal(filter_(zeros, 0), zeros)
        self.assertTrue(np.all((filter_(ones, 1 / 30) > 0) & (filter_.value < 1)))
        filter_.reset()
        np.testing.assert_array_equal(filter_(ones, 1), ones)

        expected = np.array((15, 25, 20, 100, 35, 10, 80, 20, 15, 90, 30, 5, 70, 20))
        points = apply_angles(straight_hand("Left"), "Left", expected)
        model = Kinematics(calibration_frames=1)
        model.update(points, "Left", 0.0)
        corrected, phase = model.update(points, "Left", 1 / 30)
        self.assertEqual(phase, "GESTURE TRACKING")
        angles = extract_angles(corrected, "Left")
        self.assertTrue(np.all((angles >= 0) & (angles <= 180)))
        for chain in FINGER_CHAINS:
            for start, end in zip(chain[:-1], chain[1:]):
                self.assertAlmostEqual(
                    np.linalg.norm(corrected[end] - corrected[start]),
                    model.lengths[(start, end)],
                )

    def test_left_and_right_2d_filters_are_independent(self):
        left_filter = OneEuro(*POINT_2D_FILTER)
        right_filter = OneEuro(*POINT_2D_FILTER)
        self.assertIsNot(left_filter, right_filter)

        left_first = np.zeros((21, 2))
        right_first = np.full((21, 2), 10.0)
        np.testing.assert_array_equal(left_filter(left_first, 0.0), left_first)
        np.testing.assert_array_equal(right_filter(right_first, 0.0), right_first)

        left_second = left_filter(np.full((21, 2), 20.0), 1 / 30)
        right_second = right_filter(np.full((21, 2), 30.0), 1 / 30)
        self.assertTrue(np.all((left_second > 0) & (left_second < 20)))
        self.assertTrue(np.all((right_second > 10) & (right_second < 30)))
        np.testing.assert_allclose(right_second - left_second, 10.0)

    def test_handedness_geometry_and_bad_frame_hold(self):
        handedness = StableHandedness(3)
        self.assertEqual(handedness.update("Right"), "Right")
        self.assertEqual(handedness.update("Left"), "Right")
        self.assertEqual(handedness.update("Left"), "Right")
        self.assertEqual(handedness.update("Left"), "Left")

        points = np.zeros((21, 3))
        points[:, 2] = 400
        self.assertIsNone(geometry_error(points, 30))
        self.assertEqual(geometry_error(points, 31), "reprojection")
        points[1, 2] = MAX_DEPTH_MM + 1
        self.assertEqual(geometry_error(points, 1), "depth")
        points[:, 2] = 400
        points[1] = (400, 0, 400)
        self.assertEqual(geometry_error(points, 1), "hand-size")

        processor = object.__new__(StereoProcessor)
        processor.bad_frames = 0
        processor.kinematics = Kinematics(1)
        processor.left_points_filter = OneEuro(*POINT_2D_FILTER)
        processor.right_points_filter = OneEuro(*POINT_2D_FILTER)
        processor.left_points_filter(np.zeros((21, 2)), 0.0)
        processor.right_points_filter(np.ones((21, 2)), 0.0)
        processor.kinematics.points_filter(np.zeros((21, 3)), 0.0)
        processor.kinematics.angle_filter(np.zeros(14), 0.0)
        processor.last = {
            "handedness": "Right",
            "keypoint_absolute": np.zeros((21, 3)),
            "keypoint_relative": np.zeros((21, 3)),
        }
        for _ in range(3):
            result = processor._reject(processor._empty(), "detection")
            self.assertTrue(result["found"] and result["stale"])
            self.assertIsNotNone(processor.left_points_filter.value)
            self.assertIsNotNone(processor.right_points_filter.value)
        result = processor._reject(processor._empty(), "detection")
        self.assertFalse(result["found"])
        self.assertIsNone(processor.last)
        self.assertIsNone(processor.left_points_filter.value)
        self.assertIsNone(processor.right_points_filter.value)
        self.assertIsNone(processor.kinematics.points_filter.value)
        self.assertIsNone(processor.kinematics.angle_filter.value)

    def test_mediapipe_model_and_configuration(self):
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


class RetargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = RobotModel()

    def test_optimized_urdf_topology_limits_and_meshes(self):
        self.assertEqual(self.model.names, ROBOT_JOINT_NAMES)
        self.assertEqual(len(self.model.joints), 31)
        self.assertEqual(len(self.model.fk(np.zeros(21))), 32)
        self.assertTrue(np.all(self.model.lower <= self.model.upper))
        np.testing.assert_allclose(
            np.degrees(self.model.aa_neutral), MCP_AA_NEUTRAL_DEG, atol=1e-8
        )

        root = ET.parse(URDF_PATH).getroot()
        meshes = list(root.iter("mesh"))
        self.assertEqual(len(meshes), 39)
        for mesh in meshes:
            filename = mesh.get("filename")
            self.assertTrue(filename.startswith("../meshes/"))
            self.assertTrue((URDF_PATH.parent / filename).resolve().is_file())

        limits = {
            joint.get("name"): (
                float(joint.find("limit").get("lower")),
                float(joint.find("limit").get("upper")),
            )
            for joint in root.findall("joint")
            if joint.get("type") == "revolute"
        }
        np.testing.assert_array_equal(
            np.column_stack((self.model.lower, self.model.upper)),
            np.asarray([limits[name] for name in self.model.names]),
        )
        np.testing.assert_allclose(
            self.model.palm_frame.T @ self.model.palm_frame, np.eye(3), atol=1e-8
        )

    def test_retarget_is_finite_and_respects_optimized_limits(self):
        expected = np.array((20, 30, 25, 20, 40, 30, 20, 50, 30, 15, 40, 25, 10, 30))
        points = relative_points(
            apply_angles(straight_hand("Left"), "Left", expected)
        )
        retargeter = Retargeter(self.model)
        result = retargeter.solve(points)
        self.assertIsNotNone(result)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(np.all(result >= self.model.lower - 1e-10))
        self.assertTrue(np.all(result <= self.model.upper + 1e-10))
        np.testing.assert_allclose(np.degrees(result[AA]), MCP_AA_NEUTRAL_DEG)
        np.testing.assert_allclose(
            np.degrees(result[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]]),
            expected[[11, 12, 13, 8, 9, 10, 5, 6, 7, 2, 3, 4]],
        )
        tips, directions = self.model.fingertip_pads(result)
        self.assertEqual(tips.shape, (5, 3))
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1, atol=1e-8)

    def test_ros_publishes_points_and_robot_joints(self):
        class Message:
            def __init__(self):
                self.layout = None
                self.data = []

        class Dimension:
            def __init__(self, label="", size=0, stride=0):
                self.label, self.size, self.stride = label, size, stride

        class Layout:
            def __init__(self, dim=None, data_offset=0):
                self.dim, self.data_offset = dim, data_offset

        class Publisher:
            def __init__(self, topic):
                self.topic, self.messages = topic, []

            def publish(self, message):
                self.messages.append(message)

        class Node:
            def __init__(self):
                self.publishers = []
                self.destroyed = False

            def create_publisher(self, _message, topic, depth):
                self.depth = depth
                publisher = Publisher(topic)
                self.publishers.append(publisher)
                return publisher

            def destroy_node(self):
                self.destroyed = True

        node = Node()
        rclpy = types.ModuleType("rclpy")
        rclpy.init = lambda args=None: None
        rclpy.create_node = lambda _name: node
        rclpy.shutdown_called = False

        def shutdown():
            rclpy.shutdown_called = True

        rclpy.shutdown = shutdown
        messages = types.ModuleType("std_msgs.msg")
        messages.Float32MultiArray = Message
        messages.MultiArrayDimension = Dimension
        messages.MultiArrayLayout = Layout
        std_msgs = types.ModuleType("std_msgs")
        std_msgs.msg = messages

        with patch.dict(
            sys.modules,
            {"rclpy": rclpy, "std_msgs": std_msgs, "std_msgs.msg": messages},
        ):
            output = RosOutput()
            output.points(np.zeros((21, 3)), "Left")
            output.joints(np.arange(21))
            output.close()

        self.assertEqual([publisher.topic for publisher in node.publishers],
                         [KEYPOINT_TOPIC, ROBOT_TOPIC])
        point_message = node.publishers[0].messages[0]
        robot_message = node.publishers[1].messages[0]
        self.assertEqual(len(point_message.data), 63)
        self.assertEqual(len(robot_message.data), 21)
        self.assertEqual(point_message.layout.dim[0].label,
                         f"{KEYPOINT_LAYOUT}:hand=Left")
        self.assertEqual(robot_message.layout.dim[0].label, ROBOT_LAYOUT)
        self.assertTrue(node.destroyed)
        self.assertTrue(rclpy.shutdown_called)


if __name__ == "__main__":
    unittest.main()
