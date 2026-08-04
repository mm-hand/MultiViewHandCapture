import sys
import threading
import time
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
    MP_DETECTION_CONFIDENCE,
    MP_PRESENCE_CONFIDENCE,
    MP_TRACKING_CONFIDENCE,
    POINT_2D_FILTER,
    POINT_3D_FILTER,
    RETARGET_FINGER_SHAPE_WEIGHT,
    RETARGET_FTOL,
    RETARGET_MAX_EVALUATIONS,
    RETARGET_MIDPOINT_WEIGHT,
    RETARGET_PALM_TIPS_SCALE,
    RETARGET_PALM_TIPS_WEIGHT,
    RETARGET_TEMPORAL_WEIGHT,
    RETARGET_THUMB_FINGERTIPS_SCALE,
    RETARGET_THUMB_FINGERTIPS_WEIGHT,
    RETARGET_THUMB_PAD_WEIGHT,
    RETARGET_THUMB_SHAPE_WEIGHT,
    ROBOT_JOINT_NAMES,
    ROBOT_LAYOUT,
    ROBOT_TOPIC,
    THUMB_PAD_AXIS,
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
from retarget import (
    Retargeter,
    RetargetWorker,
    RobotModel,
    human_palm_frame,
    human_retarget_points,
)
from track import RosOutput, _frame_wxyz, _parse_args


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

    def test_tracking_frame_stays_at_wrist_and_retarget_frame_uses_cmc(self):
        local_hands = []
        for handedness in ("Left", "Right"):
            tracked = relative_points(straight_hand(handedness))
            original = tracked.copy()
            origin, frame = human_palm_frame(tracked)
            local = human_retarget_points(tracked)

            np.testing.assert_allclose(tracked[0], 0, atol=1e-12)
            np.testing.assert_allclose(origin, tracked[1], atol=1e-12)
            np.testing.assert_allclose(local[1], 0, atol=1e-12)
            np.testing.assert_allclose(tracked, original, atol=0)
            np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(frame), 1.0)
            self.assertGreater(local[9, 0] - local[0, 0], 0)
            np.testing.assert_allclose(local[9, 1:] - local[0, 1:], 0, atol=1e-12)
            self.assertGreater(local[17, 1] - local[5, 1], 0)
            self.assertEqual(tuple(_frame_wxyz(np.eye(3))), (1.0, 0.0, 0.0, 0.0))
            local_hands.append(local)
        np.testing.assert_allclose(local_hands[0], local_hands[1], atol=1e-12)

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

    def test_original_urdf_topology_limits_and_meshes(self):
        self.assertEqual(self.model.names, ROBOT_JOINT_NAMES)
        self.assertEqual(len(self.model.joints), 31)
        self.assertEqual(len(self.model.fk(np.zeros(21))), 32)
        self.assertTrue(np.all(self.model.lower <= self.model.upper))

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
        expected_limits = {
            "Thumb_CMC": (0.0, 1.343904),
            "Ring_MCP_AA": (0.0, 1.064651),
            "Middle_MCP_AA": (0.0, 1.029744),
            "Index_MCP_AA": (0.0, 1.012291),
            "Little_MCP_AA": (0.0, 0.942478),
            "Thumb_MCP_AA": (0.0, 0.994838),
        }
        for name, expected in expected_limits.items():
            np.testing.assert_allclose(limits[name], expected, atol=0)
            np.testing.assert_allclose(
                self.model.joints[name]["origin"][:3, :3], np.eye(3), atol=1e-12
            )
        np.testing.assert_array_equal(self.model.seed, np.zeros(21))
        np.testing.assert_allclose(self.model.palm_frame, np.eye(3), atol=1e-12)

        reference = self.model.seed.copy()
        transforms = self.model.fk(reference)
        np.testing.assert_allclose(
            self.model.palm_frame, transforms["base_link"][:3, :3], atol=1e-12
        )
        cmc = self.model.joints["Thumb_CMC"]
        cmc_frame = transforms[cmc["parent"]] @ cmc["origin"]
        cmc_origin = cmc_frame[:3, 3]
        axis = cmc_frame[:3, :3] @ cmc["axis"]
        mcp = transforms[self.model.joints["Thumb_MCP_AA"]["child"]][:3, 3]
        expected = cmc_origin + axis * np.dot(mcp - cmc_origin, axis)
        np.testing.assert_allclose(self.model.palm_position, expected, atol=1e-10)
        np.testing.assert_allclose(
            np.cross(self.model.palm_position - cmc_origin, axis), 0, atol=1e-10
        )
        for angle in self.model.lower[20], self.model.upper[20]:
            reference[20] = angle
            transforms = self.model.fk(reference)
            mcp = transforms[self.model.joints["Thumb_MCP_AA"]["child"]][:3, 3]
            projection = cmc_origin + axis * np.dot(mcp - cmc_origin, axis)
            np.testing.assert_allclose(projection, self.model.palm_position, atol=1e-10)

    def test_analytic_retarget_jacobians_and_loss_gradients(self):
        q = (self.model.lower + self.model.upper) / 2
        values, jacobians = self.model.features(q, True)
        numeric = [np.empty_like(jacobian) for jacobian in jacobians]
        step = 1e-6
        for joint in range(len(q)):
            plus, minus = q.copy(), q.copy()
            plus[joint] += step
            minus[joint] -= step
            plus_values, minus_values = self.model.features(plus), self.model.features(minus)
            for group in range(len(values)):
                numeric[group][..., joint] = (
                    plus_values[group] - minus_values[group]
                ) / (2 * step)
        for actual, expected in zip(jacobians, numeric):
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)

        points = relative_points(straight_hand("Left"))
        retargeter = Retargeter(self.model)
        targets = retargeter._targets(points)
        losses = retargeter._losses(q, targets)
        for loss_index, gradient_index in ((0, 1), (2, 3)):
            gradient = np.empty(len(q))
            for joint in range(len(q)):
                plus, minus = q.copy(), q.copy()
                plus[joint] += step
                minus[joint] -= step
                gradient[joint] = (
                    retargeter._losses(plus, targets)[loss_index]
                    - retargeter._losses(minus, targets)[loss_index]
                ) / (2 * step)
            np.testing.assert_allclose(
                losses[gradient_index], gradient, atol=1e-6, rtol=1e-5
            )

    def test_thumb_pad_target_and_joint_regularizers(self):
        retargeter = Retargeter(self.model)
        targets = retargeter._targets(relative_points(straight_hand("Left")))
        distances = np.linalg.norm(targets[0][1:] - targets[0][0], axis=1)
        np.testing.assert_array_equal(
            targets[4], np.argsort(distances, kind="stable")[:2] + 1
        )
        self.assertAlmostEqual(np.linalg.norm(self.model.features(self.model.seed)[2]), 1)
        midpoint = self.model.midpoint
        without_previous = retargeter._losses(midpoint, targets)
        with_previous = retargeter._losses(midpoint, targets, self.model.lower)
        self.assertEqual(without_previous[0], with_previous[0])
        np.testing.assert_array_equal(without_previous[1], with_previous[1])
        self.assertAlmostEqual(
            with_previous[2] - without_previous[2], RETARGET_TEMPORAL_WEIGHT / 4
        )
        midpoint_loss, midpoint_gradient = retargeter._joint_term(
            midpoint, midpoint, RETARGET_MIDPOINT_WEIGHT
        )
        self.assertEqual(midpoint_loss, 0)
        np.testing.assert_array_equal(midpoint_gradient, np.zeros(21))
        lower_loss, _ = retargeter._joint_term(
            self.model.lower, midpoint, RETARGET_MIDPOINT_WEIGHT
        )
        self.assertAlmostEqual(lower_loss, RETARGET_MIDPOINT_WEIGHT / 4)

    def test_single_stage_retarget_and_limits(self):
        expected = np.array((20, 30, 25, 20, 40, 30, 20, 50, 30, 15, 40, 25, 10, 30))
        points = relative_points(
            apply_angles(straight_hand("Left"), "Left", expected)
        )
        retargeter = Retargeter(self.model)
        targets = retargeter._targets(points)
        local = human_retarget_points(points)
        np.testing.assert_allclose(points[0], 0, atol=1e-12)
        np.testing.assert_allclose(targets[0], local[[4, 8, 12, 16, 20]])
        self.assertEqual(
            (
                RETARGET_PALM_TIPS_SCALE,
                RETARGET_THUMB_FINGERTIPS_SCALE,
                RETARGET_PALM_TIPS_WEIGHT,
                RETARGET_THUMB_FINGERTIPS_WEIGHT,
                RETARGET_THUMB_SHAPE_WEIGHT,
                RETARGET_FINGER_SHAPE_WEIGHT,
                RETARGET_THUMB_PAD_WEIGHT,
                RETARGET_TEMPORAL_WEIGHT,
                RETARGET_MIDPOINT_WEIGHT,
            ),
            (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.1, 0.01),
        )
        self.assertEqual(THUMB_PAD_AXIS, (-0.200671, 0.970119, -0.136380))
        self.assertEqual((RETARGET_FTOL, RETARGET_MAX_EVALUATIONS), (3e-5, 20))
        self.assertFalse(retargeter.has_previous)
        result = retargeter.solve(points)
        self.assertIsNotNone(result)
        self.assertTrue(retargeter.has_previous)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(np.all(result >= self.model.lower - 1e-10))
        self.assertTrue(np.all(result <= self.model.upper + 1e-10))
        self.assertEqual(len(retargeter.losses), 2)
        self.assertLessEqual(retargeter.losses[0], retargeter.losses[1])
        for angles in (
            np.zeros(14),
            np.array((45, 55, 80, 90, 75, 80, 90, 75, 80, 90, 75, 80, 90, 75)),
        ):
            pose = relative_points(
                apply_angles(straight_hand("Left"), "Left", angles)
            )
            candidate = retargeter.solve(pose)
            self.assertIsNotNone(candidate)
            self.assertTrue(np.all(candidate >= self.model.lower - 1e-10))
            self.assertTrue(np.all(candidate <= self.model.upper + 1e-10))
        self.assertIsNone(retargeter.solve(np.zeros((20, 3))))
        degenerate = points.copy()
        degenerate[9] = degenerate[0]
        self.assertIsNone(retargeter.solve(degenerate))
        retargeter.pause()
        np.testing.assert_array_equal(retargeter.q, self.model.seed)
        self.assertFalse(retargeter.has_previous)
        self.assertIsNone(retargeter.losses)

    def test_single_stage_calls_slsqp_once_and_keeps_state_on_failure(self):
        points = relative_points(straight_hand("Left"))
        retargeter = Retargeter(self.model)
        calls = []

        def fake_minimize(fun, x0, **kwargs):
            calls.append(kwargs)
            loss, gradient = fun(x0)
            self.assertTrue(np.isfinite(loss) and np.isfinite(gradient).all())
            return types.SimpleNamespace(success=True, x=np.asarray(x0).copy())

        with patch("retarget.minimize", side_effect=fake_minimize):
            result = retargeter.solve(points)
        np.testing.assert_array_equal(result, self.model.seed)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("constraints", calls[0])
        self.assertNotIn("maxiter", calls[0]["options"])
        previous, losses = retargeter.q.copy(), retargeter.losses
        failure = types.SimpleNamespace(success=False, x=np.full(21, np.nan))
        with patch("retarget.minimize", return_value=failure):
            self.assertIsNone(retargeter.solve(points))
        np.testing.assert_array_equal(retargeter.q, previous)
        self.assertEqual(retargeter.losses, losses)

    def test_retarget_stops_after_evaluation_budget_with_best_candidate(self):
        points = relative_points(straight_hand("Left"))
        retargeter = Retargeter(self.model)
        evaluated = []

        def fake_losses(q, _targets, _previous):
            evaluated.append(np.asarray(q).copy())
            loss = float(100 - len(evaluated))
            return loss, np.zeros(21), loss, np.zeros(21)

        def exhaust(fun, _x0, **_kwargs):
            for step in range(1, RETARGET_MAX_EVALUATIONS + 2):
                fraction = step / (RETARGET_MAX_EVALUATIONS + 2)
                fun(self.model.lower + fraction * self.model.joint_range)
            self.fail("Evaluation budget was not enforced")

        with patch.object(retargeter, "_losses", side_effect=fake_losses), \
             patch("retarget.minimize", side_effect=exhaust):
            result = retargeter.solve(points)
        self.assertEqual(len(evaluated), RETARGET_MAX_EVALUATIONS)
        np.testing.assert_array_equal(result, evaluated[-1])
        self.assertEqual(retargeter.losses, (80.0, 80.0))

    def test_retarget_worker_keeps_latest_frame_and_discards_paused_work(self):
        class FakeRetargeter:
            model = object()

            def __init__(self):
                self.calls, self.pauses = [], 0
                self.started, self.release = threading.Event(), threading.Event()

            def solve(self, points):
                value = int(points[0, 0])
                self.calls.append(value)
                if not self.release.is_set():
                    self.started.set()
                    self.release.wait(1)
                return np.full(21, value, float)

            def pause(self):
                self.pauses += 1

        fake, worker = FakeRetargeter(), None

        def wait_for(expected):
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                result = worker.poll()
                if result is not None and result[0] == expected:
                    return result
                time.sleep(0.001)
            self.fail(f"Timed out waiting for retarget result {expected}")

        worker = RetargetWorker(fake)
        try:
            worker.submit(np.full((21, 3), 1))
            self.assertTrue(fake.started.wait(1))
            worker.submit(np.full((21, 3), 2))
            worker.submit(np.full((21, 3), 3))
            fake.release.set()
            np.testing.assert_array_equal(wait_for(3), np.full(21, 3))
            self.assertEqual(fake.calls, [1, 3])
            self.assertIsNone(worker.poll())

            fake.started.clear()
            fake.release.clear()
            worker.submit(np.full((21, 3), 4))
            self.assertTrue(fake.started.wait(1))
            worker.pause()
            worker.submit(np.full((21, 3), 5))
            fake.release.set()
            np.testing.assert_array_equal(wait_for(5), np.full(21, 5))
            self.assertEqual(fake.calls, [1, 3, 4, 5])
            self.assertEqual(fake.pauses, 1)
        finally:
            worker.close()

        def fail(_points):
            raise ValueError("broken")

        broken = types.SimpleNamespace(model=None, pause=lambda: None, solve=fail)
        worker = RetargetWorker(broken)
        try:
            worker.submit(np.zeros((21, 3)))
            worker.thread.join(1)
            with self.assertRaisesRegex(RuntimeError, "Retarget worker failed"):
                worker.poll()
        finally:
            worker.close()

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
