import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

import numpy as np

from config import (
    KEYPOINT_LAYOUT,
    KEYPOINT_TOPIC,
    PAD_DIRECTION_LAYOUT,
    PAD_DIRECTION_TOPIC,
    RETARGET_MAX_EVALUATIONS,
    ROBOT_JOINT_NAMES,
    ROBOT_LAYOUT,
    ROBOT_TOPIC,
    STANDARD_PALM_SIZE,
)
from input.frame import InputFrame, relative_points
from retarget import (
    Retargeter, RetargetWorker, RobotModel, compute_cmc_frame, human_thumb_angles,
    human_thumb_geometry,
)
from ros import RosOutput
from viewer import (
    _angle_text, _arrow_points, _human_angle_segments, _loss_text,
    _rotate as _rotate_vector,
)

PAD_DIRECTIONS = np.tile((0.0, 0.0, -1.0), (5, 1))


def hand():
    points = np.zeros((21, 3), float)
    for start, x, y in zip((5, 9, 13, 17), (-25, 0, 20, 35), (45, 50, 45, 38)):
        points[start:start + 4] = (
            (x, y, 0), (x, y + 30, 0), (x, y + 50, 0), (x, y + 65, 0)
        )
    points[1:5] = ((-30, 15, 0), (-45, 25, 0), (-58, 34, 4), (-70, 42, 8))
    return relative_points(points)


class RetargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = RobotModel()
        cls.points = hand()

    def test_robot_contract_and_fingertip_frames(self):
        model = self.model
        self.assertEqual(model.names, ROBOT_JOINT_NAMES)
        self.assertEqual(len(model.fk(model.seed)), 32)
        self.assertTrue(np.all(model.lower < model.upper))
        points, directions = model.fingertip_pads(model.seed)
        self.assertEqual((points.shape, directions.shape), ((5, 3), (5, 3)))
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1)
        arrows = _arrow_points(points, directions)
        np.testing.assert_allclose(arrows[:, 0], points)
        np.testing.assert_allclose(
            np.linalg.norm(arrows[:, 1] - points, axis=1), .025, atol=1e-8
        )

    def test_frames_and_direct_fingers(self):
        origin, rotation = compute_cmc_frame(self.points)
        np.testing.assert_allclose(origin, self.points[1])
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        local = (self.points - origin) @ rotation
        q = self.model.finger_angles(local)
        self.assertTrue(np.isfinite(q).all())
        self.assertTrue(np.all((q >= self.model.lower) & (q <= self.model.upper)))
        np.testing.assert_allclose(q[self.model.finger_joints[:, 2:]], 0, atol=1e-12)

    def test_analytic_jacobians_and_loss_gradient(self):
        q = (self.model.lower + self.model.upper) / 2
        values, jacobians = self.model.features(q)
        np.testing.assert_array_equal(values[1].ravel(), q[18:20])
        numeric = [np.empty_like(item) for item in jacobians]
        step = 1e-6
        for column, joint in enumerate(self.model.thumb):
            plus, minus = q.copy(), q.copy()
            plus[joint] += step
            minus[joint] -= step
            a, b = self.model.features(plus)[0], self.model.features(minus)[0]
            for group in range(len(values)):
                numeric[group][..., column] = (a[group] - b[group]) / (2 * step)
        for actual, expected in zip(jacobians, numeric):
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)

        retargeter = Retargeter(self.model)
        targets = retargeter._targets(
            self.points, finger_pad_directions=PAD_DIRECTIONS
        )
        loss, gradient, _ = retargeter._losses(q, targets)
        numeric = np.empty(5)
        for column, joint in enumerate(self.model.thumb):
            plus, minus = q.copy(), q.copy()
            plus[joint] += step
            minus[joint] -= step
            numeric[column] = (
                retargeter._losses(plus, targets)[0]
                - retargeter._losses(minus, targets)[0]
            ) / (2 * step)
        self.assertTrue(np.isfinite(loss))
        np.testing.assert_allclose(gradient, numeric, atol=1e-6, rtol=1e-5)

    def test_thumb_pad_loss_and_solve(self):
        target_q = self.model.seed.copy()
        thumb = self.model.thumb
        target_q[thumb] = self.model.lower[thumb] + .25 * (
            self.model.upper[thumb] - self.model.lower[thumb]
        )
        target = self.model.fingertip_pads(target_q)[1][0] @ self.model.palm_frame
        _, rotation = compute_cmc_frame(self.points)
        directions = np.tile(target @ rotation.T, (5, 1))
        with patch("retarget.C.RETARGET_THUMB_PAD_WEIGHT", 0.0):
            baseline = Retargeter(self.model).solve(self.points, 0.0, directions)
        with patch("retarget.C.RETARGET_THUMB_PAD_WEIGHT", .25):
            retargeter = Retargeter(self.model)
            aligned = retargeter.solve(self.points, 0.0, directions)
        baseline_pad = self.model.fingertip_pads(baseline)[1][0] @ self.model.palm_frame
        aligned_pad = self.model.fingertip_pads(aligned)[1][0] @ self.model.palm_frame
        self.assertGreater(aligned_pad @ target, baseline_pad @ target)
        self.assertEqual(
            tuple(retargeter.loss_terms),
            ("thumb_tip", "thumb_mcp_angle", "thumb_ip_angle",
             "thumb_to_fingertips", "thumb_pad", "total"),
        )
        self.assertAlmostEqual(
            sum(value for name, value in retargeter.loss_terms.items() if name != "total"),
            retargeter.loss_terms["total"],
        )
        self.assertIn("thumb pad", _loss_text(retargeter.loss_terms))

        with self.assertRaisesRegex(ValueError, "required"):
            Retargeter(self.model).solve(self.points)
        for bad in (np.zeros((20, 3)), np.full((21, 3), np.nan)):
            self.assertIsNone(Retargeter(self.model).solve(bad, finger_pad_directions=PAD_DIRECTIONS))

    def test_optimizer_budget_keeps_best_candidate(self):
        retargeter = Retargeter(self.model)
        evaluated = []

        def losses(q, _targets):
            evaluated.append(q.copy())
            value = float(100 - len(evaluated))
            terms = dict.fromkeys(
                ("thumb_tip", "thumb_mcp_angle", "thumb_ip_angle",
                 "thumb_to_fingertips", "thumb_pad"), 0.0
            )
            terms["total"] = value
            return value, np.zeros(5), terms

        def exhaust(fun, _x0, **_kwargs):
            for step in range(RETARGET_MAX_EVALUATIONS + 2):
                fraction = (step + 1) / (RETARGET_MAX_EVALUATIONS + 3)
                fun(self.model.lower[self.model.thumb] + fraction * (
                    self.model.upper - self.model.lower
                )[self.model.thumb])

        with patch.object(retargeter, "_losses", side_effect=losses), patch(
            "retarget.minimize", side_effect=exhaust
        ):
            result = retargeter.solve(self.points, finger_pad_directions=PAD_DIRECTIONS)
        self.assertEqual(len(evaluated), RETARGET_MAX_EVALUATIONS + 1)
        np.testing.assert_array_equal(result, evaluated[-1])

    def test_output_filter_and_pause(self):
        retargeter = Retargeter(self.model)
        first = retargeter.solve(self.points, 0.0, PAD_DIRECTIONS)
        raw = retargeter.q.copy()
        np.testing.assert_allclose(first, raw)
        moved = self.points.copy()
        moved[4] += (.02, -.01, .01)
        second = retargeter.solve(moved, 1 / 30, PAD_DIRECTIONS)
        self.assertTrue(np.all(np.abs(second - first) <= np.abs(retargeter.q - first) + 1e-12))
        retargeter.pause()
        self.assertFalse(retargeter.has_previous)
        self.assertIsNone(retargeter.loss_terms)
        np.testing.assert_array_equal(retargeter.q, self.model.seed)

    def test_worker_keeps_latest_and_copies_directions(self):
        class Fake:
            model = object()

            def __init__(self):
                self.calls = []
                self.started, self.release = threading.Event(), threading.Event()

            def solve(self, points, _timestamp, directions):
                value = int(points[0, 0])
                self.calls.append((value, None if directions is None else directions.copy()))
                if not self.release.is_set():
                    self.started.set()
                    self.release.wait(1)
                self.loss_terms = {"total": float(value)}
                return np.full(21, value, float)

            def pause(self):
                pass

        fake, worker = Fake(), None
        worker = RetargetWorker(fake)
        try:
            directions = np.ones((5, 3))
            worker.submit(np.ones((21, 3)), 1.0, directions)
            directions[:] = 9
            self.assertTrue(fake.started.wait(1))
            worker.submit(np.full((21, 3), 2), 2.0)
            worker.submit(np.full((21, 3), 3), 3.0)
            fake.release.set()
            deadline, output = time.monotonic() + 1, None
            while time.monotonic() < deadline and (output is None or output[0][0] != 3):
                output = worker.poll()
                time.sleep(.001)
            self.assertEqual([call[0] for call in fake.calls], [1, 3])
            np.testing.assert_array_equal(fake.calls[0][1], np.ones((5, 3)))
            self.assertEqual(output[1], {"total": 3.0})
        finally:
            worker.close()

    def test_ros_contract(self):
        class Message:
            def __init__(self): self.layout, self.data = None, []
        class Dimension:
            def __init__(self, label="", size=0, stride=0):
                self.label, self.size, self.stride = label, size, stride
        class Layout:
            def __init__(self, dim=None, data_offset=0): self.dim = dim
        class Publisher:
            def __init__(self, topic): self.topic, self.messages = topic, []
            def publish(self, message): self.messages.append(message)
        class Node:
            def __init__(self): self.publishers = []
            def create_publisher(self, _type, topic, _depth):
                publisher = Publisher(topic); self.publishers.append(publisher); return publisher
            def destroy_node(self): pass

        node = Node()
        rclpy = types.ModuleType("rclpy")
        rclpy.init = lambda args=None: None
        rclpy.create_node = lambda _name: node
        rclpy.shutdown = lambda: None
        messages = types.ModuleType("std_msgs.msg")
        messages.Float32MultiArray, messages.MultiArrayDimension = Message, Dimension
        messages.MultiArrayLayout = Layout
        package = types.ModuleType("std_msgs"); package.msg = messages
        with patch.dict(sys.modules, {
            "rclpy": rclpy, "std_msgs": package, "std_msgs.msg": messages,
        }):
            output = RosOutput()
            output.hand(InputFrame(
                1.0, self.points, "Left", True, "TRACKING",
                finger_pad_directions=PAD_DIRECTIONS,
            ))
            output.joints(np.arange(21))
            output.close()
        self.assertEqual(
            [publisher.topic for publisher in node.publishers],
            [KEYPOINT_TOPIC, PAD_DIRECTION_TOPIC, ROBOT_TOPIC],
        )
        sizes = [len(publisher.messages[0].data) for publisher in node.publishers]
        self.assertEqual(sizes, [63, 15, 21])
        labels = [publisher.messages[0].layout.dim[0].label for publisher in node.publishers]
        self.assertEqual(labels, [
            f"{KEYPOINT_LAYOUT}:hand=Left",
            f"{PAD_DIRECTION_LAYOUT}:hand=Left",
            ROBOT_LAYOUT,
        ])

    def test_unsigned_thumb_angles_scales_and_arcs(self):
        points = np.zeros((21, 3))
        points[1:5] = ((0, 0, 0), (1, 0, 0), (1.8, .3, -.6), (2.2, -.1, -1.4))
        angles = human_thumb_angles(points)
        self.assertTrue(np.all(angles > 0))
        self.assertTrue(np.all(angles <= np.pi))
        segments, measured_angles, axes = human_thumb_geometry(points)
        expected = np.arccos(np.clip(np.sum(segments[:-1] * segments[1:], axis=1), -1, 1))
        np.testing.assert_allclose(np.abs(measured_angles), expected)
        np.testing.assert_allclose(
            [_rotate_vector(first, axis, angle)
             for first, axis, angle in zip(segments[:-1], axes, measured_angles)],
            segments[1:], atol=1e-7,
        )
        rotation = np.array(((0, -1, 0), (1, 0, 0), (0, 0, 1)), float)
        np.testing.assert_allclose(
            human_thumb_angles(3 * points @ rotation.T + 7), angles,
        )
        np.testing.assert_allclose(human_thumb_angles(points * (-1, 1, 1)), angles)
        measured, arcs = _human_angle_segments(points)
        np.testing.assert_allclose(measured, angles)
        self.assertEqual((arcs[0].shape, arcs[1].shape), ((26, 2, 3), (26, 2, 3)))
        for index, arc in enumerate(arcs):
            origin = points[index + 2]
            np.testing.assert_allclose(
                (arc[0, 1] - origin) / np.linalg.norm(arc[0, 1] - origin),
                segments[index], atol=1e-7,
            )
            np.testing.assert_allclose(
                (arc[1, 1] - origin) / np.linalg.norm(arc[1, 1] - origin),
                segments[index + 1], atol=2e-7,
            )
            np.testing.assert_allclose(arc[2, 0], arc[0, 1], atol=1e-7)
            np.testing.assert_allclose(arc[-1, 1], arc[1, 1], atol=1e-7)
        self.assertNotIn("-", _angle_text("Human thumb", ("MCP", "IP"), angles))

        straight = points.copy()
        straight[1:5] = ((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0))
        straight_angles, straight_arcs = _human_angle_segments(straight)
        np.testing.assert_array_equal(straight_angles, 0)
        self.assertTrue(np.isfinite(straight_arcs).all())

        retargeter = Retargeter(self.model)
        targets = retargeter._targets(self.points, finger_pad_directions=PAD_DIRECTIONS)
        q = self.model.seed.copy()
        with patch("retarget.C.RETARGET_THUMB_MCP_ANGLE_SCALE", 2.0), patch(
            "retarget.C.RETARGET_THUMB_IP_ANGLE_SCALE", .5
        ), patch(
            "retarget.C.RETARGET_THUMB_MCP_ANGLE_WEIGHT", 20.0
        ), patch(
            "retarget.C.RETARGET_THUMB_IP_ANGLE_WEIGHT", 20.0
        ):
            terms = retargeter._losses(q, targets)[2]
        self.assertAlmostEqual(
            terms["thumb_mcp_angle"], 20 * (q[18] - 2 * targets[1][0, 0]) ** 2
        )
        self.assertAlmostEqual(
            terms["thumb_ip_angle"], 20 * (q[19] - .5 * targets[1][1, 0]) ** 2
        )
        with patch("retarget.C.RETARGET_THUMB_MCP_ANGLE_SCALE", -1):
            with self.assertRaisesRegex(ValueError, "scales"):
                Retargeter(self.model)

    def test_thumb_to_fingertips_loss(self):
        retargeter = Retargeter(self.model)
        targets = retargeter._targets(
            self.points, finger_pad_directions=PAD_DIRECTIONS
        )
        local = (self.points - compute_cmc_frame(self.points)[0]) @ compute_cmc_frame(
            self.points
        )[1]
        np.testing.assert_allclose(
            targets[2], local[[8, 12, 16, 20]] - local[4]
        )
        q = self.model.seed.copy()
        values = self.model.features(q)[0]
        self.assertEqual(values[2].shape, (4, 3))
        expected = np.mean(np.sum((values[2] - targets[2]) ** 2, axis=1)) / (
            STANDARD_PALM_SIZE ** 2
        )
        with patch("retarget.C.RETARGET_THUMB_TO_FINGERTIPS_WEIGHT", 1.0):
            enabled_loss, enabled_gradient, enabled = retargeter._losses(q, targets)
        self.assertAlmostEqual(enabled["thumb_to_fingertips"], expected)
        with patch("retarget.C.RETARGET_THUMB_TO_FINGERTIPS_WEIGHT", 0.0):
            disabled_loss, disabled_gradient, disabled = retargeter._losses(q, targets)
        self.assertEqual(disabled["thumb_to_fingertips"], 0.0)
        relative_gradient = retargeter._term(
            values[2], self.model.features(q)[1][2], targets[2], 1.0, 1.0,
            STANDARD_PALM_SIZE,
        )[1]
        self.assertAlmostEqual(enabled_loss - disabled_loss, expected)
        np.testing.assert_allclose(
            enabled_gradient - disabled_gradient, relative_gradient, atol=1e-12
        )
        self.assertIn("thumb-fingertips", _loss_text(enabled))


if __name__ == "__main__":
    unittest.main()
