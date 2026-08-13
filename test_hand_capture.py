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
    ROBOT_JOINT_NAMES,
    ROBOT_LAYOUT,
    ROBOT_TOPIC,
)
from input.frame import InputFrame, relative_hand
from retarget import (
    Retargeter, RetargetWorker, RobotModel, compute_cmc_frame,
)
from ros import RosOutput
from viewer import _arrow_points, _loss_text

PAD_DIRECTIONS = np.tile((0.0, 0.0, -1.0), (5, 1))


def hand():
    points = np.zeros((21, 3), float)
    for start, x, y in zip((5, 9, 13, 17), (-25, 0, 20, 35), (45, 50, 45, 38)):
        points[start:start + 4] = (
            (x, y, 0), (x, y + 30, 0), (x, y + 50, 0), (x, y + 65, 0)
        )
    points[1:5] = ((-30, 15, 0), (-45, 25, 0), (-58, 34, 4), (-70, 42, 8))
    return relative_hand(points)[0]


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
            baseline, _ = Retargeter(self.model).solve(self.points, 0.0, directions)
        with patch("retarget.C.RETARGET_THUMB_PAD_WEIGHT", .25):
            retargeter = Retargeter(self.model)
            aligned, losses = retargeter.solve(self.points, 0.0, directions)
        baseline_pad = self.model.fingertip_pads(baseline)[1][0] @ self.model.palm_frame
        aligned_pad = self.model.fingertip_pads(aligned)[1][0] @ self.model.palm_frame
        self.assertGreater(aligned_pad @ target, baseline_pad @ target)
        self.assertEqual(
            tuple(losses),
            ("thumb_tip", "thumb_mcp_angle", "thumb_ip_angle",
             "thumb_to_fingertips", "thumb_pad", "total"),
        )
        self.assertAlmostEqual(
            sum(value for name, value in losses.items() if name != "total"),
            losses["total"],
        )
        self.assertIn("thumb pad", _loss_text(losses))

        with self.assertRaisesRegex(ValueError, "required"):
            Retargeter(self.model).solve(self.points)
        for bad in (np.zeros((20, 3)), np.full((21, 3), np.nan)):
            self.assertIsNone(Retargeter(self.model).solve(bad, finger_pad_directions=PAD_DIRECTIONS))

    def test_output_filter_and_pause(self):
        retargeter = Retargeter(self.model)
        first, _ = retargeter.solve(self.points, 0.0, PAD_DIRECTIONS)
        raw = retargeter.q.copy()
        np.testing.assert_allclose(first, raw)
        moved = self.points.copy()
        moved[4] += (.02, -.01, .01)
        second, _ = retargeter.solve(moved, 1 / 30, PAD_DIRECTIONS)
        self.assertTrue(np.all(np.abs(second - first) <= np.abs(retargeter.q - first) + 1e-12))
        retargeter.pause()
        self.assertFalse(retargeter.has_previous)
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
                return np.full(21, value, float), {"total": float(value)}

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

if __name__ == "__main__":
    unittest.main()
