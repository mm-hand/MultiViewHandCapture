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
from input.frame import (
    InitialJointAngles, InputFrame, initial_joint_angles_from_points, relative_hand,
)
from retarget import (
    Retargeter, RetargetWorker, RobotModel, compute_cmc_frame,
)
from ros import RosOutput
from viewer import (
    _arrow_points, _latency_text, _loss_text, _render_time_text,
    _timing_breakdown_text,
)

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
        cls.angles = initial_joint_angles_from_points(cls.points)

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

    def test_initial_joint_angle_contract(self):
        copied = self.angles.copy()
        self.assertIsNot(copied.four_fingers, self.angles.four_fingers)
        self.assertIsNot(copied.thumb_bends, self.angles.thumb_bends)
        for four, thumb in (
            (np.zeros((3, 4)), np.zeros(2)),
            (np.zeros((4, 4)), np.zeros(3)),
            (np.full((4, 4), np.nan), np.zeros(2)),
        ):
            with self.assertRaises(ValueError):
                InitialJointAngles(four, thumb)
        with self.assertRaisesRegex(ValueError, "require initial"):
            InputFrame(
                0.0, self.points, "Left", True, "TRACKING",
                finger_pad_directions=PAD_DIRECTIONS,
            )

    def test_frames_and_direct_fingers(self):
        origin, rotation = compute_cmc_frame(self.points)
        np.testing.assert_allclose(origin, self.points[1])
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        q = self.model.initial_angle_targets(self.angles)
        self.assertTrue(np.isfinite(q).all())
        self.assertTrue(np.all((q >= self.model.lower) & (q <= self.model.upper)))
        indices = self.model.finger_joints
        np.testing.assert_allclose(
            q[indices[:, 1:]], self.model.lower[indices[:, 1:]], atol=1e-12
        )

    def test_analytic_jacobians_and_loss_gradient(self):
        q = (self.model.lower + self.model.upper) / 2
        values, jacobians = self.model.features(q)
        np.testing.assert_allclose(values[1].ravel(), q[18:20])
        np.testing.assert_array_equal(jacobians[1][:, 0, 18:20], np.eye(2))
        numeric = [np.empty_like(item) for item in jacobians]
        step = 1e-6
        for column, joint in enumerate(range(len(q))):
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
            self.points, finger_pad_directions=PAD_DIRECTIONS,
            initial_joint_angles=self.angles,
        )
        loss, gradient, _ = retargeter._losses(q, targets)
        numeric = np.empty(len(q))
        for column, joint in enumerate(range(len(q))):
            plus, minus = q.copy(), q.copy()
            plus[joint] += step
            minus[joint] -= step
            numeric[column] = (
                retargeter._losses(plus, targets)[0]
                - retargeter._losses(minus, targets)[0]
            ) / (2 * step)
        self.assertTrue(np.isfinite(loss))
        np.testing.assert_allclose(gradient, numeric, atol=1e-6, rtol=1e-5)

    def test_joint_initial_guess_and_independent_targets(self):
        retargeter = Retargeter(self.model)
        cold = retargeter._targets(
            self.points, finger_pad_directions=PAD_DIRECTIONS,
            initial_joint_angles=self.angles,
        )[5]
        np.testing.assert_allclose(
            cold[18:20],
            np.clip(self.angles.thumb_bends, self.model.lower[18:20], self.model.upper[18:20]),
        )
        retargeter.q[self.model.thumb] = .5 * (
            self.model.lower[self.model.thumb] + self.model.upper[self.model.thumb]
        )
        retargeter.has_previous = True
        targets = retargeter._targets(
            self.points, retargeter.q, PAD_DIRECTIONS, self.angles
        )
        initial = targets[5]
        np.testing.assert_array_equal(
            initial[self.model.finger_joints.ravel()], targets[4]
        )
        np.testing.assert_array_equal(initial[[16, 17, 20]], retargeter.q[[16, 17, 20]])
        np.testing.assert_allclose(initial[18:20], cold[18:20])
        _, jacobians = self.model.features(initial)
        relative_jacobian = jacobians[2]
        self.assertTrue(np.any(relative_jacobian[..., :16]))
        self.assertTrue(np.any(relative_jacobian[..., 16:]))

        before = retargeter._losses(initial, targets)[2]
        retargeter.solve(self.points, 0.0, PAD_DIRECTIONS, self.angles)
        after = retargeter._losses(retargeter.q, targets)[2]
        self.assertLess(after["total"], before["total"])
        self.assertLess(after["fingertip_vectors"], before["fingertip_vectors"])

        q = initial.copy()
        q[self.model.finger_joints.ravel()] += .05
        q = np.clip(q, self.model.lower, self.model.upper)
        with patch("retarget.C.RETARGET_FINGER_ANGLE_WEIGHT", 0.0):
            _, no_angles, terms = retargeter._losses(q, targets)
        with patch("retarget.C.RETARGET_FINGER_ANGLE_WEIGHT", 1.0):
            _, with_angles, _ = retargeter._losses(q, targets)
        self.assertEqual(terms["finger_angles"], 0)
        np.testing.assert_allclose(
            (with_angles - no_angles)[self.model.thumb], 0, atol=1e-15
        )
        self.assertTrue(np.any(with_angles[:16] != no_angles[:16]))

        with patch("retarget.C.RETARGET_FINGERTIP_VECTOR_WEIGHT", 0.0):
            _, no_vectors, terms = retargeter._losses(q, targets)
        with patch("retarget.C.RETARGET_FINGERTIP_VECTOR_WEIGHT", 2.0):
            _, with_vectors, _ = retargeter._losses(q, targets)
        self.assertEqual(terms["fingertip_vectors"], 0)
        vector_gradient = with_vectors - no_vectors
        self.assertTrue(np.any(vector_gradient[:16]))
        self.assertTrue(np.any(vector_gradient[16:]))

    def test_four_finger_flexion_uses_urdf_lower_as_human_zero(self):
        index = self.model.finger_joints[0]
        four = self.angles.four_fingers.copy()
        four[0] = (0.0, 0.0, np.radians(30), np.radians(40))
        angles = InitialJointAngles(four, self.angles.thumb_bends)
        mapped = self.model.initial_angle_targets(angles)
        self.assertAlmostEqual(mapped[index[1]], self.model.lower[index[1]])
        self.assertAlmostEqual(
            mapped[index[2]], self.model.lower[index[2]] + np.radians(30)
        )
        self.assertAlmostEqual(
            mapped[index[3]], self.model.lower[index[3]] + np.radians(40)
        )

    def test_thumb_bends_are_independent_and_clipped(self):
        retargeter = Retargeter(self.model)
        angles = InitialJointAngles(
            self.angles.four_fingers, np.radians((30.0, 400.0))
        )
        mapped = self.model.initial_angle_targets(angles)
        self.assertAlmostEqual(mapped[18], np.radians(30.0))
        self.assertAlmostEqual(mapped[19], self.model.upper[19])
        targets = retargeter._targets(
            self.points, finger_pad_directions=PAD_DIRECTIONS,
            initial_joint_angles=angles,
        )
        np.testing.assert_allclose(
            targets[1].ravel(), mapped[18:20]
        )
        changed = InitialJointAngles(
            self.angles.four_fingers, np.radians((45.0, 400.0))
        )
        changed_target = retargeter._targets(
            self.points, finger_pad_directions=PAD_DIRECTIONS,
            initial_joint_angles=changed,
        )[1].ravel()
        self.assertNotEqual(changed_target[0], targets[1].ravel()[0])
        self.assertEqual(changed_target[1], targets[1].ravel()[1])

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
            baseline, _ = Retargeter(self.model).solve(
                self.points, 0.0, directions, self.angles
            )
        with patch("retarget.C.RETARGET_THUMB_PAD_WEIGHT", .25):
            retargeter = Retargeter(self.model)
            aligned, losses = retargeter.solve(
                self.points, 0.0, directions, self.angles
            )
        baseline_pad = self.model.fingertip_pads(baseline)[1][0] @ self.model.palm_frame
        aligned_pad = self.model.fingertip_pads(aligned)[1][0] @ self.model.palm_frame
        self.assertGreater(aligned_pad @ target, baseline_pad @ target)
        self.assertEqual(
            tuple(losses),
            ("thumb_tip", "thumb_proximal_bend", "thumb_distal_bend",
             "finger_angles", "fingertip_vectors", "thumb_pad",
             "finger_pads", "total"),
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

    def test_four_finger_pad_residual_has_shared_non_thumb_gradient(self):
        target_q = self.model.seed.copy()
        indices = self.model.finger_joints.ravel()
        target_q[indices] = self.model.lower[indices] + .3 * (
            self.model.upper[indices] - self.model.lower[indices]
        )
        target_pads = self.model.fingertip_pads(target_q)[1] @ self.model.palm_frame
        _, rotation = compute_cmc_frame(self.points)
        directions = target_pads @ rotation.T
        retargeter = Retargeter(self.model)
        targets = retargeter._targets(
            self.points, finger_pad_directions=directions,
            initial_joint_angles=self.angles,
        )
        with patch("retarget.C.RETARGET_FINGER_PAD_WEIGHT", 0.0):
            _, without, terms = retargeter._losses(self.model.seed, targets)
        with patch("retarget.C.RETARGET_FINGER_PAD_WEIGHT", 2.0):
            _, with_pads, terms_with_pads = retargeter._losses(
                self.model.seed, targets
            )
        self.assertEqual(terms["finger_pads"], 0.0)
        self.assertGreater(terms_with_pads["finger_pads"], 0.0)
        gradient = with_pads - without
        self.assertTrue(np.any(np.abs(gradient[:16]) > 1e-12))
        np.testing.assert_allclose(gradient[16:], 0.0, atol=1e-15)

    def test_output_filter_and_pause(self):
        retargeter = Retargeter(self.model)
        first, _ = retargeter.solve(self.points, 0.0, PAD_DIRECTIONS, self.angles)
        raw = retargeter.q.copy()
        np.testing.assert_allclose(first, raw)
        moved = self.points.copy()
        moved[4] += (.02, -.01, .01)
        moved_angles = initial_joint_angles_from_points(moved)
        second, _ = retargeter.solve(moved, 1 / 30, PAD_DIRECTIONS, moved_angles)
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

            def solve(self, points, _timestamp, directions, initial_angles):
                value = int(points[0, 0])
                self.calls.append((
                    value,
                    None if directions is None else directions.copy(),
                    None if initial_angles is None else initial_angles.copy(),
                ))
                if not self.release.is_set():
                    self.started.set()
                    self.release.wait(1)
                return np.full(21, value, float), {"total": float(value)}

            def pause(self):
                pass

        fake, worker = Fake(), None
        worker = RetargetWorker(fake, clock=lambda: 3.025)
        try:
            directions = np.ones((5, 3))
            angles = self.angles.copy()
            worker.submit(np.ones((21, 3)), 1.0, directions, angles)
            directions[:] = 9
            angles.four_fingers[:] = 9
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
            np.testing.assert_array_equal(
                fake.calls[0][2].four_fingers, self.angles.four_fingers
            )
            self.assertEqual(output[1], {"total": 3.0})
            self.assertAlmostEqual(output[2], 0.0)
            self.assertEqual(output[3]["solve_total"], 0.0)
        finally:
            worker.close()

    def test_latency_text_contract(self):
        self.assertEqual(_latency_text(None), "waiting")
        self.assertEqual(_latency_text(float("nan")), "waiting")
        self.assertEqual(_latency_text(-1.0), "waiting")
        self.assertEqual(_latency_text(12.34), "12.3 ms")
        self.assertEqual(_render_time_text(None), "render waiting")
        self.assertEqual(_render_time_text(0.26), "render 0.3 ms")
        self.assertIn("SLSQP           3.2 ms", _timing_breakdown_text({"slsqp": 3.2}))

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
                initial_joint_angles=self.angles,
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
