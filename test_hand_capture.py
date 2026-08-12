import sys
import threading
import time
import types
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

import numpy as np

from input.frame import InputFrame, relative_points
from config import (
    KEYPOINT_LAYOUT,
    KEYPOINT_TOPIC,
    PAD_DIRECTION_LAYOUT,
    PAD_DIRECTION_TOPIC,
    RETARGET_MAX_EVALUATIONS,
    ROBOT_JOINT_NAMES,
    ROBOT_LAYOUT,
    ROBOT_TOPIC,
    URDF_PATH,
)
from retarget import (
    ROBOT_FINGERS,
    ROBOT_TIPS,
    Retargeter,
    RetargetWorker,
    RobotModel,
    human_palm_frame,
    human_retarget_points,
)
from ros import RosOutput
from viewer import Viewer, _arrow_points, _frame_wxyz, _loss_text


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


def _unit(vector):
    return vector / max(np.linalg.norm(vector), 1e-9)


def _rotate(vector, axis, degrees):
    angle, axis = np.radians(degrees), _unit(axis)
    return (vector * np.cos(angle) + np.cross(axis, vector) * np.sin(angle)
            + axis * np.dot(axis, vector) * (1 - np.cos(angle)))


def _project(vector, axis, fallback):
    projected = vector - axis * np.dot(vector, axis)
    return _unit(projected) if np.linalg.norm(projected) > 1e-9 else _unit(fallback)


def _angle(start, end, axis):
    start, end = _project(start, axis, start), _project(end, axis, start)
    return np.degrees(np.arctan2(np.dot(np.cross(start, end), axis),
                                 np.clip(np.dot(start, end), -1, 1)))


def _finger_frames(points, handedness):
    forward = _unit(points[9] - points[0])
    normal = _unit(np.cross(_unit(points[5] - points[17]), forward))
    normal = -normal if handedness == "Left" else normal
    parent = _unit(points[2] - points[1])
    palmward = points[9] - points[2]
    palmward -= parent * np.dot(palmward, parent)
    frames = [((2, 3, 4), parent, _unit(np.cross(parent, _unit(palmward))))]
    for mcp in (5, 9, 13, 17):
        proximal = _unit(points[mcp + 1] - points[mcp])
        neutral = _project(proximal, normal, points[mcp] - points[0])
        frames.append(((mcp, mcp + 1, mcp + 2, mcp + 3), neutral,
                       _unit(np.cross(neutral, normal))))
    return frames


def apply_angles(points, handedness, angles):
    result, offset = points.copy(), 0
    for chain, direction, axis in _finger_frames(points, handedness):
        for position, (start, end) in enumerate(zip(chain[:-1], chain[1:])):
            measured = _project(result[end] - result[start], axis, direction)
            change = angles[offset] - _angle(direction, measured, axis)
            origin = result[start].copy()
            for child in chain[position + 1:]:
                result[child] = origin + _rotate(result[child] - origin, axis, change)
            direction = _project(result[end] - result[start], axis, direction)
            offset += 1
    return result


class RetargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = RobotModel()

    def test_latest_urdf_topology_limits_meshes_and_tips(self):
        self.assertEqual(self.model.names, ROBOT_JOINT_NAMES)
        self.assertEqual(len(self.model.joints), 31)
        self.assertEqual(len(self.model.fk(np.zeros(21))), 32)
        self.assertTrue(np.all(self.model.lower <= self.model.upper))

        root = ET.parse(URDF_PATH).getroot()
        self.assertEqual(len(root.findall("link")), 32)
        self.assertEqual(len(root.findall("joint")), 31)
        meshes = list(root.iter("mesh"))
        self.assertEqual(len(meshes), 43)
        for mesh in meshes:
            filename = mesh.get("filename")
            self.assertTrue(filename.startswith("../meshes/"))
            self.assertTrue((URDF_PATH.parent / filename).resolve().is_file())
        referenced_meshes = {mesh.get("filename").rsplit("/", 1)[-1] for mesh in meshes}
        asset_meshes = {
            path.name for path in (URDF_PATH.parent.parent / "meshes").glob("*.stl")
        }
        self.assertEqual(len(referenced_meshes), 27)
        self.assertEqual(asset_meshes, referenced_meshes)

        expected_tips = {
            "1-tip": (
                "finger_1_fingertip_1",
                "1-tip_Link",
                (0.01536449, 0.004452079, -0.008175908),
                (0, 0, 0),
            ),
            "2-tip": (
                "finger_2_fingertip_1",
                "2-tip_Link",
                (0.015364505, 0.004452078, -0.008175907),
                (0, 0, 0),
            ),
            "3-tip": (
                "finger_3_fingertip_1",
                "3-tip_Link",
                (0.01536449, 0.004452078, -0.008175908),
                (0, 0, 0),
            ),
            "4-tip": (
                "finger_4_fingertip_1",
                "4-tip_Link",
                (0.015315325, 0.004282186, -0.008356225),
                (0, 0, 0),
            ),
            "5-tip": (
                "mmhand_thumb_1_finger_7_fingertip_1",
                "5-tip_Link",
                (0.006628915, -0.000976518, -0.017136225),
                (1.531486234, -0.730633954, -0.704777306),
            ),
        }
        joints = {joint.get("name"): joint for joint in root.findall("joint")}
        links = {link.get("name") for link in root.findall("link")}
        for name, (parent, child, xyz, rpy) in expected_tips.items():
            joint = joints[name]
            self.assertEqual(joint.get("type"), "fixed")
            self.assertEqual(joint.find("parent").get("link"), parent)
            self.assertEqual(joint.find("child").get("link"), child)
            self.assertIn(child, links)
            np.testing.assert_allclose(
                np.fromstring(joint.find("origin").get("xyz"), sep=" "), xyz, atol=0
            )
            np.testing.assert_allclose(
                np.fromstring(joint.find("origin").get("rpy"), sep=" "), rpy, atol=0
            )
        transforms = self.model.fk(self.model.seed)
        np.testing.assert_allclose(
            self.model.fingertips(self.model.seed),
            np.asarray([transforms[name][:3, 3] for name in ROBOT_TIPS]),
            atol=0,
        )
        pad_points, pad_directions = self.model.fingertip_pads(self.model.seed)
        np.testing.assert_allclose(pad_points, self.model.fingertips(self.model.seed))
        np.testing.assert_allclose(
            pad_directions,
            np.asarray([-transforms[name][:3, 2] for name in ROBOT_TIPS]),
        )
        np.testing.assert_allclose(np.linalg.norm(pad_directions, axis=1), 1)
        np.testing.assert_allclose(
            pad_directions[0], (0.667341907, 0.744176136, -0.029268711),
            atol=1e-9,
        )
        np.testing.assert_allclose(
            pad_directions[1:], np.tile((0, 0, -1), (4, 1)), atol=1e-12
        )
        self.assertGreater(
            np.degrees(np.arccos(np.clip(pad_directions[0] @ (0, 0, -1), -1, 1))),
            85,
        )
        arrows = _arrow_points(pad_points, pad_directions)
        np.testing.assert_allclose(arrows[:, 0], pad_points)
        np.testing.assert_allclose(
            np.linalg.norm(np.diff(arrows, axis=1)[:, 0], axis=1), 0.025,
            atol=1e-8,
        )
        bent = (self.model.lower + self.model.upper) / 3
        bent_transforms = self.model.fk(bent)
        bent_points, bent_directions = self.model.fingertip_pads(bent)
        np.testing.assert_allclose(
            bent_directions,
            np.asarray([-bent_transforms[name][:3, 2] for name in ROBOT_TIPS]),
        )
        self.assertFalse(np.allclose(bent_points, pad_points))

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
            "Thumb_CMC": (0.0, 0.942478),
            "Ring_MCP_AA": (-0.802851, 0.453786),
            "Middle_MCP_AA": (-0.558505, 0.523599),
            "Index_MCP_AA": (-0.488692, 0.628319),
            "Little_MCP_AA": (-1.22173, 0.366519),
            "Thumb_MCP_AA": (-1.308997, 0.436332),
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

    def test_viewer_updates_and_retains_robot_pad_arrows(self):
        viewer = object.__new__(Viewer)
        viewer.model = self.model
        viewer.last_update, viewer.status = 0.0, "WAITING"
        viewer.loss_text, viewer.preview = _loss_text(), b""
        viewer.human_cloud = types.SimpleNamespace(visible=False)
        viewer.human_bones = types.SimpleNamespace(visible=False)
        viewer.pad_directions = types.SimpleNamespace(visible=False)
        viewer.human_retarget_frame = types.SimpleNamespace(visible=False)
        viewer.robot_pad_directions = types.SimpleNamespace(points=None)
        viewer.urdf_names, viewer.robot_index = self.model.names, self.model.index
        updates = []
        viewer.urdf = types.SimpleNamespace(update_cfg=lambda q: updates.append(q))
        frame = InputFrame(0, None, None, False, "WAITING")
        robot = (self.model.lower + self.model.upper) / 3

        viewer.update(frame, robot)
        expected = _arrow_points(*self.model.fingertip_pads(robot))
        np.testing.assert_allclose(viewer.robot_pad_directions.points, expected)
        self.assertEqual(len(updates), 1)

        retained = viewer.robot_pad_directions.points.copy()
        viewer.last_update = 0
        viewer.update(frame, None)
        np.testing.assert_array_equal(viewer.robot_pad_directions.points, retained)

    def test_analytic_retarget_jacobians_and_loss_gradients(self):
        q = (self.model.lower + self.model.upper) / 2
        values, jacobians = self.model.features(q)
        numeric = [np.empty_like(jacobian) for jacobian in jacobians]
        step = 1e-6
        for column, joint in enumerate(self.model.thumb):
            plus, minus = q.copy(), q.copy()
            plus[joint] += step
            minus[joint] -= step
            plus_values, minus_values = self.model.features(plus)[0], self.model.features(minus)[0]
            for group in range(len(values)):
                numeric[group][..., column] = (
                    plus_values[group] - minus_values[group]
                ) / (2 * step)
        for actual, expected in zip(jacobians, numeric):
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)

        points = relative_points(straight_hand("Left"))
        retargeter = Retargeter(self.model)
        targets = retargeter._targets(points)
        losses = retargeter._losses(q, targets)
        gradient = np.empty(5)
        for column, joint in enumerate(self.model.thumb):
            plus, minus = q.copy(), q.copy()
            plus[joint] += step
            minus[joint] -= step
            gradient[column] = (
                retargeter._losses(plus, targets)[0]
                - retargeter._losses(minus, targets)[0]
            ) / (2 * step)
        np.testing.assert_allclose(losses[1], gradient, atol=1e-6, rtol=1e-5)

    def test_active_thumb_targets(self):
        retargeter = Retargeter(self.model)
        points = relative_points(straight_hand("Left"))
        targets = retargeter._targets(points)
        local = human_retarget_points(points)
        vectors = np.diff(local[[2, 3, 4]], axis=0)
        np.testing.assert_allclose(
            targets[1], vectors / np.linalg.norm(vectors, axis=1)[:, None]
        )
        self.assertEqual(
            (targets[0].shape, targets[1].shape, targets[2].shape),
            ((1, 3), (2, 3), (21,)),
        )

    def test_direct_four_finger_angles_and_urdf_neutral(self):
        expected = np.array((20, 30, 25, 20, 40, 30, 20, 50, 30, 15, 40, 25, 10, 30))
        local = human_retarget_points(
            relative_points(apply_angles(straight_hand("Left"), "Left", expected))
        )
        direct = self.model.finger_angles(local)
        for row, (chain, joints, flexion) in enumerate(zip(
            ROBOT_FINGERS, self.model.finger_joints, expected[2:].reshape(4, 3)
        )):
            np.testing.assert_allclose(direct[joints[1:]], np.radians(flexion), atol=1e-10)
            self.assertAlmostEqual(direct[joints[0]], self.model.finger_neutral[row])
            self.assertNotAlmostEqual(direct[joints[0]], 0)
            midpoint = (self.model.lower[joints[0]] + self.model.upper[joints[0]]) / 2
            self.assertNotAlmostEqual(direct[joints[0]], midpoint)
            q = self.model.seed.copy()
            q[joints[0]] = self.model.finger_neutral[row]
            transforms = self.model.fk(q)
            direction = (
                transforms[chain[1]][:3, 3] - transforms[chain[0]][:3, 3]
            ) @ self.model.palm_frame
            self.assertGreater(direction[0], 0)
            self.assertAlmostEqual(direction[1], 0, places=10)

        side = local.copy()
        angle = np.radians(10)
        rotation = np.array(((np.cos(angle), -np.sin(angle)),
                             (np.sin(angle), np.cos(angle))))
        origin = side[5, :2].copy()
        side[6:9, :2] = (side[6:9, :2] - origin) @ rotation.T + origin
        spread = self.model.finger_angles(side)
        index = self.model.finger_joints[0, 0]
        self.assertAlmostEqual(
            spread[index],
            self.model.finger_neutral[0] + angle / self.model.finger_axis[0],
        )

    def test_single_stage_retarget_and_limits(self):
        expected = np.array((20, 30, 25, 20, 40, 30, 20, 50, 30, 15, 40, 25, 10, 30))
        points = relative_points(
            apply_angles(straight_hand("Left"), "Left", expected)
        )
        retargeter = Retargeter(self.model)
        targets = retargeter._targets(points)
        local = human_retarget_points(points)
        np.testing.assert_allclose(points[0], 0, atol=1e-12)
        np.testing.assert_allclose(targets[0], local[4:5])
        direction_params = (
            ("RETARGET_THUMB_MCP_IP_WEIGHT", "thumb_mcp_ip"),
            ("RETARGET_THUMB_IP_TIP_WEIGHT", "thumb_ip_tip"),
        )
        baseline = retargeter._losses(self.model.seed, targets)[2]
        for parameter, name in direction_params:
            with patch(f"retarget.C.{parameter}", 0.0):
                changed = retargeter._losses(self.model.seed, targets)[2]
            self.assertEqual(changed[name], 0.0)
            for _, other in direction_params:
                if other != name:
                    self.assertEqual(changed[other], baseline[other])
        self.assertFalse(retargeter.has_previous)
        result = retargeter.solve(points)
        self.assertIsNotNone(result)
        self.assertTrue(retargeter.has_previous)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(np.all(result >= self.model.lower - 1e-10))
        self.assertTrue(np.all(result <= self.model.upper + 1e-10))
        self.assertEqual(
            tuple(retargeter.loss_terms),
            ("thumb_tip", "thumb_mcp_ip", "thumb_ip_tip", "total"),
        )
        self.assertAlmostEqual(
            sum(value for name, value in retargeter.loss_terms.items() if name != "total"),
            retargeter.loss_terms["total"],
        )
        loss_text = _loss_text(retargeter.loss_terms)
        self.assertIn("thumb MCP-IP", loss_text)
        self.assertIn("thumb IP-TIP", loss_text)
        self.assertIn("100.0%", loss_text)
        self.assertIn("  0.0%", _loss_text(dict.fromkeys(retargeter.loss_terms, 0.0)))
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
        self.assertIsNone(retargeter.loss_terms)

    def test_thumb_ik_calls_slsqp_once_and_keeps_best_candidate(self):
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
        expected = retargeter._targets(points)[2]
        np.testing.assert_allclose(result, expected, atol=1e-15)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]["bounds"].lb), 5)
        self.assertNotIn("constraints", calls[0])
        self.assertNotIn("maxiter", calls[0]["options"])
        previous, losses = retargeter.q.copy(), retargeter.loss_terms.copy()
        failure = types.SimpleNamespace(success=False, x=np.full(21, np.nan))
        with patch("retarget.minimize", return_value=failure):
            fallback = retargeter.solve(points)
        np.testing.assert_allclose(fallback, previous, atol=1e-15)
        self.assertEqual(retargeter.loss_terms, losses)

    def test_retarget_stops_after_evaluation_budget_with_best_candidate(self):
        points = relative_points(straight_hand("Left"))
        retargeter = Retargeter(self.model)
        evaluated = []

        def fake_losses(q, _targets):
            evaluated.append(np.asarray(q).copy())
            loss = float(100 - len(evaluated))
            terms = dict.fromkeys(
                ("thumb_tip", "thumb_mcp_ip", "thumb_ip_tip"),
                0.0,
            )
            terms["total"] = loss
            return loss, np.zeros(5), terms

        def exhaust(fun, _x0, **_kwargs):
            for step in range(1, RETARGET_MAX_EVALUATIONS + 2):
                fraction = step / (RETARGET_MAX_EVALUATIONS + 2)
                joints = self.model.thumb
                fun(self.model.lower[joints] + fraction * (self.model.upper - self.model.lower)[joints])
            self.fail("Evaluation budget was not enforced")

        with patch.object(retargeter, "_losses", side_effect=fake_losses), \
             patch("retarget.minimize", side_effect=exhaust):
            result = retargeter.solve(points)
        self.assertEqual(len(evaluated), RETARGET_MAX_EVALUATIONS + 1)
        np.testing.assert_array_equal(result, evaluated[-1])
        self.assertEqual(retargeter.loss_terms["total"], 99.0 - RETARGET_MAX_EVALUATIONS)

    def test_retarget_output_filter_uses_timestamps_and_resets(self):
        first_pose = relative_points(straight_hand("Left"))
        second_pose = relative_points(apply_angles(
            straight_hand("Left"), "Left",
            np.array((60, 70, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25)),
        ))
        retargeter = Retargeter(self.model)
        first = retargeter.solve(first_pose, 0.0)
        raw_first = retargeter.q.copy()
        np.testing.assert_allclose(first, raw_first, atol=1e-15)
        second = retargeter.solve(second_pose, 1 / 30)
        raw_second = retargeter.q.copy()
        changed = np.abs(raw_second - first) > 1e-8
        self.assertTrue(changed.any())
        self.assertTrue(np.all(np.abs(second - first) <= np.abs(raw_second - first) + 1e-12))
        self.assertTrue(np.any(np.abs(second[changed] - raw_second[changed]) > 1e-8))
        self.assertEqual(retargeter.output_filter.value.shape, (21,))
        targets = retargeter._targets(second_pose, raw_first)
        expected = retargeter._losses(second, targets)[2]
        for name, value in expected.items():
            self.assertAlmostEqual(retargeter.loss_terms[name], value)
        retargeter.pause()
        reset = retargeter.solve(second_pose, 1.0)
        np.testing.assert_allclose(reset, retargeter.q, atol=1e-15)

    def test_retarget_worker_keeps_latest_frame_and_discards_paused_work(self):
        class FakeRetargeter:
            model = object()

            def __init__(self):
                self.calls, self.timestamps, self.pauses = [], [], 0
                self.started, self.release = threading.Event(), threading.Event()

            def solve(self, points, timestamp):
                value = int(points[0, 0])
                self.calls.append(value)
                self.timestamps.append(timestamp)
                if not self.release.is_set():
                    self.started.set()
                    self.release.wait(1)
                self.loss_terms = {"total": float(value)}
                return np.full(21, value, float)

            def pause(self):
                self.pauses += 1

        fake, worker = FakeRetargeter(), None

        def wait_for(expected):
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                output = worker.poll()
                if output is not None and output[0][0] == expected:
                    self.assertEqual(output[1], {"total": float(expected)})
                    return output[0]
                time.sleep(0.001)
            self.fail(f"Timed out waiting for retarget result {expected}")

        worker = RetargetWorker(fake)
        try:
            worker.submit(np.full((21, 3), 1), 1.0)
            self.assertTrue(fake.started.wait(1))
            worker.submit(np.full((21, 3), 2), 2.0)
            worker.submit(np.full((21, 3), 3), 3.0)
            fake.release.set()
            np.testing.assert_array_equal(wait_for(3), np.full(21, 3))
            self.assertEqual(fake.calls, [1, 3])
            self.assertEqual(fake.timestamps, [1.0, 3.0])
            self.assertIsNone(worker.poll())

            fake.started.clear()
            fake.release.clear()
            worker.submit(np.full((21, 3), 4), 4.0)
            self.assertTrue(fake.started.wait(1))
            worker.pause()
            worker.submit(np.full((21, 3), 5), 5.0)
            fake.release.set()
            np.testing.assert_array_equal(wait_for(5), np.full(21, 5))
            self.assertEqual(fake.calls, [1, 3, 4, 5])
            self.assertEqual(fake.timestamps, [1.0, 3.0, 4.0, 5.0])
            self.assertEqual(fake.pauses, 1)
        finally:
            worker.close()

        def fail(_points, _timestamp):
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
            frame = InputFrame(
                1.0, np.zeros((21, 3)), "Left", True, "TRACKING",
                finger_pad_directions=np.ones((5, 3)),
            )
            output.hand(frame)
            output.hand(InputFrame(2.0, np.zeros((21, 3)), "Left", True, "TRACKING"))
            output.joints(np.arange(21))
            output.close()

        self.assertEqual([publisher.topic for publisher in node.publishers],
                         [KEYPOINT_TOPIC, PAD_DIRECTION_TOPIC, ROBOT_TOPIC])
        point_message = node.publishers[0].messages[0]
        direction_message = node.publishers[1].messages[0]
        robot_message = node.publishers[2].messages[0]
        self.assertEqual(len(point_message.data), 63)
        self.assertEqual(len(direction_message.data), 15)
        self.assertEqual(len(node.publishers[1].messages), 1)
        self.assertEqual(len(robot_message.data), 21)
        self.assertEqual(point_message.layout.dim[0].label,
                         f"{KEYPOINT_LAYOUT}:hand=Left")
        self.assertEqual(direction_message.layout.dim[0].label,
                         f"{PAD_DIRECTION_LAYOUT}:hand=Left")
        self.assertEqual(robot_message.layout.dim[0].label, ROBOT_LAYOUT)
        self.assertTrue(node.destroyed)
        self.assertTrue(rclpy.shutdown_called)


if __name__ == "__main__":
    unittest.main()
