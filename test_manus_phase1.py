import unittest
from unittest.mock import patch

import numpy as np

from hand import relative_points
from manus.adapter import (
    MANUS_TO_STANDARD21,
    adapt_raw_skeleton,
    convert_manus25_to_standard21,
    sdk_quaternion_wxyz_to_xyzw,
    semantic_standard21_mapping,
)
from manus.source import ManusSource, _SdkFrame, _sdk_frame_to_packet, parse_legacy_bridge_message
from retarget import compute_cmc_frame, human_retarget_points
from track import _parse_args, _source


def standard_hand_world():
    points = np.zeros((21, 3), float)
    points[1:5] = ((0.01, -0.035, 0), (0.025, -0.045, 0), (0.04, -0.05, 0), (0.055, -0.052, 0))
    for mcp, lateral in zip((5, 9, 13, 17), (-0.03, -0.01, 0.01, 0.03)):
        points[mcp : mcp + 4] = [(0.04 + 0.025 * index, lateral, 0.002 * index) for index in range(4)]
    return points


def raw_packet(received_at=1.0):
    positions = np.zeros((25, 3), float)
    positions[MANUS_TO_STANDARD21] = standard_hand_world()
    positions[[5, 10, 15, 20]] = ((0.015, -0.03, 0), (0.02, -0.01, 0), (0.02, 0.01, 0), (0.015, 0.03, 0))
    rotations = np.tile((0.0, 0.0, 0.0, 1.0), (25, 1))
    rotations[4] = (0.1, 0.2, 0.3, 0.9)
    return {
        "glove_id": "abc123",
        "positions": positions,
        "rotations": rotations,
        "quaternion_convention": "xyzw",
        "received_at": received_at,
        "coordinate_mode": "WORLD/GLOBAL",
    }


class FakeTransport:
    def __init__(self, *values):
        self.values = list(values)
        self.closed = False

    def read(self):
        return self.values.pop(0) if self.values else None

    def close(self):
        self.closed = True


class ManusPhase1Tests(unittest.TestCase):
    def test_cli_matrix_and_source_factory(self):
        self.assertEqual(_parse_args([]).source, "vision")
        for source_name in ("vision", "manus"):
            args = _parse_args(["--source", source_name, "--ros"])
            self.assertEqual(args.source, source_name)
            self.assertTrue(args.ros)
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            _parse_args(["--source", "xxx"])
        selected = object()
        with patch("manus.source.ManusSource", return_value=selected):
            self.assertIs(_source("manus"), selected)

    def test_manus_25_to_standard21_mapping(self):
        points = np.repeat(np.arange(25, dtype=float)[:, None], 3, axis=1)
        output = convert_manus25_to_standard21(points)
        expected = [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24]
        np.testing.assert_array_equal(output[:, 0], expected)
        self.assertTrue(set((5, 10, 15, 20)).isdisjoint(output[:, 0]))
        self.assertIn(1, output[:, 0])

    def test_manus_position_shape_and_finite_validation(self):
        for shape in ((24, 3), (21, 3), (25, 2)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                convert_manus25_to_standard21(np.zeros(shape))
        for value in (np.nan, np.inf):
            points = np.zeros((25, 3))
            points[7, 1] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                convert_manus25_to_standard21(points)
        self.assertEqual(convert_manus25_to_standard21(np.zeros((25, 3))).shape, (21, 3))

    def test_sdk_quaternion_component_convention(self):
        # SDK ManusQuaternion fields are w,x,y,z -- each component is distinct.
        np.testing.assert_array_equal(
            sdk_quaternion_wxyz_to_xyzw((11.0, 22.0, 33.0, 44.0)),
            (22.0, 33.0, 44.0, 11.0),
        )

    def test_official_sdk_c_abi_keeps_raw_wxyz_until_adapter(self):
        frame = _SdkFrame()
        frame.glove_id = 42
        frame.node_count = 25
        frame.side = 1
        frame.publish_time = 123456
        for row in range(25):
            frame.nodes[row].node_id = row
            frame.nodes[row].position[:] = (row, row + 0.25, row + 0.5)
            frame.nodes[row].rotation_wxyz[:] = (11, 22, 33, 44)
        packet = _sdk_frame_to_packet(frame, 8.0)
        np.testing.assert_array_equal(packet["rotations"][4], (11, 22, 33, 44))
        self.assertEqual(packet["quaternion_convention"], "wxyz")
        np.testing.assert_array_equal(
            sdk_quaternion_wxyz_to_xyzw(packet["rotations"][4]),
            (22, 33, 44, 11),
        )

    def test_node_info_semantics_are_preferred(self):
        info = [{"nodeId": 0, "parentId": 0, "chainType": 13, "side": 1, "fingerJointType": 0}]
        row = 1
        for chain in range(5, 10):
            joints = (1, 2, 4, 5) if chain == 5 else (1, 2, 3, 4, 5)
            parent = 0
            for joint in joints:
                info.append({"nodeId": row, "parentId": parent, "chainType": chain, "side": 1, "fingerJointType": joint})
                parent, row = row, row + 1
        self.assertEqual(row, 25)
        np.testing.assert_array_equal(semantic_standard21_mapping(info), MANUS_TO_STANDARD21)
        packet = raw_packet()
        adapted = adapt_raw_skeleton(packet["positions"], packet["rotations"], node_info=info)
        self.assertEqual(adapted.mapping_source, "NodeInfo")
        self.assertEqual(adapted.thumb_tip_row, 4)

    def test_orientation_passthrough_adapter_source_handframe(self):
        packet = raw_packet(received_at=5.0)
        # Model the official ManusQuaternion memory/field order w,x,y,z.
        packet["rotations"][4] = (0.9, 0.1, 0.2, 0.3)
        packet["quaternion_convention"] = "wxyz"
        expected = np.array((0.1, 0.2, 0.3, 0.9))
        source = ManusSource(
            FakeTransport(packet),
            glove_id_to_handedness={"abc123": "Left"},
            stale_seconds=0.2,
            clock=lambda: 5.0,
        )
        frame = source.read()
        self.assertTrue(frame.ready)
        self.assertEqual(frame.handedness, "Left")
        self.assertEqual(frame.points.shape, (21, 3))
        np.testing.assert_array_equal(frame.thumb_tip_orientation_world_xyzw, expected)
        source.close()

    def test_stale_clears_points_and_orientation(self):
        source = ManusSource(
            FakeTransport(raw_packet(received_at=1.0)),
            glove_id_to_handedness={"abc123": "Left"},
            stale_seconds=0.2,
            clock=lambda: 2.0,
        )
        frame = source.read()
        self.assertIsNone(frame.points)
        self.assertIsNone(frame.thumb_tip_orientation_world_xyzw)
        self.assertFalse(frame.ready)

    def test_legacy_bridge_protocol_keeps_xyzw(self):
        values = np.zeros((25, 7), float)
        values[:, 6] = 1.0
        values[4, 3:7] = (0.1, 0.2, 0.3, 0.9)
        message = "abc123," + ",".join(map(str, values.ravel()))
        packet = parse_legacy_bridge_message(message, received_at=3.0)[0]
        np.testing.assert_array_equal(packet["rotations"][4], (0.1, 0.2, 0.3, 0.9))
        self.assertEqual(packet["quaternion_convention"], "xyzw")

    def test_cmc_helper_refactor_regression(self):
        points = relative_points(standard_hand_world())
        x_axis = points[9] - points[0]
        x_axis /= np.linalg.norm(x_axis)
        side = points[17] - points[5]
        y_axis = side - x_axis * np.dot(side, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis)
        y_axis = np.cross(z_axis, x_axis)
        old_rotation = np.column_stack((x_axis, y_axis, z_axis))
        old_result = (points - points[1]) @ old_rotation

        origin, rotation = compute_cmc_frame(points)
        np.testing.assert_allclose(origin, points[1], atol=0)
        np.testing.assert_allclose(rotation, old_rotation, atol=1e-15)
        np.testing.assert_allclose(human_retarget_points(points), old_result, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
