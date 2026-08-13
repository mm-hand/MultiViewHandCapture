import unittest
import numpy as np

from input.frame import relative_hand
from input.manus.adapter import (
    MANUS_TO_STANDARD21,
    adapt_raw_skeleton,
    convert_manus25_to_standard21,
)
from input.manus.source import ManusPacket, ManusSource, _SdkFrame, _sdk_frame_to_packet


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
    return ManusPacket(
        positions,
        np.tile((1.0, 0.0, 0.0, 0.0), (25, 1)),
        list(range(25)),
        [],
        "Left",
        received_at,
    )


class FakeTransport:
    def __init__(self, *values):
        self.values = list(values)
        self.closed = False

    def read(self):
        return self.values.pop(0) if self.values else None

    def close(self):
        self.closed = True


class ManusPhase1Tests(unittest.TestCase):
    def test_manus_25_to_standard21_mapping(self):
        points = np.repeat(np.arange(25, dtype=float)[:, None], 3, axis=1)
        output = convert_manus25_to_standard21(points)
        expected = [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24]
        np.testing.assert_array_equal(output[:, 0], expected)
        self.assertTrue(set((5, 10, 15, 20)).isdisjoint(output[:, 0]))
        self.assertIn(1, output[:, 0])

    def test_official_sdk_packet_keeps_rotations(self):
        frame = _SdkFrame()
        frame.glove_id = 42
        frame.node_count = 25
        frame.side = 1
        frame.publish_time = 123456
        for row in range(25):
            frame.nodes[row].node_id = row
            frame.nodes[row].position[:] = (row, row + 0.25, row + 0.5)
            frame.nodes[row].rotation_wxyz[0] = 1
        packet = _sdk_frame_to_packet(frame, 8.0)
        self.assertEqual(packet.rotations_wxyz.shape, (25, 4))
        np.testing.assert_array_equal(packet.rotations_wxyz[:, 0], 1)

    def test_source_returns_common_frame_with_pad_directions(self):
        packet = raw_packet(received_at=5.0)
        source = ManusSource(
            FakeTransport(packet),
            stale_seconds=0.2,
            clock=lambda: 5.0,
        )
        frame = source.read()
        self.assertTrue(frame.ready)
        self.assertEqual(frame.handedness, "Left")
        self.assertEqual(frame.points.shape, (21, 3))
        self.assertEqual(frame.finger_pad_directions.shape, (5, 3))
        np.testing.assert_allclose(np.linalg.norm(frame.finger_pad_directions, axis=1), 1)
        source.close()

    def test_tip_quaternions_define_pad_directions(self):
        packet = raw_packet()
        root = np.sqrt(.5)
        packet.rotations_wxyz[MANUS_TO_STANDARD21[[4, 8, 12, 16, 20]]] = (
            root, root, 0, 0
        )
        adapted = adapt_raw_skeleton(
            packet.positions, rotations_wxyz=packet.rotations_wxyz
        )
        standard = convert_manus25_to_standard21(packet.positions)
        expected = relative_hand(standard, np.tile((0, 1., 0), (5, 1)))[1]
        np.testing.assert_allclose(adapted.directions, expected, atol=1e-7)

    def test_stale_clears_points(self):
        source = ManusSource(
            FakeTransport(raw_packet(received_at=1.0)),
            stale_seconds=0.2,
            clock=lambda: 2.0,
        )
        frame = source.read()
        self.assertIsNone(frame.points)
        self.assertFalse(frame.ready)

if __name__ == "__main__":
    unittest.main()
