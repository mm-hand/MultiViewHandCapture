import types
import unittest
from unittest.mock import patch

import numpy as np

from hand import HandFrame, relative_hand
from track import _parse_args, _source
from wilor.source import (
    Detector,
    JOINT_ORDER,
    PAD_FACE_ROWS,
    Reconstructor,
    TIP_VERTICES,
    Track,
    Tracker,
    draw_mesh,
    mesh_hand,
)


def standard_hand():
    points = np.zeros((21, 3), np.float32)
    points[1:5] = ((0.02, 0.01, 0.01), (0.035, 0.02, 0.015),
                   (0.045, 0.03, 0.02), (0.055, 0.04, 0.025))
    for start, x in zip((5, 9, 13, 17), (0.03, 0.01, -0.01, -0.03)):
        points[start:start + 4] = ((x, 0.05, 0), (x, 0.075, 0),
                                   (x, 0.095, 0), (x, 0.11, 0))
    return points


def synthetic_mesh():
    vertices = np.zeros((778, 3), np.float32)
    regressor = np.zeros((16, 778), np.float32)
    regressor[np.arange(16), np.arange(16)] = 1
    raw = np.zeros((21, 3), np.float32)
    raw[JOINT_ORDER] = standard_hand()
    vertices[:16] = raw[:16]
    vertices[TIP_VERTICES] = raw[16:]
    faces = np.zeros((int(PAD_FACE_ROWS.max()) + 1, 3), np.int32)
    vertex = 100
    for finger, rows in enumerate(PAD_FACE_ROWS):
        for row in rows:
            faces[row] = (vertex, vertex + 1, vertex + 2)
            origin = np.array((finger * 0.01, 0, 0), np.float32)
            vertices[vertex:vertex + 3] = origin + ((0, 0, 0), (0.004, 0, 0),
                                                    (0, 0, 0.004))
            vertex += 3
    return vertices, regressor, faces


class FakeDetectorSession:
    def __init__(self, output):
        self.output = output

    def get_inputs(self):
        return [types.SimpleNamespace(type="tensor(float16)")]

    def run(self, *_):
        return [self.output]


class FakeReconstructorSession:
    def __init__(self, camera):
        self.camera = np.asarray(camera, np.float32)[None]
        self.input = None

    def get_inputs(self):
        return [types.SimpleNamespace(type="tensor(float16)")]

    def run(self, _, inputs):
        self.input = inputs["images"]
        return [np.zeros((1, 778, 3), np.float16), self.camera.copy()]


class WilorTests(unittest.TestCase):
    def test_cli_and_source_factory(self):
        self.assertEqual(_parse_args(["--source", "wilor"]).source, "wilor")
        selected = object()
        with patch("wilor.source.WilorSource", return_value=selected):
            self.assertIs(_source("wilor"), selected)

    def test_hand_frame_direction_default_and_validation(self):
        frame = HandFrame(0, None, None, False, "waiting")
        self.assertIsNone(frame.finger_pad_directions)
        with self.assertRaisesRegex(ValueError, "shape"):
            relative_hand(standard_hand(), np.zeros((4, 3)))
        with self.assertRaisesRegex(ValueError, "nonzero"):
            relative_hand(standard_hand(), np.zeros((5, 3)))

    def test_tracking_frame_is_rigid_transform_invariant(self):
        points = standard_hand()
        directions = np.tile((0.0, 0.0, -1.0), (5, 1))
        angle = 0.7
        rotation = np.array(((np.cos(angle), -np.sin(angle), 0),
                             (np.sin(angle), np.cos(angle), 0),
                             (0, 0, 1)))
        expected_points, expected_directions = relative_hand(points, directions)
        actual_points, actual_directions = relative_hand(
            points @ rotation.T + (2, 3, 4), directions @ rotation.T
        )
        np.testing.assert_allclose(actual_points, expected_points, atol=1e-7)
        np.testing.assert_allclose(actual_directions, expected_directions, atol=1e-7)

    def test_mesh_to_standard21_and_pad_directions(self):
        vertices, regressor, faces = synthetic_mesh()
        points, directions, handedness = mesh_hand(vertices, 1, regressor, faces)
        self.assertEqual(handedness, "Right")
        self.assertEqual(points.shape, (21, 3))
        self.assertEqual(directions.shape, (5, 3))
        self.assertTrue(np.isfinite(points).all())
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1, atol=1e-7)

        left_points, left_directions, handedness = mesh_hand(
            vertices, 0, regressor, faces
        )
        self.assertEqual(handedness, "Left")
        self.assertTrue(np.isfinite(left_points).all())
        np.testing.assert_allclose(np.linalg.norm(left_directions, axis=1), 1, atol=1e-7)

    def test_mesh_rejects_bad_vertices_and_faces(self):
        vertices, regressor, faces = synthetic_mesh()
        with self.assertRaisesRegex(ValueError, "shape"):
            mesh_hand(vertices[:10], 1, regressor, faces)
        vertices[faces[PAD_FACE_ROWS].ravel()] = 0
        with self.assertRaisesRegex(ValueError, "Degenerate"):
            mesh_hand(vertices, 1, regressor, faces)

    def test_detector_threshold_coordinates_and_handedness(self):
        raw = np.array([[[320, 100], [320, 100], [100, 20], [100, 20],
                         [0.1, 0.2], [0.9, 0.1]]], np.float16)
        detector = Detector(FakeDetectorSession(raw), 0.3, 0.5)
        detections = detector(np.zeros((480, 640, 3), np.uint8))
        self.assertEqual(len(detections), 1)
        box, handedness, score = detections[0]
        np.testing.assert_allclose(box, (270, 190, 370, 290), atol=1)
        self.assertEqual(handedness, 1)
        self.assertAlmostEqual(score, 0.9, places=2)

    def test_reconstruction_camera_translation_and_mesh_overlay(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        box = np.array((270, 190, 370, 290), np.float32)
        for right, expected_x in ((1, 0.25), (0, -0.25)):
            session = FakeReconstructorSession((2, 0.25, -0.1))
            vertices, translation = Reconstructor(session, 2)(
                frame, [Track(box.copy(), right, 0.9, np.zeros(4))]
            )
            self.assertEqual(vertices.shape, (1, 778, 3))
            self.assertEqual(session.input.shape, (1, 3, 256, 192))
            np.testing.assert_allclose(
                translation[0], (expected_x, -0.1, 46.875), atol=1e-5
            )
        with self.assertRaisesRegex(ValueError, "camera scale"):
            Reconstructor(FakeReconstructorSession((-1, 0, 0)), 2)(
                frame, [Track(box.copy(), 1, 0.9, np.zeros(4))]
            )

        vertices = np.array(((-0.01, -0.01, 0), (0.01, -0.01, 0),
                             (0, 0.01, 0)), np.float32)
        faces = np.array(((0, 1, 2),), np.int32)
        right_image, left_image = frame.copy(), frame.copy()
        draw_mesh(right_image, vertices, (0, 0, 1), 1, faces)
        draw_mesh(left_image, vertices, (0, 0, 1), 0, faces)
        self.assertGreater(right_image[240, 320, 2], right_image[240, 320, 0])
        self.assertGreater(left_image[240, 320, 0], left_image[240, 320, 2])

    def test_tracker_predicts_then_expires(self):
        tracker = Tracker()
        detection = [(np.array((1, 2, 3, 4), np.float32), 0, 0.8)]
        self.assertEqual(len(tracker.update(detection)), 1)
        self.assertEqual(len(tracker.update()), 1)
        self.assertEqual(len(tracker.update([])), 1)
        self.assertEqual(len(tracker.update([])), 1)
        self.assertEqual(tracker.update([]), [])


if __name__ == "__main__":
    unittest.main()
