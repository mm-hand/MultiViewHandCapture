from dataclasses import dataclass

import numpy as np

from config import STANDARD_PALM_SIZE

EPS = 1e-8

FINGER_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)
SKELETON_EDGES = tuple(
    edge for chain in FINGER_CHAINS for edge in zip(chain[:-1], chain[1:])
)


@dataclass(slots=True)
class InputFrame:
    timestamp: float
    points: np.ndarray | None
    handedness: str | None
    ready: bool
    status: str
    # Outward unit normals for Thumb, Index, Middle, Ring, Little finger pads.
    # Coordinates use the same palm-local tracking frame as points.
    finger_pad_directions: np.ndarray | None = None
    preview: np.ndarray | None = None


def _unit(vector, fallback=(1.0, 0.0, 0.0)):
    norm = np.linalg.norm(vector)
    if norm > EPS:
        return vector / norm
    fallback = np.asarray(fallback, float)
    return fallback / max(np.linalg.norm(fallback), EPS)


def relative_hand(points, directions=None):
    """Normalize standard-21 points and optional directions to the tracking frame."""
    points = np.asarray(points, float)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("Hand landmarks must be finite with shape (21, 3)")
    centered = points - points[0]
    palm_size = np.mean([np.linalg.norm(centered[index]) for index in (5, 9, 13, 17)])
    if palm_size < EPS:
        return None, None
    z_axis = _unit(centered[9])
    x_axis = _unit(np.cross(centered[5] - centered[17], z_axis))
    y_axis = _unit(np.cross(z_axis, x_axis))
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    normalized = centered * STANDARD_PALM_SIZE / palm_size @ rotation
    if directions is None:
        return normalized, None
    directions = np.asarray(directions, float)
    if directions.shape != (5, 3) or not np.isfinite(directions).all():
        raise ValueError("Finger-pad directions must be finite with shape (5, 3)")
    local = directions @ rotation
    lengths = np.linalg.norm(local, axis=1, keepdims=True)
    if np.any(lengths < EPS):
        raise ValueError("Finger-pad directions must be nonzero")
    return normalized, local / lengths


def relative_points(points):
    """Return normalized standard-21 points in the wrist-local frame."""
    return relative_hand(points)[0]
