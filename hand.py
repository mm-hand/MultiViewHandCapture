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
class HandFrame:
    timestamp: float
    points: np.ndarray | None
    handedness: str | None
    ready: bool
    status: str
    # MANUS Thumb Tip Raw Skeleton node GLOBAL rotation, relative to the
    # configured MANUS WORLD coordinate system. Quaternion order is xyzw.
    # It is intentionally independent from points and is not CMC-normalized.
    thumb_tip_orientation_world_xyzw: np.ndarray | None = None
    preview: np.ndarray | None = None


def _unit(vector, fallback=(1.0, 0.0, 0.0)):
    norm = np.linalg.norm(vector)
    if norm > EPS:
        return vector / norm
    fallback = np.asarray(fallback, float)
    return fallback / max(np.linalg.norm(fallback), EPS)


def relative_points(points):
    """Return the input-independent normalized standard-21 tracking points.

    The returned coordinates keep the historical wrist-origin frame used by
    VisionSource and Retargeter. Both Vision and MANUS call this one helper.
    """
    points = np.asarray(points, float)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("Hand landmarks must be finite with shape (21, 3)")
    centered = points - points[0]
    palm_size = np.mean([np.linalg.norm(centered[index]) for index in (5, 9, 13, 17)])
    if palm_size < EPS:
        return None
    points = centered * STANDARD_PALM_SIZE / palm_size
    z_axis = _unit(points[9])
    x_axis = _unit(np.cross(points[5] - points[17], z_axis))
    y_axis = _unit(np.cross(z_axis, x_axis))
    return points @ np.column_stack((x_axis, y_axis, z_axis))
