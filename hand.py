from dataclasses import dataclass

import numpy as np

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
    preview: np.ndarray | None = None
