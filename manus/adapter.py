"""MANUS-only topology, unit, and quaternion boundary conversions."""

from dataclasses import dataclass

import numpy as np

from hand import relative_points


# Official/legacy 25-node layout fallback. Non-thumb metacarpals 5/10/15/20
# are deliberately omitted; Thumb Metacarpal (standard Thumb CMC) is retained.
MANUS_TO_STANDARD21 = np.array(
    (0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24),
    dtype=np.intp,
)
FALLBACK_THUMB_TIP_ROW = 4
STANDARD21_THUMB_TIP = 4

_CHAIN_NAMES = {
    5: "thumb",
    6: "index",
    7: "middle",
    8: "ring",
    9: "pinky",
    13: "hand",
}
_JOINT_NAMES = {0: "invalid", 1: "metacarpal", 2: "proximal", 3: "intermediate", 4: "distal", 5: "tip"}
_SIDE_NAMES = {1: "Left", 2: "Right"}


@dataclass(frozen=True, slots=True)
class AdaptedManusFrame:
    points: np.ndarray
    thumb_tip_orientation_world_xyzw: np.ndarray
    mapping: np.ndarray
    thumb_tip_row: int
    mapping_source: str


def _field(value, name):
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _enum_name(value, numeric_names):
    if isinstance(value, (int, np.integer)):
        return numeric_names.get(int(value), str(int(value)))
    name = getattr(value, "name", value)
    name = str(name).lower()
    for prefix in ("chaintype_finger", "chaintype_", "fingerjointtype_", "side_"):
        name = name.removeprefix(prefix)
    return name


def handedness_from_node_info(node_info):
    """Return Left/Right from NodeInfo.side, or None if it is unavailable."""
    items = () if node_info is None else node_info
    sides = {
        _SIDE_NAMES.get(int(side), None)
        if isinstance(side, (int, np.integer))
        else str(getattr(side, "name", side)).removeprefix("Side_").title()
        for item in items
        if (side := _field(item, "side")) is not None
    }
    sides &= {"Left", "Right"}
    return sides.pop() if len(sides) == 1 else None


def _chain_order(entries):
    """Order (row, node_id, parent_id, joint) entries by their hierarchy."""
    remaining = list(entries)
    ids = {entry[1] for entry in remaining}
    ordered = []
    while remaining:
        candidates = [entry for entry in remaining if entry[2] not in ids or entry[2] in {item[1] for item in ordered}]
        if not candidates:
            candidates = remaining
        entry = min(candidates, key=lambda item: item[0])
        ordered.append(entry)
        remaining.remove(entry)
    return ordered


def semantic_standard21_mapping(node_info, node_ids=None):
    """Build standard-21 row indices from MANUS NodeInfo semantics.

    NodeInfo is authoritative when complete. The caller may then fall back to
    ``MANUS_TO_STANDARD21`` for the documented 25-row layout.
    """
    if node_info is None or len(node_info) == 0:
        raise ValueError("MANUS NodeInfo is unavailable")
    node_ids = None if node_ids is None else list(map(int, node_ids))
    row_by_id = None if node_ids is None else {node_id: row for row, node_id in enumerate(node_ids)}
    groups = {name: [] for name in ("thumb", "index", "middle", "ring", "pinky", "hand")}
    for fallback_row, item in enumerate(node_info):
        node_id = int(_field(item, "nodeId"))
        row = fallback_row if row_by_id is None else row_by_id.get(node_id)
        if row is None:
            continue
        chain = _enum_name(_field(item, "chainType"), _CHAIN_NAMES)
        joint = _enum_name(_field(item, "fingerJointType"), _JOINT_NAMES)
        if chain in groups:
            groups[chain].append((row, node_id, int(_field(item, "parentId")), joint))
    hands = _chain_order(groups["hand"])
    if not hands:
        raise ValueError("NodeInfo has no Hand/Wrist node")
    mapping = [hands[0][0]]
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        chain = _chain_order(groups[finger])
        if finger != "thumb":
            chain = [entry for entry in chain if entry[3] != "metacarpal"]
        if len(chain) != 4:
            raise ValueError(f"NodeInfo {finger} chain does not map to four standard joints")
        mapping.extend(entry[0] for entry in chain)
    result = np.asarray(mapping, dtype=np.intp)
    if result.shape != (21,) or len(set(result.tolist())) != 21:
        raise ValueError("NodeInfo produced an invalid standard-21 mapping")
    return result


def convert_manus25_to_standard21(points25, mapping=None, scale_to_m=1.0):
    """Validate and map finite MANUS 25x3 positions to standard 21x3 meters."""
    points = np.asarray(points25, dtype=float)
    if points.shape != (25, 3):
        raise ValueError("MANUS positions must have shape (25, 3)")
    if not np.isfinite(points).all():
        raise ValueError("MANUS positions must be finite")
    scale = float(scale_to_m)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("MANUS position scale must be finite and positive")
    indices = MANUS_TO_STANDARD21 if mapping is None else np.asarray(mapping, dtype=np.intp)
    if indices.shape != (21,) or np.any(indices < 0) or np.any(indices >= 25):
        raise ValueError("MANUS standard-21 mapping must contain 21 valid rows")
    return points[indices].copy() * scale


def quaternion_to_xyzw(quaternion, raw_convention):
    """Convert a finite quaternion from an explicit convention to xyzw."""
    raw = np.asarray(quaternion, dtype=float)
    if raw.shape != (4,) or not np.isfinite(raw).all():
        raise ValueError("MANUS quaternion must be finite with shape (4,)")
    if np.linalg.norm(raw) <= np.finfo(float).eps:
        raise ValueError("MANUS quaternion must be nonzero")
    if raw_convention == "wxyz":
        return raw[[1, 2, 3, 0]].copy()
    if raw_convention == "xyzw":
        return raw.copy()
    raise ValueError("MANUS quaternion convention must be 'wxyz' or 'xyzw'")


def sdk_quaternion_wxyz_to_xyzw(quaternion_wxyz):
    """Convert the official ManusQuaternion field order (w,x,y,z) to xyzw."""
    return quaternion_to_xyzw(quaternion_wxyz, "wxyz")


def adapt_raw_skeleton(
    positions25,
    rotations,
    *,
    node_info=None,
    node_ids=None,
    scale_to_m=1.0,
    quaternion_convention="xyzw",
):
    """Adapt one WORLD/GLOBAL Raw Skeleton without rotating its quaternion."""
    if node_info is not None and len(node_info) > 0:
        mapping = semantic_standard21_mapping(node_info, node_ids)
        mapping_source = "NodeInfo"
    else:
        mapping, mapping_source = MANUS_TO_STANDARD21, "official-25 fallback"

    rotations = np.asarray(rotations, dtype=float)
    if rotations.shape != (25, 4) or not np.isfinite(rotations).all():
        raise ValueError("MANUS rotations must be finite with shape (25, 4)")
    thumb_tip_row = int(mapping[STANDARD21_THUMB_TIP])
    if mapping_source != "NodeInfo" and thumb_tip_row != FALLBACK_THUMB_TIP_ROW:
        raise AssertionError("Official MANUS 25 layout Thumb Tip must be row 4")
    orientation = quaternion_to_xyzw(rotations[thumb_tip_row], quaternion_convention)
    standard_world = convert_manus25_to_standard21(positions25, mapping, scale_to_m)
    normalized = relative_points(standard_world)
    if normalized is None:
        raise ValueError("MANUS hand geometry has a degenerate palm size")
    return AdaptedManusFrame(
        points=normalized,
        thumb_tip_orientation_world_xyzw=orientation,
        mapping=np.asarray(mapping).copy(),
        thumb_tip_row=thumb_tip_row,
        mapping_source=mapping_source,
    )
