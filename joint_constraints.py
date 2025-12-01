import numpy as np
from config import JOINT_ANGLE_LIMITS

# MediaPipe Hands 21-point landmark index mapping
# Format: landmark_name: index
MP_JOINTS = {
    "wrist": 0,

    # Thumb
    "thumb_cmc": 1,
    "thumb_mcp": 2,
    "thumb_ip": 3,
    "thumb_tip": 4,

    # Index
    "index_mcp": 5,
    "index_pip": 6,
    "index_dip": 7,
    "index_tip": 8,

    # Middle
    "middle_mcp": 9,
    "middle_pip": 10,
    "middle_dip": 11,
    "middle_tip": 12,

    # Ring
    "ring_mcp": 13,
    "ring_pip": 14,
    "ring_dip": 15,
    "ring_tip": 16,

    # Little
    "little_mcp": 17,
    "little_pip": 18,
    "little_dip": 19,
    "little_tip": 20,
}

# Define joint connections for angle computation
# Each tuple: (parent_joint_name, joint_name, child_joint_name)
# The angle is computed at joint_name from parent → joint → child
JOINT_CONNECTIONS = [
    ("thumb_cmc",   "thumb_mcp",   "thumb_ip"),
    ("thumb_mcp",   "thumb_ip",    "thumb_tip"),

    ("index_mcp",   "index_pip",   "index_dip"),
    ("index_pip",   "index_dip",   "index_tip"),

    ("middle_mcp",  "middle_pip",  "middle_dip"),
    ("middle_pip",  "middle_dip",  "middle_tip"),

    ("ring_mcp",    "ring_pip",    "ring_dip"),
    ("ring_pip",    "ring_dip",    "ring_tip"),

    ("little_mcp",  "little_pip",  "little_dip"),
    ("little_pip",  "little_dip",  "little_tip"),
]

def vector_angle(v1, v2):
    """
    Compute angle between vectors v1 and v2 in degrees.
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

def clamp_angle(joint_name, angle):
    """
    Clamp a joint angle within its flexion/extension limits from config.py
    """
    limits = JOINT_ANGLE_LIMITS.get(joint_name)
    if not limits:
        return angle

    flexion_limit = limits["flexion"]
    extension_limit = limits["extension"]

    if angle > 0:  # flexion
        return min(angle, flexion_limit)
    else:  # extension
        return max(angle, -extension_limit)

def apply_joint_constraints(landmarks_3d):
    """
    Apply joint angle constraints to 3D hand landmarks.

    Parameters:
    landmarks_3d : np.ndarray of shape (21, 3)
        3D coordinates of MediaPipe hand landmarks.

    Returns:
    np.ndarray of shape (21, 3)
        Adjusted coordinates satisfying joint angle limits.
    """
    adjusted_landmarks = landmarks_3d.copy()

    for parent_name, joint_name, child_name in JOINT_CONNECTIONS:
        parent_idx = MP_JOINTS[parent_name]
        joint_idx = MP_JOINTS[joint_name]
        child_idx = MP_JOINTS[child_name]

        v1 = adjusted_landmarks[parent_idx] - adjusted_landmarks[joint_idx]
        v2 = adjusted_landmarks[child_idx] - adjusted_landmarks[joint_idx]

        angle = vector_angle(v1, v2)

        # For simplicity, we treat angles > 90° as flexion
        if angle > 90:
            signed_angle = angle - 90
        else:
            signed_angle = -(90 - angle)  # extension

        clamped_angle = clamp_angle(joint_name, signed_angle)

        # Compute angle difference
        delta_angle = clamped_angle - signed_angle
        if abs(delta_angle) > 1e-3:
            # Rotate child segment to enforce limit
            adjusted_landmarks[child_idx] = rotate_point_around_joint(
                adjusted_landmarks[child_idx],
                adjusted_landmarks[joint_idx],
                v1,
                delta_angle
            )

    return adjusted_landmarks

def rotate_point_around_joint(point, joint_pos, axis_vec, delta_deg):
    """
    Rotate a 3D point around a joint axis by delta_deg degrees.
    Axis defined by axis_vec (from joint to parent).
    """
    # Normalize axis
    axis = axis_vec / np.linalg.norm(axis_vec)
    theta = np.radians(delta_deg)

    # Rodrigues' rotation formula
    p = point - joint_pos
    p_rot = (p * np.cos(theta) +
             np.cross(axis, p) * np.sin(theta) +
             axis * np.dot(axis, p) * (1 - np.cos(theta)))
    return p_rot + joint_pos
