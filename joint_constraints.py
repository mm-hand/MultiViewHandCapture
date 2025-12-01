import numpy as np
from config import JOINT_ANGLE_LIMITS

# ==========================================================
# Data Structures
# ==========================================================
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_CHAINS = {
    "thumb":  [WRIST, THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP],
    "index":  [WRIST, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP],
    "middle": [WRIST, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP],
    "ring":   [WRIST, RING_MCP, RING_PIP, RING_DIP, RING_TIP],
    "little": [WRIST, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP]
}

JOINT_CONFIG_MAP = {
    2: ("thumb", "mcp"), 3: ("thumb", "ip"),
    5: ("index", "mcp"), 6: ("index", "pip"), 7: ("index", "dip"),
    9: ("middle", "mcp"), 10: ("middle", "pip"), 11: ("middle", "dip"),
    13: ("ring", "mcp"), 14: ("ring", "pip"), 15: ("ring", "dip"),
    17: ("little", "mcp"), 18: ("little", "pip"), 19: ("little", "dip"),
}

# ==========================================================
# Math Helpers
# ==========================================================

def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return np.array([0.0, 0.0, 0.0])
    return v / norm

def rotate_points_around_axis(points, pivot, axis, angle_deg):
    """
    Standard Rodrigues rotation.
    """
    axis = normalize(axis)
    if np.linalg.norm(axis) < 1e-6:
        return points

    theta = np.radians(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    v = points - pivot 
    cross_prod = np.cross(axis, v)
    dot_prod = np.sum(axis * v, axis=1)[:, np.newaxis]
    
    v_rot = (v * cos_t + 
             cross_prod * sin_t + 
             axis * dot_prod * (1.0 - cos_t))
    
    return v_rot + pivot

# ==========================================================
# Core Logic
# ==========================================================

def apply_joint_constraints(landmarks_3d):
    """
    Applies joint constraints using a STABLE reference frame derived from the palm.
    It avoids 'u x v' for axis calculation to prevent jitter/flipping at 0 degrees.
    """
    corrected = landmarks_3d.copy()

    # 1. Compute Palm Basis (Stable Global Frame for the Hand)
    p_wrist = corrected[WRIST]
    p_index_mcp = corrected[INDEX_MCP]
    p_pinky_mcp = corrected[PINKY_MCP]

    # Vector pointing from Wrist to Fingers
    vec_hand_dir = normalize(p_index_mcp - p_wrist)
    # Vector pointing from Index to Pinky (Transverse)
    vec_transverse = normalize(p_pinky_mcp - p_index_mcp)
    # Palm Normal (approximate, pointing out of back of hand)
    palm_normal = normalize(np.cross(vec_hand_dir, vec_transverse))

    for finger_name, indices in FINGER_CHAINS.items():
        # Iterate joints. Start from 1.
        for i in range(1, len(indices) - 1):
            idx_parent = indices[i-1]
            idx_pivot  = indices[i]
            idx_child  = indices[i+1]

            # Current Positions
            p_parent = corrected[idx_parent]
            p_pivot  = corrected[idx_pivot]
            p_child  = corrected[idx_child]

            # Bone Vectors
            u = p_pivot - p_parent  # Parent Bone
            v = p_child - p_pivot   # Child Bone
            
            u_norm = normalize(u)
            v_norm = normalize(v)

            # --- STABILITY FIX: Define the Hinge Axis ---
            # Instead of using (u x v) which is unstable when straight,
            # we calculate the Ideal Hinge Axis based on the Parent Bone + Palm Normal.
            # This ensures we rotate the finger along its natural track, not a random noise axis.
            
            if finger_name == "thumb":
                # Thumb mechanism is complex (saddle joint). 
                # Simplification: Axis is roughly aligned with Palm Normal for Flexion.
                hinge_axis = palm_normal
            else:
                # For fingers, the hinge axis is perpendicular to the bone and the palm normal.
                # Think of the pin in a door hinge. It points "sideways" relative to the finger.
                # hinge_axis = u_norm x palm_normal
                hinge_axis = normalize(np.cross(u_norm, palm_normal))

            # --- Calculate Signed Angle ---
            # We project v onto the plane perpendicular to hinge_axis to find the pure flexion angle.
            
            # Vector v projected onto the flexion plane
            # v_proj = v - (v . axis) * axis
            proj_v = v_norm - hinge_axis * np.dot(v_norm, hinge_axis)
            proj_v = normalize(proj_v)
            
            # The "Zero" vector (Straight finger) is just u_norm (projected? u is already perpendicular to axis ideally)
            # Let's verify u is perpendicular to hinge_axis.
            # hinge_axis = u x normal. Yes, u is perp to hinge_axis.
            ref_zero = u_norm
            
            # Calculate angle between ref_zero (Parent) and proj_v (Child projected)
            # Dot product
            dot_val = np.dot(ref_zero, proj_v)
            dot_val = np.clip(dot_val, -1.0, 1.0)
            angle_mag = np.degrees(np.arccos(dot_val))
            
            # Determine Sign using Cross Product relative to Hinge Axis
            # (ref_zero x proj_v) should be parallel to hinge_axis for Flexion?
            # Let's check: 
            # Fingers: u (fwd) x v (down/flex) -> Points Right (Same as hinge_axis derived from u x normal)
            cross_check = np.cross(ref_zero, proj_v)
            sign_check = np.dot(cross_check, hinge_axis)
            
            # Assign Sign
            signed_angle = angle_mag if sign_check > 0 else -angle_mag
            
            # --- Dead Zone for Stability ---
            # If angle is extremely small, ignore it to prevent jitter loops
            if abs(signed_angle) < 5.0:
                continue

            # --- Check Limits ---
            if idx_pivot not in JOINT_CONFIG_MAP:
                continue
                
            fname_cfg, joint_type = JOINT_CONFIG_MAP[idx_pivot]
            limits = JOINT_ANGLE_LIMITS[fname_cfg][joint_type]
            
            max_flex = limits["flexion"]
            max_ext  = limits["extension"] # Stored as positive magnitude

            angle_correction = 0.0

            if signed_angle > max_flex:
                # Flexed too much
                angle_correction = -(signed_angle - max_flex)
            elif signed_angle < -max_ext:
                # Extended too much (negative angle less than negative limit)
                # e.g., Angle -30, Limit 10 (Target -10). Diff = -10 - (-30) = +20
                angle_correction = (-max_ext) - signed_angle
            
            # --- Apply Correction ---
            if abs(angle_correction) > 0.5:
                # Use the STABLE hinge_axis for rotation, not the unstable u x v axis.
                # This prevents the finger from twisting sideways.
                
                indices_to_rotate = indices[i+1:]
                points_subset = corrected[indices_to_rotate]
                
                rotated_subset = rotate_points_around_axis(
                    points_subset,
                    p_pivot,
                    hinge_axis, # <--- The Key Fix
                    angle_correction
                )
                
                corrected[indices_to_rotate] = rotated_subset
                
    return corrected
