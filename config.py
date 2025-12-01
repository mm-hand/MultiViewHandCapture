# ==========================================================
# Stereo Camera Config (side-by-side: left eye + right eye)
# ==========================================================
CAMERA_INDEX = 0              # Default camera device index
SINGLE_WIDTH  = 1280          # Width of one camera image
HEIGHT        = 720           # Height of one camera image
FULL_WIDTH    = SINGLE_WIDTH * 2  # Stereo image total width (e.g. 2560 for 1280x720 * 2)

# Calibration settings
BOARD_SIZE    = (9, 6)        # Checkerboard pattern size
SQUARE_SIZE   = 23.5          # Checkerboard square size in mm

# Calibration parameters
CALIBRATION_FRAMES = 100      # Number of frames for bone length calibration

# ==========================================================
# Joint angle limits (in degrees)
# Positive = flexion, Negative = extension
# Names match MediaPipe joint naming in JOINT_CONNECTIONS mapping
# ==========================================================
JOINT_ANGLE_LIMITS = {
    # Thumb
    "thumb_mcp": {"flexion": 60, "extension": 10},
    "thumb_ip":  {"flexion": 80, "extension": 20},

    # Index
    "index_mcp": {"flexion": 90, "extension": 45},
    "index_pip": {"flexion": 110, "extension": 10},
    "index_dip": {"flexion": 80, "extension": 10},

    # Middle
    "middle_mcp": {"flexion": 90, "extension": 45},
    "middle_pip": {"flexion": 110, "extension": 10},
    "middle_dip": {"flexion": 80, "extension": 10},

    # Ring
    "ring_mcp": {"flexion": 90, "extension": 45},
    "ring_pip": {"flexion": 110, "extension": 10},
    "ring_dip": {"flexion": 80, "extension": 10},

    # Little
    "little_mcp": {"flexion": 90, "extension": 45},
    "little_pip": {"flexion": 110, "extension": 10},
    "little_dip": {"flexion": 80, "extension": 10},
}
