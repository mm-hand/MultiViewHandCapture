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
# Joint Angle Limits (Physiological Constraints)
# Unit: Degrees
# Flexion: Bending towards the palm (Positive angles)
# Extension: Bending towards back of hand (Negative angles)
#
# NOTE: To prevent instability, we set strict limits for PIP/DIP.
# ==========================================================
JOINT_ANGLE_LIMITS = {
    "thumb": {
        # Thumb moves differently. These are approximations.
        "mcp": {"flexion": 60.0, "extension": 20.0},
        "ip":  {"flexion": 80.0, "extension": 20.0},
    },
    "index": {
        # MCP (Knuckle) can extend back significantly
        "mcp": {"flexion": 90.0,  "extension": 30.0}, 
        # PIP (Middle Joint) acts like a hinge, almost no extension
        "pip": {"flexion": 110.0, "extension": 0.0},  
        # DIP (Tip Joint)
        "dip": {"flexion": 90.0,  "extension": 0.0},  
    },
    "middle": {
        "mcp": {"flexion": 90.0,  "extension": 30.0},
        "pip": {"flexion": 110.0, "extension": 0.0},
        "dip": {"flexion": 90.0,  "extension": 0.0},
    },
    "ring": {
        "mcp": {"flexion": 90.0,  "extension": 30.0},
        "pip": {"flexion": 110.0, "extension": 0.0},
        "dip": {"flexion": 90.0,  "extension": 0.0},
    },
    "little": {
        "mcp": {"flexion": 95.0,  "extension": 40.0},
        "pip": {"flexion": 110.0, "extension": 0.0},
        "dip": {"flexion": 90.0,  "extension": 0.0},
    }
}