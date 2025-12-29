# ==========================================================
# Stereo Camera Config (side-by-side: left eye + right eye)
# ==========================================================
CAMERA_INDEX = 2              # Default camera device index
SINGLE_WIDTH  = 1280          # Width of one camera image
HEIGHT        = 720           # Height of one camera image
FULL_WIDTH    = SINGLE_WIDTH * 2  # Stereo image total width (e.g. 2560 for 10x70 * 2)

# Calibration settings
BOARD_SIZE    = (9, 6)        # Checkerboard pattern size
SQUARE_SIZE   = 23.5          # Checkerboard square size in mm

# Calibration parameters
CALIBRATION_FRAMES = 100      # Number of frames for bone length calibration

# Image rotation parameters
# Positive values: clockwise rotation, Negative values: counterclockwise rotation
ROTATE_LEFT = 0              # Rotation angle for left camera image (degrees)
ROTATE_RIGHT = 0             # Rotation angle for right camera image (degrees)