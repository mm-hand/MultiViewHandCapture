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
