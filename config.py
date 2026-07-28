from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARAMS_PATH = ROOT / "stereo_params.json"
URDF_PATH = ROOT / "assets/mmhand/urdf/hand.urdf"

# Stereo camera and human-hand reconstruction.
CAMERA_INDEX = 0
SINGLE_WIDTH, HEIGHT = 1280, 720
FULL_WIDTH = SINGLE_WIDTH * 2
ROTATE_LEFT, ROTATE_RIGHT = 180, -180
BOARD_SIZE, SQUARE_SIZE = (9, 6), 23.5
CALIBRATION_FRAMES = 100
BONE_TOLERANCE = 0.20
POINT_FILTER = (1.0, 0.01, 1.0)
ANGLE_FILTER = (1.0, 0.02, 1.0)
MAX_REPROJECTION_ERROR = 30.0
DEPTH_RANGE = (100.0, 1500.0)
MAX_HAND_RADIUS = 300.0
STALE_FRAMES, HAND_SWITCH_FRAMES = 3, 5
STANDARD_PALM_SIZE = 0.086

# MMHand J00...J20 order.
ROBOT_JOINT_NAMES = (
    "Little_MCP_AA", "Little_MCP_FE",
    "finger_4_distal_phalanx_1_PIP_Joint", "finger_4_fingertip_1_DIP_Joint",
    "Ring_MCP_AA", "Ring_MCP_FE",
    "finger_3_distal_phalanx_1_PIP_Joint", "finger_3_fingertip_1_DIP_Joint",
    "Middle_MCP_AA", "Middle_MCP_FE",
    "finger_2_distal_phalanx_1_PIP_Joint", "finger_2_fingertip_1_DIP_Joint",
    "Index_MCP_AA", "Index_MCP_FE",
    "finger_1_distal_phalanx_1_PIP_Joint", "finger_1_fingertip_1_DIP_Joint",
    "Thumb_MCP_AA", "Thumb_MCP_FE",
    "mmhand_thumb_1_finger_7_distal_phalanx_1_PIP_Joint",
    "mmhand_thumb_1_finger_7_fingertip_1_DIP_Joint", "Thumb_CMC",
)
# Four-finger A-A zero points (Little, Ring, Middle, Index). No spread offset.
MCP_AA_NEUTRAL_DEG = (36, 29, 31, 23)

# Human wrist->MCP/IP/TIP targets robot palm-origin->MCP/DIP/TIP.
MMHAND_PALM_ORIGIN = (-0.055636, -0.038440, -0.098073)
THUMB_POSITION_SCALING = (1.725623, 1.561178, 1.616345)
THUMB_POSITION_WEIGHTS = (2.177353, 0.424196, 1.082691)
THUMB_PAD_AXIS = (-0.200671, 0.970119, -0.136380)
THUMB_PAD_WEIGHT = 0.05
THUMB_MAX_EVAL = 40
THUMB_FTOL = 1e-5

# Runtime output and viewer.
KEYPOINT_TOPIC = "/hand/keypoints"
KEYPOINT_LAYOUT = "mvhc:keypoints:v1:palm_local_m:size=0.086"
ROBOT_TOPIC = "/raw_ik_target"
ROBOT_LAYOUT = "mmhand:J00-J20:urdf_deg"
WEB_PORT, VIEW_PORTS, WEB_FPS = 8080, (8081, 8082), 15
FINGER_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)
SKELETON_EDGES = tuple(
    edge
    for chain in FINGER_CHAINS
    for edge in zip(chain[:-1], chain[1:])
)
