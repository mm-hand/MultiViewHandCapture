from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARAMS_PATH = ROOT / "stereo_params.json"
URDF_PATH = ROOT / "assets/mmhand/urdf/hand.urdf"
HAND_LANDMARKER_PATH = ROOT / "assets/mediapipe/hand_landmarker.task"

# Camera: "d435" uses factory-calibrated stereo IR; "stereo" uses stereo_params.json.
CAMERA_TYPE = "stereo"
CAMERA_INDEX = 0
D435_WIDTH, D435_HEIGHT, D435_FPS = 1280, 720, 30

# Legacy side-by-side stereo camera and human-hand reconstruction.
SINGLE_WIDTH, HEIGHT = 1280, 720
FULL_WIDTH = SINGLE_WIDTH * 2
ROTATE_LEFT, ROTATE_RIGHT = 180, -180
BOARD_SIZE, SQUARE_SIZE = (9, 6), 23.5
MIN_CALIBRATION_PAIRS = 10
CALIBRATION_FRAMES, CALIBRATION_HZ = 100, 10
BONE_TOLERANCE = 0.20
POINT_FILTER = (1.0, 0.01, 1.0)
ANGLE_FILTER = (1.0, 0.02, 1.0)
MP_DETECTION_CONFIDENCE = 0.5
MP_PRESENCE_CONFIDENCE = 0.6
MP_TRACKING_CONFIDENCE = 0.6
MAX_REPROJECTION_ERROR = 30.0
MAX_DEPTH_MM = 1500.0
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
# Mechanical A-A neutral positions; four-finger order is Little, Ring, Middle, Index.
MCP_AA_NEUTRAL_DEG = (36, 29, 31, 23)
THUMB_MCP_AA_NEUTRAL_DEG = 28

# Thumb tip position, tip-to-fingertip vectors, and pad direction.
MMHAND_PALM_ORIGIN = (-0.083953, -0.037473, -0.047264)
THUMB_TIP_SCALE = 0.614962
THUMB_TIP_WEIGHT = 7.871241
THUMB_PAD_AXIS = (-0.200671, 0.970119, -0.136380)
# Fingertip-link local pad normals; order is Index, Middle, Ring, Little.
FINGER_PAD_AXES = (
    (-0.131523, 0.099710, -0.986286),
    (-0.160095, 0.116825, -0.980164),
    (-0.180981, 0.121096, -0.976003),
    (-0.180981, 0.121096, -0.976003),
)
THUMB_PAD_WEIGHT = 1.286285
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
