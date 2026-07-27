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
ENABLE_ANGLE_CONSTRAINTS = True
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
VECTOR_ORIGIN_LINKS = ("base_link",) * 16
VECTOR_TASK_LINKS = (
    "5-tip_Link", "1-tip_Link", "2-tip_Link", "3-tip_Link", "4-tip_Link",
    "mmhand_thumb_1_finger_7_fingertip_1",
    "finger_1_distal_phalanx_1", "finger_2_distal_phalanx_1",
    "finger_3_distal_phalanx_1", "finger_4_distal_phalanx_1",
    "mmhand_thumb_1_finger_7_distal_phalanx_1",
    "finger_1_proximal_phalanx_1", "finger_2_proximal_phalanx_1",
    "finger_3_proximal_phalanx_1", "finger_4_proximal_phalanx_1",
    "mmhand_thumb_1_finger_7_proximal_phalanx_1",
)
VECTOR_HUMAN_ORIGINS = (0,) * 16
VECTOR_HUMAN_TASKS = (4, 8, 12, 16, 20, 3, 7, 11, 15, 19, 2, 6, 10, 14, 18, 1)
OPERATOR2MANO_LEFT = ((0, 0, -1), (1, 0, 0), (0, -1, 0))
# Per-vector scales in VECTOR_TASK_LINKS order; retune after the URDF swap.
VECTOR_SCALING = (
    1.20, 1.20, 1.05, 1.05, 1.05,
    1.00, 1.40, 1.40, 1.40, 0.85,
    1.20, 0.85, 1.20, 1.35, 1.20, 1.20,
)
VECTOR_HUBER_DELTA = 0.02
VECTOR_NORM_DELTA = 0.004
VECTOR_LOW_PASS_ALPHA = 0.4
VECTOR_MAX_EVAL = 50

# Runtime output and viewer.
KEYPOINT_TOPIC = "/hand/keypoints"
KEYPOINT_LAYOUT = "mvhc:keypoints:v1:palm_local_m:size=0.086"
ROBOT_TOPIC = "/raw_ik_target"
ROBOT_LAYOUT = "mmhand:J00-J20:urdf_deg"
WEB_PORT, VIEW_PORTS, WEB_FPS = 8080, (8081, 8082), 15
SKELETON_EDGES = tuple(
    edge
    for chain in (
        (0, 1, 2, 3, 4),
        (0, 5, 6, 7, 8),
        (0, 9, 10, 11, 12),
        (0, 13, 14, 15, 16),
        (0, 17, 18, 19, 20),
    )
    for edge in zip(chain[:-1], chain[1:])
)
