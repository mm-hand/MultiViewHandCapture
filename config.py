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
# (human origin, human task, robot origin link, robot task link).
VECTOR_MAP = (
    (0, 4, "base_link", "5-tip_Link"),
    (0, 8, "base_link", "1-tip_Link"),
    (0, 12, "base_link", "2-tip_Link"),
    (0, 16, "base_link", "3-tip_Link"),
    (0, 20, "base_link", "4-tip_Link"),
    (5, 6, "finger_1_proximal_phalanx_1", "finger_1_distal_phalanx_1"),
    (6, 7, "finger_1_distal_phalanx_1", "finger_1_fingertip_1"),
    (7, 8, "finger_1_fingertip_1", "1-tip_Link"),
    (9, 10, "finger_2_proximal_phalanx_1", "finger_2_distal_phalanx_1"),
    (10, 11, "finger_2_distal_phalanx_1", "finger_2_fingertip_1"),
    (11, 12, "finger_2_fingertip_1", "2-tip_Link"),
    (13, 14, "finger_3_proximal_phalanx_1", "finger_3_distal_phalanx_1"),
    (14, 15, "finger_3_distal_phalanx_1", "finger_3_fingertip_1"),
    (15, 16, "finger_3_fingertip_1", "3-tip_Link"),
    (17, 18, "finger_4_proximal_phalanx_1", "finger_4_distal_phalanx_1"),
    (18, 19, "finger_4_distal_phalanx_1", "finger_4_fingertip_1"),
    (19, 20, "finger_4_fingertip_1", "4-tip_Link"),
    (
        1, 2,
        "mmhand_thumb_1_thumb_abduction_adduction_link_1",
        "mmhand_thumb_1_finger_7_distal_phalanx_1",
    ),
    (
        2, 3,
        "mmhand_thumb_1_finger_7_distal_phalanx_1",
        "mmhand_thumb_1_finger_7_fingertip_1",
    ),
    (3, 4, "mmhand_thumb_1_finger_7_fingertip_1", "5-tip_Link"),
    (4, 8, "5-tip_Link", "1-tip_Link"),
    (4, 12, "5-tip_Link", "2-tip_Link"),
    (4, 16, "5-tip_Link", "3-tip_Link"),
    (4, 20, "5-tip_Link", "4-tip_Link"),
)
OPERATOR2MANO_LEFT = ((0, 0, -1), (1, 0, 0), (0, -1, 0))
# Geometric scales from all recordings and the open-hand segment in left_tune.mkv.
VECTOR_SCALING = (
    1.697215, 1.090577, 1.062837, 1.133296, 1.271152,
    1.245351, 1.332252, 1.363307,
    1.057328, 1.197269, 1.260720,
    1.123498, 1.255542, 1.261153,
    1.369040, 1.535150, 1.461806,
    1.600343, 1.008299, 1.049378,
    1.367503, 1.205160, 1.186174, 1.145264,
)
# Static weights tuned on all five recordings.
VECTOR_WEIGHTS = (
    0.245057, 0.237399, 0.251608, 0.489318, 0.960855,
    0.980228, 1.920096, 1.826812,
    1.897360, 1.973788, 0.980228,
    1.028292, 0.999209, 0.980228,
    0.980228, 0.515148, 2.021833,
    0.527158, 0.258860, 1.005382,
    0.980228, 0.980228, 0.980228, 0.980228,
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
