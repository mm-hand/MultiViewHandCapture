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
MCP_AA_NEUTRAL_DEG = (36, 29, 31, 23, 28)  # Little, Ring, Middle, Index, Thumb.
# Per-vector neutral-frame rotations, calibrated from left_tune.mkv.
VECTOR_ROTATION_VECS = (
    (0.89959769, 0.89247362, 0.06387778),
    (0.04028359, 0.21695396, 0.01624794),
    (0.07462502, 0.16987143, -0.08575732),
    (0.11353107, 0.13740476, -0.10540530),
    (0.17198139, 0.18717731, -0.35970889),
    (-0.01271927, 0.04123630, 0.25290847),
    (-0.00987920, -0.04184901, 0.06527924),
    (0.01529810, 0.02884924, 0.19071928),
    (0.05014359, -0.03532022, 0.05490010),
    (0.04018065, -0.17008292, -0.09821931),
    (0.06709930, -0.14495640, 0.04072943),
    (0.12941539, -0.11712640, -0.00382518),
    (0.12571349, -0.24494515, -0.13678874),
    (0.12274756, -0.17488518, 0.06258845),
    (0.27346888, -0.10055991, -0.52715215),
    (0.24398395, -0.11706779, -0.60769274),
    (0.21385018, 0.01607406, -0.42385572),
    (1.31574394, 0.80159376, -0.28817246),
    (1.38102472, 0.41049415, -0.80207055),
    (1.37108473, 1.02603933, -0.15475385),
    (0.45963209, -0.72573070, -0.21603680),
    (0.64158565, -0.62079319, -0.27023846),
    (0.78484956, -0.44376477, -0.27969633),
    (0.87613931, -0.25559640, -0.46282112),
)
# Neutral length ratios for the rotated vector targets.
VECTOR_SCALING = (
    1.646325, 1.153011, 1.119575, 1.168372, 1.253057,
    1.033171, 1.180150, 1.165999,
    1.003589, 1.171999, 1.209110,
    1.072414, 1.280615, 1.233337,
    1.513199, 1.593462, 1.419369,
    1.529716, 1.079202, 1.056938,
    1.683070, 1.403126, 1.329913, 1.184692,
)
# Static weights tuned on all five recordings.
VECTOR_WEIGHTS = (
    0.091634, 0.088771, 0.085877, 0.165907, 0.080864,
    1.453033, 2.940538, 0.195484,
    1.599245, 1.700529, 0.428513,
    1.719218, 1.961359, 0.278395,
    1.718025, 0.884504, 0.148249,
    0.145976, 3.588109, 3.802986,
    0.366537, 0.187289, 0.185422, 0.183535,
)
THUMB_BEND_OFFSET_DEG = (9.388779, 6.408744)
THUMB_ANGLE_WEIGHTS = (0.5, 0.02)
THUMB_ANGLE_HUBER_DEG = 10.0
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
