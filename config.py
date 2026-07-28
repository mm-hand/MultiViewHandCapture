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
)
OPERATOR2MANO_LEFT = ((0, 0, -1), (1, 0, 0), (0, -1, 0))
MCP_AA_NEUTRAL_DEG = (36, 29, 31, 23, 28)  # Little, Ring, Middle, Index, Thumb.
MCP_AA_SAFE_OFFSET_DEG = (-8, -1.5, 1.5, 6, 0)
# Per-vector neutral-frame rotations, calibrated from left_tune.mkv.
VECTOR_ROTATION_VECS = (
    (0.89959769, 0.89247362, 0.06387778),
    (0.03364310, 0.21702190, -0.03535082),
    (0.07256866, 0.16941199, -0.09898844),
    (0.11578148, 0.13862215, -0.09216581),
    (0.18363117, 0.19713661, -0.28541225),
    (-0.01068331, 0.04180138, 0.14820198),
    (-0.01207190, -0.04126587, -0.03942400),
    (-0.00073636, 0.03016983, 0.08878984),
    (0.04963234, -0.03596850, 0.02872912),
    (0.03795568, -0.17063525, -0.12433219),
    (0.06091129, -0.14574080, 0.01566078),
    (0.13097981, -0.11541941, 0.02228553),
    (0.12887853, -0.24321259, -0.11077493),
    (0.12929766, -0.17279371, 0.08750726),
    (0.27852702, -0.08073565, -0.38852776),
    (0.24999889, -0.09913104, -0.46891951),
    (0.23369354, 0.03741923, -0.28889244),
    (1.31574394, 0.80159376, -0.28817246),
    (1.38102472, 0.41049415, -0.80207055),
    (1.37108473, 1.02603933, -0.15475385),
)
# Neutral length ratios for the rotated vector targets.
VECTOR_SCALING = (
    1.646325, 1.162209, 1.120428, 1.168234, 1.267343,
    1.033171, 1.180150, 1.165999,
    1.003589, 1.171999, 1.209110,
    1.072414, 1.280615, 1.233337,
    1.513199, 1.593462, 1.419369,
    1.529716, 1.079202, 1.056938,
)
# Static weights tuned on all five recordings.
VECTOR_WEIGHTS = (
    0.050651, 0.091204, 0.088231, 0.170454, 0.083080,
    1.492859, 3.021134, 0.200842,
    1.643078, 1.747138, 0.440258,
    1.766340, 2.015117, 0.286025,
    1.765114, 0.908747, 0.152312,
    0.149977, 0.020217, 3.907221,
)
THUMB_NEUTRAL_BEND_DEG = (4.308824, 7.236667)
THUMB_ANGLE_WEIGHTS = (4.0, 4.0)
THUMB_ANGLE_HUBER_DEG = 10.0
OPPOSITION_SCALING = (1.683070, 1.403126, 1.329913, 1.184692)
OPPOSITION_WEIGHTS = (4.0, 4.0, 4.0, 4.0)
OPPOSITION_HUBER_DELTA = 0.01
TIP_RADII = (0.009307970, 0.009262807, 0.009286031, 0.009249611, 0.009249611)
TIP_CLEARANCE = 0.0015
FINGER_SAFETY_WEIGHT = 10.0
FINGER_SAFETY_HUBER = 0.005
THUMB_PAD_AXIS = (-0.200671, 0.970119, -0.136380)
THUMB_PAD_WEIGHT = 0.025
THUMB_PAD_HUBER_DEG = 20.0
THUMB_PAD_GATE = (0.10, 0.35)
VECTOR_HUBER_DELTA = 0.02
VECTOR_NORM_DELTA = 0.004
VECTOR_MAX_EVAL = 100
VECTOR_FTOL = 1e-4

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
