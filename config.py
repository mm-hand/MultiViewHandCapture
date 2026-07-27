from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARAMS_PATH = ROOT / "stereo_params.json"
URDF_PATH = ROOT / "assets/mmhand/urdf/hand.urdf"
URDF_CONTRACT_PATH = ROOT / "assets/mmhand/urdf_contract.json"

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

# dex-retargeting HKU Hand V2 ordinary Vector method.
ROBOT_JOINT_NAMES = tuple(
    [f"{finger}-{joint}" for finger in range(1, 5) for joint in range(1, 5)]
    + [f"5-{joint}" for joint in range(1, 6)]
)
# Robot order -> the existing MMHand J00...J20 ROS contract.
ROBOT_TO_CONTRACT = (12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3, 16, 17, 18, 19, 20)
VECTOR_ORIGIN_LINKS = ("base_link",) * 16
VECTOR_TASK_LINKS = (
    "5-tip_Link", "1-tip_Link", "2-tip_Link", "3-tip_Link", "4-tip_Link",
    "5-4_Link", "1-3_Link", "2-3_Link", "3-3_Link", "4-3_Link",
    "5-3_Link", "1-2_Link", "2-2_Link", "3-2_Link", "4-2_Link", "5-2_Link",
)
VECTOR_HUMAN_ORIGINS = (0,) * 16
VECTOR_HUMAN_TASKS = (4, 8, 12, 16, 20, 3, 7, 11, 15, 19, 2, 6, 10, 14, 18, 1)
OPERATOR2MANO_LEFT = ((0, 0, -1), (1, 0, 0), (0, -1, 0))
# Per-vector scales in VECTOR_TASK_LINKS order, tuned on all four local recordings.
VECTOR_SCALING = (
    1.05, 1.20, 1.05, 1.05, 1.05,
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
