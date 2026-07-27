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
MAX_REPROJECTION_ERROR = 8.0
DEPTH_RANGE = (100.0, 1500.0)
MAX_HAND_RADIUS = 300.0
STALE_FRAMES, HAND_SWITCH_FRAMES = 3, 5
STANDARD_PALM_SIZE = 0.086

# MMHand semantics. q always follows the contract's J00...J20 order.
ROBOT_BASE_LINK = "base_link"
ROBOT_CHAINS = {
    "pinky": ((17, 18, 19, 20), (0, 1, 2, 3)),
    "ring": ((13, 14, 15, 16), (4, 5, 6, 7)),
    "middle": ((9, 10, 11, 12), (8, 9, 10, 11)),
    "index": ((5, 6, 7, 8), (12, 13, 14, 15)),
    "thumb": ((1, 2, 3, 4), (20, 16, 17, 18, 19)),
}
ROBOT_TIP_LINKS = {
    "pinky": "finger_4_fingertip_1",
    "ring": "finger_3_fingertip_1",
    "middle": "finger_2_fingertip_1",
    "index": "finger_1_fingertip_1",
    "thumb": "mmhand_thumb_1_finger_7_fingertip_1",
}
ROBOT_TIP_OFFSETS = {
    "pinky": (0.0275333939, 0.0199599086, -0.0025148951),
    "ring": (0.0275333939, 0.0199599086, -0.0025148951),
    "middle": (0.0292021721, 0.0182175597, -0.0025476633),
    "index": (0.0295156873, 0.0177043729, -0.0025476633),
    "thumb": (0.0174200966, -0.0060131696, -0.0291124871),
}
ROBOT_FE_INDICES = (1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19)
HUMAN_TO_ROBOT_PALM = ((0, 0, 1), (0, 1, 0), (1, 0, 0))

# Full-hand finite-difference pseudoinverse IK.
IK_WEIGHTS = {"local": 1.0, "anchor_tip": 0.2, "hand_tip": 0.2, "thumb_tip": 1.0}
IK_ITERATIONS = 6
IK_EPSILON = 1e-4
IK_PINV_RCOND = 1e-5
IK_MAX_STEP = 0.18
IK_STOP_RMS = 1e-3
IK_MAX_RMS = 0.75
IK_NO_IMPROVEMENT = 2
JOINT_FILTER = (1.0, 0.02, 1.0)

# Runtime output and viewer.
KEYPOINT_TOPIC = "/hand/keypoints"
KEYPOINT_LAYOUT = "mvhc:keypoints:v1:palm_local_m:size=0.086"
WEB_PORT, VIEW_PORTS, WEB_FPS = 8080, (8081, 8082, 8083), 15
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
