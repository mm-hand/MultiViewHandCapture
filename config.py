from pathlib import Path

# 仓库根目录；仅用于生成下面的资源路径。
_ROOT = Path(__file__).resolve().parent
# 普通双目棋盘标定结果；由 vision/calibrate.py 生成。
PARAMS_PATH = _ROOT / "vision/stereo_params.json"
# 正式 MMHand URDF 路径；FK、关节限位和 Viser 显示都读取该文件。
URDF_PATH = _ROOT / "assets/mmhand/urdf/hand.urdf"
# MediaPipe Tasks HandLandmarker 模型路径。
HAND_LANDMARKER_PATH = _ROOT / "vision/assets/hand_landmarker.task"


# Camera ----------------------------------------------------------------------

# 输入来源："stereo"、"d435"，未来可扩展为 "manus"。
INPUT_SOURCE = "stereo"
# 普通相机的 OpenCV 编号，或 D435 的设备枚举序号。
CAMERA_INDEX = 0
# D435 单路红外图像宽度，单位 pixel。
D435_WIDTH = 1280
# D435 单路红外图像高度，单位 pixel。
D435_HEIGHT = 720
# D435 双红外流目标帧率，单位 frame/s。
D435_FPS = 30
# 普通双目单路尺寸；相机输出左右横向拼接的 MJPEG 图像。
SINGLE_WIDTH, HEIGHT = 1280, 720
FULL_WIDTH = SINGLE_WIDTH * 2
# 拆分后左右图像的顺时针旋转角度，单位 degree。
ROTATE_LEFT, ROTATE_RIGHT = 180, -180
# 普通双目标定板内角点数、格长（mm）和最少有效图像对数。
BOARD_SIZE, SQUARE_SIZE = (9, 6), 23.5
MIN_CALIBRATION_PAIRS = 10

# Hand calibration and filtering ----------------------------------------------

# 启动时用于估计人手骨长中位数的有效样本数量。
CALIBRATION_FRAMES = 10 # TODO
# 骨长标定的最大采样频率，单位 Hz。
CALIBRATION_HZ = 10
# 每根骨相对标定长度允许变化的比例；0.20 表示允许 ±20%。
BONE_TOLERANCE = 0.20

# 二维像素点 One Euro 参数：
# (最低截止频率 Hz, 速度自适应系数 Hz/(pixel/s), 速度截止频率 Hz)。
POINT_2D_FILTER = (
    1.0,   # min_cutoff：静止时的平滑强度，越小越稳定但延迟越大。
    0.01,  # beta：像素运动越快时增加截止频率的幅度，越大越灵敏。
    1.0,   # derivative_cutoff：像素速度估计的低通截止频率。
)
# 三维毫米点 One Euro 参数：
# (最低截止频率 Hz, 速度自适应系数 Hz/(mm/s), 速度截止频率 Hz)。
POINT_3D_FILTER = (
    0.5,    # min_cutoff：静止三维点的平滑强度。
    0.001,  # beta：三维点运动时提高响应速度的系数。
    1.0,    # derivative_cutoff：三维点速度估计的低通截止频率。
)
# 屈曲角 One Euro 参数：
# (最低截止频率 Hz, 速度自适应系数 Hz/(degree/s), 速度截止频率 Hz)。
ANGLE_FILTER = (
    1.0,   # min_cutoff：静止关节角的平滑强度。
    0.02,  # beta：关节运动时提高响应速度的系数。
    1.0,   # derivative_cutoff：角速度估计的低通截止频率。
)


# MediaPipe and geometry gates ------------------------------------------------

# MediaPipe 初次手掌检测最低置信度，范围 0～1。
MP_DETECTION_CONFIDENCE = 0.5
# MediaPipe 判断当前 ROI 中仍存在手的最低置信度，范围 0～1。
MP_PRESENCE_CONFIDENCE = 0.6
# MediaPipe 跟踪框最低 IoU 置信度，范围 0～1；低于它会重新检测手掌。
MP_TRACKING_CONFIDENCE = 0.6
# 左右三角化点的最大平均重投影误差，单位 pixel。
MAX_REPROJECTION_ERROR = 30.0
# 任一三维点允许的最大相机 Z 坐标，单位 mm；当前不设置近距离下限。
MAX_DEPTH_MM = 1500.0
# 任一关键点距离腕点允许的最大半径，单位 mm。
MAX_HAND_RADIUS = 300.0
# 连续坏帧期间最多保留上一有效结果的帧数；下一帧会清空并重置滤波器。
STALE_FRAMES = 3
# 切换稳定 Left/Right 手性标签所需的连续一致帧数。
HAND_SWITCH_FRAMES = 5
# 标准化人手的目标掌尺寸，单位 m；用于 keypoint_relative 和 retarget。
STANDARD_PALM_SIZE = 0.086


# MMHand joint mapping ---------------------------------------------------------

# MMHand ROS/FK 数组顺序；每个元素对应固定的 J00～J20。
ROBOT_JOINT_NAMES = (
    "Little_MCP_AA",                                  # J00：小指 MCP 外展/内收。
    "Little_MCP_FE",                                  # J01：小指 MCP 屈曲/伸展。
    "finger_4_distal_phalanx_1_PIP_Joint",            # J02：小指 PIP 屈曲。
    "finger_4_fingertip_1_DIP_Joint",                 # J03：小指 DIP 屈曲。
    "Ring_MCP_AA",                                    # J04：无名指 MCP 外展/内收。
    "Ring_MCP_FE",                                    # J05：无名指 MCP 屈曲/伸展。
    "finger_3_distal_phalanx_1_PIP_Joint",            # J06：无名指 PIP 屈曲。
    "finger_3_fingertip_1_DIP_Joint",                 # J07：无名指 DIP 屈曲。
    "Middle_MCP_AA",                                  # J08：中指 MCP 外展/内收。
    "Middle_MCP_FE",                                  # J09：中指 MCP 屈曲/伸展。
    "finger_2_distal_phalanx_1_PIP_Joint",            # J10：中指 PIP 屈曲。
    "finger_2_fingertip_1_DIP_Joint",                 # J11：中指 DIP 屈曲。
    "Index_MCP_AA",                                   # J12：食指 MCP 外展/内收。
    "Index_MCP_FE",                                   # J13：食指 MCP 屈曲/伸展。
    "finger_1_distal_phalanx_1_PIP_Joint",            # J14：食指 PIP 屈曲。
    "finger_1_fingertip_1_DIP_Joint",                 # J15：食指 DIP 屈曲。
    "Thumb_MCP_AA",                                   # J16：拇指 MCP 外展/内收。
    "Thumb_MCP_FE",                                   # J17：拇指 MCP 屈曲/伸展。
    "mmhand_thumb_1_finger_7_distal_phalanx_1_PIP_Joint",  # J18：拇指 PIP 屈曲。
    "mmhand_thumb_1_finger_7_fingertip_1_DIP_Joint",       # J19：拇指 DIP 屈曲。
    "Thumb_CMC",                                      # J20：拇指 CMC 旋转。
)
# MMHand retarget --------------------------------------------------------------

# 掌原点到拇指尖的人手向量缩放系数；1.0 表示不缩放。
RETARGET_THUMB_TIP_SCALE = 1.0
# 掌原点到拇指尖位置误差的损失权重。
RETARGET_THUMB_TIP_WEIGHT = 1.0
# 人手 MCP→IP 对应 MMHand 拇指中节单位方向误差权重。
RETARGET_THUMB_MCP_IP_WEIGHT = 5.0
# 人手 IP→TIP 对应 MMHand 拇指远节单位方向误差权重。
RETARGET_THUMB_IP_TIP_WEIGHT = 1.0
# 最终 21 个 MMHand 输出角的 One Euro 参数，角度单位 degree。
RETARGET_ANGLE_FILTER = (1.0, 0.02, 1.0)
# 每帧最多计算的不同 SLSQP 关节候选数；超限时采用最低损失候选。
RETARGET_MAX_EVALUATIONS = 30
# SLSQP 的目标和步长停止精度。
RETARGET_FTOL = 3e-5


# Runtime output and viewer ----------------------------------------------------

# 标准化 21 点 ROS 2 topic 名称。
KEYPOINT_TOPIC = "/hand/keypoints"
# 关键点 Float32MultiArray layout 标签：掌局部坐标、单位 m、掌尺寸 0.086 m。
KEYPOINT_LAYOUT = "mvhc:keypoints:v1:palm_local_m:size=0.086"
# MMHand 21 关节原始 IK 目标 ROS 2 topic 名称。
ROBOT_TOPIC = "/raw_ik_target"
# 机器人 Float32MultiArray layout 标签：J00～J20 顺序、单位 degree。
ROBOT_LAYOUT = "mmhand:J00-J20:urdf_deg"
# 汇总左右图和两个 Viser iframe 的 HTTP dashboard 端口。
WEB_PORT = 8080
# 两个 Viser 服务端口：(标准化人手窗口, MMHand 机器人窗口)。
VIEW_PORTS = (
    8081,  # 第 1 项：标准化人手 Viser 端口。
    8082,  # 第 2 项：MMHand Viser 端口。
)
# Dashboard JPEG 和状态文本的最大刷新频率，单位 Hz。
WEB_FPS = 15
