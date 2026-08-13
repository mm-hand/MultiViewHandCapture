from pathlib import Path

_ROOT = Path(__file__).resolve().parent
URDF_PATH = _ROOT / "assets/mmhand/urdf/mmhand_collision_coacd.urdf"
WILOR_ASSET_DIR = _ROOT / "input/wilor/assets"


# Input -----------------------------------------------------------------------

INPUT_SOURCE = "wilor"  # "wilor" or "manus"
CAMERA_DEVICE = "/dev/video4"  # V4L2 index or /dev/videoN path
CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS = 640, 480, 30
WILOR_DEVICE_ID = 0
WILOR_CONFIDENCE = 0.30
WILOR_IOU = 0.50
WILOR_DETECT_EVERY = 3
WILOR_CROP_FACTOR = 2.0
WILOR_THUMB_PAD_ROTATION_DEG = 45.0
WILOR_POINT_FILTER = (0.5, 1.0, 1.0)
WILOR_DIRECTION_FILTER = (0.5, 0.25, 1.0)
STANDARD_PALM_SIZE = 0.086


# MANUS ------------------------------------------------------------------------

# 官方 MANUS Core SDK 3.1.1 Integrated runtime 及本项目薄封装。
MANUS_SDK_VERSION = "3.1.1"
MANUS_SDK_BRIDGE_PATH = _ROOT / "input/manus/assets/libmanus_sdk_bridge.so"
# bridge 的 CoordinateSystemVUH.unitScale=1.0，官方定义为 meter。
MANUS_POSITION_SCALE_TO_M = 1.0
# Maximum age of a MANUS frame, in seconds.
MANUS_STALE_SECONDS = 0.20
# Outward finger-pad axis in every MANUS Tip node's local frame.
MANUS_PAD_LOCAL_AXIS = (0.0, 0.0, -1.0)


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
# 人手拇指 MCP/IP 无符号正角的独立缩放及 MMHand J18/J19 损失权重。
RETARGET_THUMB_MCP_ANGLE_SCALE = 2.0
RETARGET_THUMB_IP_ANGLE_SCALE = 3.0
RETARGET_THUMB_MCP_ANGLE_WEIGHT = 1.0
RETARGET_THUMB_IP_ANGLE_WEIGHT = 1.0
# 拇指尖到食指、中指、无名指、小指尖四条完整相对向量的损失权重。
RETARGET_THUMB_TO_FINGERTIPS_WEIGHT = 2.0
# Human-to-MMHand thumb pad direction loss weight.
RETARGET_THUMB_PAD_WEIGHT = 2.0
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
KEYPOINT_LAYOUT = "mmhand_teleop:keypoints:v1:palm_local_m:size=0.086"
PAD_DIRECTION_TOPIC = "/hand/finger_pad_directions"
PAD_DIRECTION_LAYOUT = "mmhand_teleop:finger_pad_directions:v1:palm_local_unit"
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
