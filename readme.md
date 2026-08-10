# MultiView Hand Capture

从同步双目图像恢复 MediaPipe 21 点三维手部关键点，并将稳定左手实时映射到
MMHand 的 21 个关节。项目支持普通拼接双目和 Intel RealSense D435，共用同一套
检测、三角化、滤波、retarget、Web 显示和 ROS 2 输出流程。

## 安装

```bash
git clone https://github.com/mm-hand/MultiViewHandCapture.git
cd MultiViewHandCapture
mamba env create -f environment.yml
conda activate mmhand
```

更新已有环境：

```bash
mamba env update -n mmhand -f environment.yml --prune
conda activate mmhand
```

MediaPipe Tasks 的 HandLandmarker 模型已包含在
`assets/mediapipe/hand_landmarker.task`，运行时无需下载。

## 快速开始

所有运行参数集中在 `config.py`。

### 普通拼接双目

普通模式由 `camera.StereoCamera` 读取单个 `2560×720` MJPEG 设备，拆成两个
`1280×720` 图像。首次使用、移动镜头或改变分辨率后需要重新标定：

```python
CAMERA_TYPE = "stereo"
CAMERA_INDEX = 0
```

```bash
python calibrate.py
```

标定窗口中按 `C` 保存有效棋盘图像对，按 `Q` 求解。默认棋盘为 `9×6` 内角点、
格长 `23.5 mm`，至少需要 10 对；结果写入不提交的 `stereo_params.json`。
`pattern.png` 可用于打印标定板，但实际格长必须与 `SQUARE_SIZE` 一致。

### RealSense D435

```python
CAMERA_TYPE = "d435"
CAMERA_INDEX = 0
```

D435 模式读取同步的 `infrared 1/2` Y8 流，使用 SDK 提供的双目标定，不启用深度流，
也不需要运行 `calibrate.py`。

### 启动

```bash
python track.py
python track.py --ros  # 同时发布 ROS 2 topic
```

打开 `http://localhost:8080`：

- 上方显示左右双目图像；加权 retarget 损失及占比浮动在右上角。
- 左下显示 `Normalized hand`，包含人手 retarget 坐标轴和原点。
- 右下显示 MMHand URDF，包含 MMHand 掌坐标轴和原点。

右手可以跟踪、显示和发布关键点，但 MMHand retarget 只处理完成骨长标定的稳定左手。

## 配置

`config.py` 中每个参数都有单位和用途注释。常用分组如下：

| 分组 | 参数 |
|---|---|
| 相机 | `CAMERA_TYPE`、`CAMERA_INDEX`、图像尺寸、旋转角、D435 帧率 |
| 标定与约束 | `BOARD_SIZE`、`SQUARE_SIZE`、`CALIBRATION_*`、`BONE_TOLERANCE` |
| One Euro | `POINT_2D_FILTER`、`POINT_3D_FILTER`、`ANGLE_FILTER` |
| 检测门限 | `MP_*_CONFIDENCE`、`MAX_REPROJECTION_ERROR`、`MAX_DEPTH_MM`、`MAX_HAND_RADIUS` |
| 状态 | `STALE_FRAMES`、`HAND_SWITCH_FRAMES`、`STANDARD_PALM_SIZE` |
| Retarget | `RETARGET_THUMB_*`、`RETARGET_ANGLE_FILTER`、`RETARGET_MAX_EVALUATIONS`、`RETARGET_FTOL` |
| 输出 | `KEYPOINT_TOPIC`、`ROBOT_TOPIC`、`WEB_PORT`、`VIEW_PORTS`、`WEB_FPS` |

One Euro 元组依次为
`(min_cutoff, beta, derivative_cutoff)`。二维点使用 pixel，三维点使用 mm，角度使用
degree，因此不同信号的参数不能直接互换。配置在对象创建时读取，修改后需重启程序。

MMHand 数组和 ROS 输出使用固定的 J00–J20 顺序：

```text
Little  A-A, F-E, PIP, DIP
Ring    A-A, F-E, PIP, DIP
Middle  A-A, F-E, PIP, DIP
Index   A-A, F-E, PIP, DIP
Thumb   MCP A-A, MCP F-E, PIP, DIP, CMC
```

## 坐标系

### 跟踪坐标系

`keypoint_relative` 和 ROS 关键点使用腕点坐标系。原点为 MediaPipe point 0，轴为：

```text
+z = unit(point9 - point0)
+x = unit((point5 - point17) × +z)
+y = +z × +x
```

关键点再按掌尺寸缩放到 `STANDARD_PALM_SIZE`，默认 `0.086 m`。该坐标系用于跟踪、
显示和关键点发布，不受 retarget 坐标定义影响。

### 人手 retarget 坐标系

Retarget 在独立的 CMC 坐标系中临时转换关键点：

```text
origin = point1
+x = unit(point9 - point0)
+y = unit(project_to_plane(point17 - point5, normal=+x))
+z = +x × +y
```

这套定义只依赖人手关键点，不读取 MMHand URDF，也不会写回跟踪结果。

### MMHand 掌坐标系

MMHand 掌轴直接采用 URDF 根 `base_link` 的轴方向：`+x` 大致由腕部指向四指，`+y`
大致由食指侧指向小指侧，`+z` 为掌面法向。原点是 `Thumb_MCP_AA` 原点在
`Thumb_CMC` 转轴上的正交投影。坐标方向、关节轴、父子关系、尺寸和限位均从
`assets/mmhand/urdf/hand.urdf` 读取。

## 处理流程

模块由 runtime 统一装配，数据按以下方向传递：

```text
track (runtime)
├── camera → hand_core → retarget
├── viewer
└── ros
```

`track.py` 只负责选择相机、装配组件、转发跟踪和 retarget 结果，以及按逆序释放资源。

### 三维手部跟踪

```text
同步左右图像
→ 左右 HandLandmarker 21 点
→ 左右独立 2D One Euro
→ 去畸变和双目三角化
→ 重投影、深度和手尺寸检查
→ 3D One Euro
→ 骨长约束
→ 无符号屈曲角提取与 One Euro
→ 21 点重建和标准化
```

普通双目从 `stereo_params.json` 读取 `K1、D1、K2、D2、R、T`；D435 从 SDK 获取
相同参数。`T` 使用毫米，因此三角化的 `keypoint_absolute` 也是毫米。

启动后以 `CALIBRATION_HZ` 限速收集有效样本，用每根骨长的中位数建立手模型。
PIP、DIP 和拇指弯曲采用相邻单位骨向量的无符号夹角：

```text
angle = acos(clamp(u · v, -1, 1))
```

连续坏帧期间短暂保留上一结果供显示，但 stale 帧不进入 retarget 或 ROS。超过
`STALE_FRAMES` 后清空结果，并重置二维、三维和角度滤波器。

### MMHand retarget

四指的 16 个关节直接由人手几何得到，不进入 IK：

```text
MCP A-A = 人手近节侧摆角，经 URDF 零位方向和关节轴符号换算
MCP F-E = 近节相对掌平面的屈曲角
PIP/DIP = 相邻指骨夹角
```

MMHand 的 MCP A-A 零度不等于中位方向。代码用 URDF 零位 FK 求近节方向和关节轴，
将人手侧摆角转换到真实机器人零位，最后裁进 URDF 限位。

一次 SLSQP 仅优化 J16–J20 五个拇指关节，总损失由三个已加权目标组成：

- 掌原点到拇指尖的位置向量。
- 人手 MCP→IP 与 MMHand PIP→DIP 的单位方向。
- 人手 IP→TIP 与 MMHand DIP→TIP 的单位方向。

位置误差按标准掌尺寸归一化；三个权重和拇指尖缩放由 `RETARGET_THUMB_*` 配置。
优化使用解析 Jacobian、上一帧原始解热启动，并受
`RETARGET_MAX_EVALUATIONS` 候选评估预算约束；超限或求解器失败时保留最低损失的
有限候选。最终 21 个输出角再独立经过 One Euro 滤波，原始解仍用于下一帧热启动。

Retargeter 由独立线程持有。主线程只提交最新有效帧，线程忙时未处理的旧输入会被
新输入覆盖，避免排队累积延迟。暂停会同时清除热启动和输出滤波状态。Web 损失使用
实际滤波输出重新计算，因此与发布姿态一致。

## ROS 2 输出

`python track.py --ros` 发布两个 `std_msgs/msg/Float32MultiArray`：

| Topic | data | 单位与坐标 |
|---|---|---|
| `/hand/keypoints` | 63 个值，`x0,y0,z0,...,x20,y20,z20` | m；腕点掌局部坐标；标准掌尺寸 |
| `/raw_ik_target` | 21 个 J00–J20 关节角 | degree；URDF 关节顺序 |

消息的 `layout` 分别使用 `KEYPOINT_LAYOUT`（并附加 Left/Right）和 `ROBOT_LAYOUT`。

```bash
ros2 topic echo /hand/keypoints
ros2 topic echo /raw_ik_target
```

## 文件结构

```text
config.py             所有运行配置
camera.py             普通拼接双目、RealSense D435 和图像拆分
one_euro.py           One Euro 滤波器
hand_core.py          MediaPipe、三角化、手部约束与标准化
retarget.py           URDF 运动学、解析 Jacobian、四指映射和拇指 IK
viewer.py             Web dashboard、Viser 场景和图像显示
ros.py                ROS 2 消息构造与 topic 发布
track.py              参数解析、组件装配和实时循环
calibrate.py          普通拼接双目标定
test_hand_capture.py  核心行为测试
assets/               MediaPipe 模型、MMHand URDF、网格和许可证
test/                 一次性机器人工作空间优化资料，不属于日常测试
```

运行快速核心测试：

```bash
python -m unittest -v test_hand_capture.py
```

测试不依赖 `test_data/` 中的录像，也不会执行 `test/` 下的一次性优化程序。
