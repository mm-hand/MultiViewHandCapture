# MultiView Hand Capture

将设备输入统一为标准 21 点手姿态，再把稳定左手实时映射到 MMHand 的 21 个关节。
当前输入支持普通拼接双目、Intel RealSense D435、MANUS Raw Skeleton
和 D435 彩色单目 WiLoR，并共用
retarget、Web 显示和 ROS 2 输出。

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
`vision/assets/hand_landmarker.task`，运行时无需下载。

抓取仿真是可选依赖：

```bash
python -m pip install -r simulation/requirements.txt
```

WiLoR 仅需额外安装 ONNX Runtime CUDA：

```bash
python -m pip install -r wilor/requirements.txt
```

## 快速开始

所有运行参数集中在 `config.py`。

### 普通拼接双目

普通模式由 `vision.camera.StereoCamera` 读取单个 `2560×720` MJPEG 设备，拆成两个
`1280×720` 图像。首次使用、移动镜头或改变分辨率后需要重新标定：

```python
CAMERA_TYPE = "stereo"
CAMERA_INDEX = 0
```

```bash
python -m vision.calibrate
# 或者：cd vision && python calibrate.py
```

也可以从项目根目录直接运行 `python vision/calibrate.py`；两种方式使用同一个
`config.py` 和 `vision/stereo_params.json`。

标定窗口中按 `C` 保存有效棋盘图像对，按 `Q` 求解。默认棋盘为 `9×6` 内角点、
格长 `23.5 mm`，至少需要 10 对；结果写入不提交的 `vision/stereo_params.json`。
`vision/assets/pattern.png` 可用于打印标定板，但实际格长必须与 `SQUARE_SIZE` 一致。

### RealSense D435

```python
CAMERA_TYPE = "d435"
CAMERA_INDEX = 0
```

D435 模式读取同步的 `infrared 1/2` Y8 流，使用 SDK 提供的双目标定，不启用深度流，
也不需要运行 `vision.calibrate`。

### 启动

```bash
python track.py
python track.py --ros  # 同时发布 ROS 2 topic
python track.py --sim  # 同时打开 SAPIEN 抓取仿真
python track.py --source vision
python track.py --source vision --ros
python track.py --source manus
python track.py --source manus --ros
python track.py --source wilor
python track.py --source wilor --ros
```

打开 `http://localhost:8080`：

- 上方显示当前 Source 的输入预览；加权 retarget 损失及占比浮动在右上角。
- 左下显示 `Normalized hand`，包含人手 retarget 坐标轴和原点。
- 右下显示 MMHand URDF，包含 MMHand 掌坐标轴、原点和五指指腹外法线箭头。

右手可以跟踪、显示和发布关键点，但 MMHand retarget 只处理完成骨长标定的稳定左手。

## 配置

`config.py` 中每个参数都有单位和用途注释。常用分组如下：

| 分组 | 参数 |
|---|---|
| 输入与相机 | CLI `--source`、`CAMERA_TYPE`、`CAMERA_INDEX`、图像尺寸、旋转角、D435 帧率 |
| MANUS | `MANUS_SDK_VERSION`、`MANUS_SDK_BRIDGE_PATH`、`MANUS_STALE_SECONDS` |
| WiLoR | `WILOR_*` 单目分辨率、CUDA device、检测阈值与检测间隔 |
| 标定与约束 | `BOARD_SIZE`、`SQUARE_SIZE`、`CALIBRATION_*`、`BONE_TOLERANCE` |
| One Euro | `POINT_2D_FILTER`、`POINT_3D_FILTER`、`ANGLE_FILTER` |
| 检测门限 | `MP_*_CONFIDENCE`、`MAX_REPROJECTION_ERROR`、`MAX_DEPTH_MM`、`MAX_HAND_RADIUS` |
| 状态 | `STALE_FRAMES`、`HAND_SWITCH_FRAMES`、`STANDARD_PALM_SIZE` |
| Retarget | `RETARGET_THUMB_*`、`RETARGET_ANGLE_FILTER`、`RETARGET_MAX_EVALUATIONS`、`RETARGET_FTOL` |
| 输出 | `KEYPOINT_TOPIC`、`ROBOT_TOPIC`、`WEB_PORT`、`VIEW_PORTS`、`WEB_FPS` |

One Euro 元组依次为
`(min_cutoff, beta, derivative_cutoff)`。二维点使用 pixel，三维点使用 mm，角度使用
degree，因此不同信号的参数不能直接互换。配置在对象创建时读取，修改后需重启程序。

一级输入由 CLI 的 `--source vision/manus/wilor` 决定；`CAMERA_TYPE` 只在 VisionSource
内部接受 `stereo` 或 `d435`。

MMHand 数组和 ROS 输出使用固定的 J00–J20 顺序：

```text
Little  A-A, F-E, PIP, DIP
Ring    A-A, F-E, PIP, DIP
Middle  A-A, F-E, PIP, DIP
Index   A-A, F-E, PIP, DIP
Thumb   MCP A-A, MCP F-E, PIP, DIP, CMC
```

## 标准输入接口

所有输入源只向 runtime 返回 `hand.HandFrame`：

```text
timestamp   输入时间戳，单位 second
points      新鲜的标准 21×3 手姿态；无效或 stale 时为 None
handedness  Left、Right 或 None
ready       输入设备是否已完成自身校准
status      Viewer 显示的简短状态
finger_pad_directions
            Thumb、Index、Middle、Ring、Little 的 5×3 指腹外向单位向量；
            与 points 共用掌局部坐标，目前仅 WiLoR 提供
thumb_tip_orientation_world_xyzw
            MANUS Thumb Tip 相对 MANUS WORLD 的 GLOBAL quaternion [x,y,z,w]；
            Vision、invalid 或 stale frame 为 None
preview     可选输入预览图像
```

标准点顺序为 Wrist，Thumb CMC/MCP/IP/TIP，以及 Index、Middle、Ring、Little 各自的
MCP/PIP/DIP/TIP。Source 只需实现 `read() -> HandFrame | None` 和 `close()`；没有基类、
注册器或设备专用 runtime。

## 坐标系

### 跟踪坐标系

`HandFrame.points` 和 ROS 关键点使用腕点坐标系。原点为 point 0，轴为：

```text
+z = unit(point9 - point0)
+x = unit((point5 - point17) × +z)
+y = +z × +x
```

关键点再按掌尺寸缩放到 `STANDARD_PALM_SIZE`，默认 `0.086 m`。指腹向量只旋转到该
坐标系，不缩放。该坐标系用于跟踪、显示和关键点发布，不受 retarget 坐标定义影响。

### 人手 retarget 坐标系

Retarget 在独立的 CMC 坐标系中临时转换关键点：

```text
origin = point1
+x = unit(point9 - point0)
+y = unit(project_to_plane(point17 - point5, normal=+x))
+z = +x × +y
```

`compute_cmc_frame()` 返回的三列为上述轴在 world/输入坐标中的表示，矩阵语义是
`R_world_from_cmc`。这套定义只依赖人手关键点，不读取 MMHand URDF，也不会写回跟踪结果。

### MMHand 掌坐标系

MMHand 掌轴直接采用 URDF 根 `base_link` 的轴方向：`+x` 大致由腕部指向四指，`+y`
大致由食指侧指向小指侧，`+z` 为掌面法向。原点是 `Thumb_MCP_AA` 原点在
`Thumb_CMC` 转轴上的正交投影。坐标方向、关节轴、父子关系、尺寸和限位均从
`assets/mmhand/urdf/hand.urdf` 读取。

## 处理流程

模块由 runtime 统一装配，设备差异止于 Source：

```text
track (runtime)
├── vision.camera → vision.source ─┐
├── manus.source → manus.adapter ──┤→ HandFrame → retarget
├── D435 color → wilor.source ───┴
└── viewer / ros / simulation ← MMHand joints
```

`track.py` 只负责选择 Source、转发 HandFrame 和 retarget 结果，以及按逆序释放资源。

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

普通双目从 `vision/stereo_params.json` 读取 `K1、D1、K2、D2、R、T`；D435 从 SDK 获取
相同参数。`T` 使用毫米，因此三角化的 `keypoint_absolute` 也是毫米。

启动后以 `CALIBRATION_HZ` 限速收集有效样本，用每根骨长的中位数建立手模型。
PIP、DIP 和拇指弯曲采用相邻单位骨向量的无符号夹角：

```text
angle = acos(clamp(u · v, -1, 1))
```

stale 帧只更新状态和输入预览，不输出标准手姿态，也不进入 retarget 或 ROS。超过
`STALE_FRAMES` 后重置二维、三维和角度滤波器。

### WiLoR 彩色单目

`WilorSource` 只打开 D435 color stream，不读取红外双目或
`vision/stereo_params.json`。FP16 detector 与 WiLoR ONNX 在 CUDA 上运行；
MANO mesh 回归为标准 21 点，指腹面片的面积加权外法线产生五个指腹方向。
拇指方向再绕 IP→TIP 轴向掌心旋转 `45°`。
左手镜像同时作用于点和向量，避免法线翻转。输入预览在完整 RGB 上叠加黄色
MANO mesh；Normalized hand 用箭头显示五个指腹外法线，没有手时结果为 `None`。

MMHand 面板无论使用 Vision、MANUS 或 WiLoR，都从当前机器人 FK 的五个 tip link
计算并显示指腹外法线箭头；五个 tip link 的原点定义为 STL 指腹中心，局部 `-z`
定义为指腹外法线。等待新 retarget 结果时保持当前机器人姿态和箭头。

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

## Manus 与抓取仿真

`ManusSource` 默认直接加载 `manus/assets` 中的官方 MANUS Core SDK 3.1.1
Integrated runtime，不需要另开 sender。项目的薄 C++ wrapper 明确调用
`CoreSdk_InitializeCoordinateSystemWithVUH(..., true)`；`unitScale=1.0` 表示 metre，
并通过 `CoreSdk_GetRawSkeletonNodeInfoArray()` 取得节点语义。SDK 来源、checksum 和
wrapper 重建方法见 `manus/assets/README.md`。旧 ZMQ parser 仅作为可选兼容代码保留，
不是默认 MANUS 输入链。

positions 经 NodeInfo（若 transport 提供）或官方 25 点 fallback 映射为 standard21，之后
和 Vision 共用 `hand.relative_points()`。Thumb Tip quaternion 直接保存为
`HandFrame.thumb_tip_orientation_world_xyzw`；Phase 1 不转 CMC/robot frame，也不进入
retarget residual。

`simulation.GraspSimulation` 在进程内消费 retarget 输出，单位为 radian、顺序为
J00–J20；它不访问 Source 或 HandFrame，也不改变普通运行路径。仿真只加载
`config.URDF_PATH` 指向的最新 MMHand，不包含第二份 URDF、NERO 或额外 mesh。
当前 URDF 的少数惯性张量含零特征值；SAPIEN 加载时只在内存中为非正定特征值增加
`1e-9` 数值下限，不改写 URDF 文件、几何、关节、质量或限位。

安装可选依赖后运行：

```bash
python track.py --sim
```

场景只有地面、MMHand 和一个竖直动态圆柱。启动及按 `R` 时，圆柱半径在
`25–35 mm`、高度在 `80–120 mm` 内独立随机，终端打印实际尺寸；接触数变化时打印
与圆柱接触的指尖数量。

| 按键 | 功能 |
|---|---|
| `Up/Down`、`Left/Right` | 腕部沿世界 X、Y 移动 |
| `U/J` | 腕部沿世界 Z 正/负方向移动 |
| `I/K` | 腕部绕局部 X 轴正/负旋转 |
| `O/L` | 腕部绕局部 Y 轴正/负旋转 |
| `P/M` | 腕部绕局部 Z 轴正/负旋转 |
| `R` | 重置腕部并重新生成随机尺寸圆柱 |
| `Q/Esc` | 关闭仿真和 runtime |

无有效新 retarget 输出时，仿真保持最后一个手势并继续物理。关闭 SAPIEN Viewer
会按正常清理路径释放 retarget worker、相机、Web Viewer 和 ROS。

为避免刚性 URDF 手指接触物体后被轻易撞偏，极简仿真保留以下抓取稳定设置：

- MMHand 碰撞形状使用静/动摩擦 `2.0/1.0`，物体与地面保持 `0.3/0.3`。
- 21 个手指 drive 使用 `stiffness=1000`、`damping=100`、`force_limit=1e10`
  和 force mode，并在每个物理步补偿重力与科氏力。
- retarget 目标与实际下发的 drive target 分开保存，以 `6 rad/s` 追踪；20 Hz
  控制步内每个关节最多变化 `0.30 rad`。没有新输出时继续保持最后目标，不回零。
- PhysX 使用 `3 mm` contact offset、TGS/PCM、`25/4` solver iterations；圆柱的
  线性/角阻尼为 `2/2`，手部 self-collision mask 使用 bit 30。
- 按 `R` 会把手指 qpos/qvel 和 drive target 恢复到 URDF 零位，并重新应用上述
  drive、摩擦和 self-collision 设置；下一份有效人手目标会从零位平滑恢复跟随。

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
vision/                相机、MediaPipe、标定和视觉资产
manus/                 MANUS source、25→21 adapter 和 SDK 资产说明
wilor/                 D435 彩色单目 WiLoR ONNX source 与 LFS 资产
simulation/            极简 SAPIEN MMHand 抓取仿真
hand.py                标准 HandFrame 和 21 点拓扑
config.py              所有运行配置
one_euro.py            One Euro 滤波器
retarget.py            URDF 运动学、解析 Jacobian、四指映射和拇指 IK
viewer.py              Web dashboard、标准手和 MMHand 显示
ros.py                 ROS 2 消息构造与 topic 发布
track.py               Source 选择、组件装配和实时循环
test_hand_capture.py   核心行为测试
test_simulation.py     可选 SAPIEN headless 测试
test_manus_phase1.py   MANUS Phase 1 接口、mapping、quaternion 与 stale 测试
test_wilor.py          WiLoR mapping、指腹方向、镜像与 detector 测试
assets/mmhand/         MMHand URDF、网格和许可证
test/                  一次性机器人工作空间优化资料，不属于日常测试
```

运行快速核心测试：

```bash
python -m unittest -v test_hand_capture.py
python -m unittest -v test_simulation.py
python -m unittest -v test_manus_phase1.py
python -m unittest -v test_wilor.py
```

WiLoR 模型受 `wilor/assets/WILOR_MODEL_LICENSE.txt` 约束，MANO 和 detector
的来源见 `wilor/assets/THIRD_PARTY_NOTICES.md`；两个 ONNX 通过 Git LFS 管理。

未安装 SAPIEN 时仿真测试自动跳过；安装后会加载当前 MMHand URDF，验证关节契约、
随机圆柱范围、重置、关节限位以及上述抓取稳定设置。各组测试使用独立 Python
进程，避免 MediaPipe EGL 与 SAPIEN/PhysX 原生运行时在退出阶段相互影响。测试不依赖
`test_data/` 中的录像，也不会执行 `test/` 下的一次性优化程序。
