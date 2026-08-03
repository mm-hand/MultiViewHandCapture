# MultiView Hand Capture

## 简介

MultiView Hand Capture 从同步双目图像恢复 MediaPipe 21 个手部三维关键点，将人手
归一到固定掌尺寸，并可进一步 retarget 到 MMHand 的 21 个关节。整个项目是可以
直接运行的 Python 脚本，不需要构建 Python package 或 ROS package。

主要功能：

- 支持 Intel RealSense D435 双红外图像，直接读取出厂双目标定。
- 对 D435 左右红外图像分别运行 MediaPipe，并通过双目三角化恢复三维点。
- 对三维点和手指屈伸角分别执行 One Euro 滤波。
- 通过骨长和 `0～180°` 无符号屈伸角约束抑制关键点飞行。
- 在 MMHand 指尖显示由预设局部法向计算的指腹方向箭头。
- 将标准化左手 retarget 到 MMHand URDF。
- 在一个网页中显示左右图像、标准人手和 MMHand。
- 可选同时发布 ROS 2 关键点和机器人关节角。

项目默认且仅支持 D435。程序只使用同步左右红外图像，不启用深度流，也不读取
深度图；活动流内参、畸变参数和左右外参全部由 RealSense SDK 提供，无需棋盘标定。

## 安装

克隆仓库：

```bash
git clone https://github.com/mm-hand/MultiViewHandCapture.git
cd MultiViewHandCapture
```

首次创建 `mmhand` Conda 环境：

```bash
mamba env create -f environment.yml
conda activate mmhand
```

已有环境时更新：

```bash
mamba env update -n mmhand -f environment.yml --prune
conda activate mmhand
```

`environment.yml` 会通过清华 PyPI 镜像安装 D435 所需的 `pyrealsense2` 和不锁定
版本的最新 MediaPipe。MediaPipe 使用 `--no-deps`，避免覆盖 Conda 中的 OpenCV；
它需要的运行依赖已直接列在环境文件中。已有环境也可以单独升级 MediaPipe：

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --upgrade --no-deps mediapipe
```

代码使用 MediaPipe Tasks `HandLandmarker`，官方 full float16 模型已包含在
`assets/mediapipe/hand_landmarker.task`，运行时不需要下载模型。

## Config

运行参数集中在 `config.py`，通常只需要修改 D435 编号、分辨率和帧率。

### 相机

```python
CAMERA_INDEX = 0
D435_WIDTH = 1280
D435_HEIGHT = 720
D435_FPS = 30
```

- 打开 D435 的 `infrared 1` 和 `infrared 2`。
- 图像格式为 Y8，默认 `1280×720@30`。
- 使用活动流内参和出厂左右外参。
- 不需要也不提供外部棋盘标定流程。

### 三维重建和滤波

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `CALIBRATION_FRAMES` | 100 | 初始骨长估计使用的有效样本数 |
| `CALIBRATION_HZ` | 10 Hz | 骨长样本的最大采样频率 |
| `BONE_TOLERANCE` | 0.20 | 每根骨允许偏离标定长度的比例 |
| `POINT_2D_FILTER` | `(1.0, 0.01, 1.0)` | 去畸变前像素点 One Euro 参数 |
| `POINT_3D_FILTER` | `(0.5, 0.001, 1.0)` | 三角化后三维毫米点 One Euro 参数 |
| `ANGLE_FILTER` | `(1.0, 0.02, 1.0)` | 屈伸角 One Euro 参数 |
| `MP_DETECTION_CONFIDENCE` | 0.5 | 掌检测最低置信度 |
| `MP_PRESENCE_CONFIDENCE` | 0.6 | 手存在最低置信度，低于它会重新检测手掌 |
| `MP_TRACKING_CONFIDENCE` | 0.6 | 跟踪框最低 IoU，低于它会重新检测手掌 |
| `MAX_REPROJECTION_ERROR` | 30 px | 最大平均重投影误差 |
| `MAX_DEPTH_MM` | 1500 mm | 三角化点允许的相机 Z 上限；不设置下限 |
| `MAX_HAND_RADIUS` | 300 mm | 关键点距腕部的最大距离 |
| `STALE_FRAMES` | 3 | 坏帧时短暂保留上一结果的帧数 |
| `HAND_SWITCH_FRAMES` | 5 | 手性切换所需的连续确认帧数 |
| `STANDARD_PALM_SIZE` | 0.086 m | 标准人手掌尺寸 |

三个滤波元组的元素都依次是 One Euro 的
`min_cutoff、beta、derivative_cutoff`。二维点单位为 pixel、三维点单位为 mm、
角度单位为 degree，因此三者的 `beta` 数值不能直接混用。所有参数在对象初始化时
读取，修改 `config.py` 后需要重启程序。

### MMHand

`ROBOT_JOINT_NAMES` 定义内部数组和 ROS 输出的 J00–J20 顺序：

```text
Little  A-A, F-E, PIP, DIP
Ring    A-A, F-E, PIP, DIP
Middle  A-A, F-E, PIP, DIP
Index   A-A, F-E, PIP, DIP
Thumb   MCP A-A, MCP F-E, PIP, DIP, CMC
```

四指 A-A retarget 零位偏置：

```python
MCP_AA_NEUTRAL_DEG = (19.9474, 29.0, 26.1463, 23.0)  # Little, Ring, Middle, Index
```

该偏置把人手零侧摆映射到优化版 MMHand 四指姿态，并保证位于新 URDF 限位内；
它不属于 URDF 机械参数。拇指没有中位值；首次求解和暂停后的初值是将 URDF
零位裁进其关节上下限。

其余拇指数值优化参数属于 retarget 几何或求解配置：

| 参数 | 含义 |
|---|---|
| `MMHAND_PALM_ORIGIN` | 机器人掌坐标中对应人手腕原点的三维偏移 |
| `THUMB_TIP_SCALE` | 人手腕到拇指尖向量的缩放 |
| `THUMB_TIP_WEIGHT` | 拇指尖位置目标权重 |
| `THUMB_PAD_AXIS` | 拇指指腹 link 局部坐标中的指腹法向 |
| `FINGER_PAD_AXES` | Index、Middle、Ring、Little 指尖 link 的预设指腹法向 |
| `THUMB_PAD_WEIGHT` | 指腹朝向目标权重 |
| `THUMB_MAX_EVAL` | 每帧最多残差计算次数 |
| `THUMB_FTOL` | least-squares 停止容差 |

### Web 和 ROS

```python
WEB_PORT = 8080
VIEW_PORTS = (8081, 8082)
WEB_FPS = 15

KEYPOINT_TOPIC = "/hand/keypoints"
ROBOT_TOPIC = "/raw_ik_target"
```

## 运行

项目默认且仅使用 D435。确认 `config.py` 中：

```python
CAMERA_INDEX = 0
```

然后直接运行：

```bash
python track.py
```

### 启动参数

```bash
python track.py
python track.py --ros
```

程序始终估计和显示标准化的 21 个三维关键点，并将完成初始骨长估计的稳定左手
映射到 MMHand。`--ros` 在相同流程上同时发布关键点和机器人关节两个 ROS 2 topic。
右手关键点仍会显示和发布，但不会映射到左手机器人模型。

浏览器打开 `http://localhost:8080`：

- 上方：左右相机图像和 MediaPipe 结果。
- 左下：腕部坐标系中的 `0.086 m` 标准人手。
- 右下：retarget 后的 MMHand URDF。

程序启动后以最高 10 Hz 收集 100 个有效样本估计骨长，连续识别时约需 10 秒。
页面状态从 `CALIBRATION` 变为 `GESTURE TRACKING` 后，retarget 和机器人 ROS
输出才会开始。

## 代码文件结构

```text
config.py
    相机、滤波、质量门限、MMHand、Web 和 ROS 配置

hand_core.py
    D435 双红外接口、MediaPipe、双目三角化、滤波和标准手归一化

retarget.py
    MMHand URDF 解析、前向运动学、四指映射和拇指数值优化

track.py
    程序入口、实时循环、Web 页面、Viser 和 ROS 发布

test_hand_capture.py
    几何、滤波、MediaPipe、URDF、retarget、CLI 和 ROS 快速核心测试

test/
    一次性机器人工作空间优化资料，不属于日常测试套件

assets/mmhand/
    MMHand URDF、STL 和第三方许可证

```

核心代码保持单向依赖：

```text
config.py
    ↑
hand_core.py
    ↑
retarget.py
    ↑
track.py
```

## 代码逻辑

### 1. D435 相机参数

`track.py` 直接创建 D435 相机和处理器：

```text
RealSenseCamera
→ 左右 Y8 图像 + SDK 出厂内外参数
→ StereoProcessor
```

`RealSenseCamera` 输出同步的 `left、right、timestamp`，并将 SDK 提供的
`K1、D1、K2、D2、R、T` 交给 `StereoProcessor`。

### 2. MediaPipe 和双目三角化

```text
左右图像
→ 左右 MediaPipe Tasks HandLandmarker 21 点
→ 归一化坐标转换为像素坐标
→ 左右独立的 2D One Euro
→ 滤波后像素坐标去畸变
→ cv2.triangulatePoints
→ 21×3 毫米制相机坐标
→ reprojection、Z 上限和手尺寸检查
```

两路 HandLandmarker 使用同步 VIDEO 模式和同一帧时间戳，配置为 detection 0.5、
presence 0.6、tracking 0.6。任何三角化点大于 1500 mm 时整帧标记为 `depth`；没有
近距离下限。非有限值、平均重投影误差和手尺寸检查仍然在所有滤波之前执行。

MediaPipe Tasks 的手性标签直接用于当前未镜像输入；手性需要连续多帧确认后才会
切换。

三角化使用：

```text
P1 = K1 [I | 0]
P2 = K2 [R | T]
```

`T` 使用毫米，所以 `keypoint_absolute` 也是毫米。

### 3. 人手约束和滤波

```text
三角化 21 点
→ 3D One Euro（独立于二维滤波参数）
→ 骨长约束
→ 提取拇指 2 个、四指 12 个屈伸角
→ 两个骨向量的点积夹角（0～180°）
→ 角度 One Euro
→ 从骨长和角度恢复 21 点
```

初始骨长样本按 `CALIBRATION_HZ` 限速采集，避免相机帧率或处理速度改变估计
时长。100 个有效样本用于计算每根骨长度的中位数。之后每根骨只能在估计长度的
`1±BONE_TOLERANCE` 范围内变化。

设当前关节前后的单位骨向量为 `u、v`，屈曲角为：

```text
angle = acos(clamp(u · v, -1, 1))
```

PIP、DIP 使用相邻两段指骨；拇指使用相邻骨段。四指 MCP 使用近节骨实际方向和
它在掌平面上的投影，从而将屈曲与侧摆分开。该定义只表达弯曲幅度，伸直为 0°、
直角为 90°、完全折回为 180°，不产生负角或 `+180°/-180°` 跨界。握拳遮挡可能
让错误关键点形成接近 180° 的假角，这属于检测/三角化质量问题。

`relative_points()` 再将三维点：

- 平移到腕点原点。
- 建立掌局部坐标系。
- 缩放到 `STANDARD_PALM_SIZE=0.086 m`。

### 4. MMHand retarget

Retarget 只处理完成骨长估计、非 stale 的稳定左手。

四指不执行数值优化：

```text
人手 MCP A-A + retarget 零位偏置 → MMHand MCP A-A
人手 MCP F-E            → MMHand MCP F-E
人手 PIP                → MMHand PIP
人手 DIP                → MMHand DIP
```

拇指优化 5 个关节，残差包括：

1. 机器人拇指尖位置匹配缩放后的人手拇指尖位置。
2. 机器人拇指尖到四指尖的四条向量匹配人手对应向量。
3. 拇指指腹法向朝向四指指尖的加权位置。

第三项在每次残差计算时使用当前 MMHand 姿态。设拇指尖为 `t`、四指指尖为
`f_i`，取反距离平方权重：

```text
d_i = |f_i - t|
w_i = (1 / max(d_i, EPS)^2) / sum_j(1 / max(d_j, EPS)^2)
c   = sum_i(w_i * f_i)
direction = unit(c - t)
```

因此四指全部参与，距离拇指尖越近的手指对目标方向影响越大。

机器人 FK、关节轴和上下限均从 `assets/mmhand/urdf/hand.urdf` 读取。该文件是
工作空间优化后的正式 MMHand URDF，代码不会收紧或覆盖其限位。连续追踪时优化
使用上一帧拇指关节角作为初值；首次求解和暂停后使用裁进 URDF 上下限的零位。
初值本身不进入目标函数。

### 5. 状态、显示和输出

`StereoProcessor` 对短暂坏帧返回 stale 结果用于显示，但 stale 数据不会进入
retarget 或 ROS。有效左右手都会发布关键点；手丢失、检测到右手、初始骨长估计
未完成或求解失败时，不发布新的机器人目标。

二维滤波只在左右 HandLandmarker 同一帧都检测成功时同步更新，页面覆盖点和双目
三角化使用同一组滤波后像素坐标。前三个连续坏帧保留上一结果和全部滤波状态；
第四个连续坏帧会同时重置左右二维、三维和角度滤波器。

`Viewer` 将左右图像编码为 JPEG，并用两个 Viser 服务显示标准人手和 MMHand。
MMHand 的五个指尖显示 `0.025 m` 长的橙色指腹方向箭头，箭头由各 fingertip
link 的预设局部法向经 FK 转换得到。两个三维窗口的初始
相机均从掌心侧正视手掌；在掌基座校平视角的基础上，相机视角统一顺时针旋转 15°。
这些视角设置不会改变计算坐标、retarget 或 ROS 输出。
`RosOutput` 只在传入 `--ros` 时创建。

## ROS 2 输出

消息类型均为 `std_msgs/msg/Float32MultiArray`。

### `/hand/keypoints`

由 `python track.py --ros` 发布：

| 字段 | 内容 |
|---|---|
| data | 63 个 float：`x0,y0,z0,...,x20,y20,z20` |
| shape | `21×3` |
| 单位 | m |
| 坐标系 | 腕点原点、掌局部坐标、掌尺寸 0.086 m |
| layout | `mvhc:keypoints:v1:palm_local_m:size=0.086:hand=Left/Right` |

### `/raw_ik_target`

同样由 `python track.py --ros` 发布：

| 字段 | 内容 |
|---|---|
| data | 21 个 J00–J20 关节角 |
| shape | `21` |
| 单位 | degree |
| 顺序 | Little、Ring、Middle、Index、Thumb |
| layout | `mmhand:J00-J20:urdf_deg` |

检查 topic：

```bash
ros2 topic echo /hand/keypoints
ros2 topic echo /raw_ik_target
```

## 测试

```bash
python -m unittest -v test_hand_capture.py
```

这是不依赖本地录像的快速核心测试。`test/` 保存一次性工作空间优化代码和数据，
日常验证不运行 `unittest discover -s test`，也不会重新执行优化。
