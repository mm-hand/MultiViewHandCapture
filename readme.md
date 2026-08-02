# MultiView Hand Capture

## 简介

MultiView Hand Capture 从同步双目图像恢复 MediaPipe 21 个手部三维关键点，将人手
归一到固定掌尺寸，并可进一步 retarget 到 MMHand 的 21 个关节。整个项目是可以
直接运行的 Python 脚本，不需要构建 Python package 或 ROS package。

主要功能：

- 支持 Intel RealSense D435 双红外图像，直接读取出厂双目标定。
- 支持单个 `2560×720` MJPEG 设备提供的左右拼接双目图像。
- 对左右图像分别运行 MediaPipe，并通过普通双目三角化恢复三维点。
- 对三维点和手指屈伸角分别执行 One Euro 滤波。
- 通过骨长和非负屈伸角约束抑制关键点飞行与手指反弯。
- 在 MMHand 指尖显示由预设局部法向计算的指腹方向箭头。
- 将标准化左手 retarget 到 MMHand URDF。
- 在一个网页中显示左右图像、标准人手和 MMHand。
- 可选发布 ROS 2 关键点或机器人关节角。

D435 模式只使用同步左右红外图像，不启用深度流，也不读取深度图。左右相机内参
和基线由 RealSense SDK 提供，后续三角化、滤波和 retarget 与普通双目完全共用。

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

运行参数集中在 `config.py`，通常只需要修改相机类型和编号。

### 相机

```python
CAMERA_TYPE = "d435"       # "d435" 或 "stereo"
CAMERA_INDEX = 0           # 第几个 D435，或普通相机的 OpenCV 编号
D435_WIDTH, D435_HEIGHT, D435_FPS = 1280, 720, 30
```

`CAMERA_TYPE="d435"`：

- 打开 D435 的 `infrared 1` 和 `infrared 2`。
- 图像格式为 Y8，默认 `1280×720@30`。
- 使用活动流内参和出厂左右外参。
- 不需要运行棋盘标定。

`CAMERA_TYPE="stereo"`：

- 打开 `CAMERA_INDEX` 指定的 OpenCV/V4L2 设备。
- 输入为单个 `2560×720` MJPEG 图像。
- 左右各占 `1280×720`，并按 `ROTATE_LEFT/ROTATE_RIGHT` 旋转。
- 标定参数来自 `stereo_params.json`。

### 三维重建和滤波

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `MIN_CALIBRATION_PAIRS` | 10 | 普通双目标定要求的最少有效图像对数 |
| `CALIBRATION_FRAMES` | 100 | 初始骨长估计使用的有效样本数 |
| `CALIBRATION_HZ` | 10 Hz | 骨长样本的最大采样频率 |
| `BONE_TOLERANCE` | 0.20 | 每根骨允许偏离标定长度的比例 |
| `POINT_FILTER` | `(1.0, 0.01, 1.0)` | 三维点 One Euro 参数 |
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

`POINT_FILTER` 和 `ANGLE_FILTER` 的三个值依次是 One Euro 的
`min_cutoff、beta、derivative_cutoff`。

### MMHand

`ROBOT_JOINT_NAMES` 定义内部数组和 ROS 输出的 J00–J20 顺序：

```text
Little  A-A, F-E, PIP, DIP
Ring    A-A, F-E, PIP, DIP
Middle  A-A, F-E, PIP, DIP
Index   A-A, F-E, PIP, DIP
Thumb   MCP A-A, MCP F-E, PIP, DIP, CMC
```

机械中位角：

```python
MCP_AA_NEUTRAL_DEG = (36, 29, 31, 23)  # Little, Ring, Middle, Index
THUMB_MCP_AA_NEUTRAL_DEG = 28
```

拇指数值优化参数：

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

### D435

确认 `config.py` 中：

```python
CAMERA_TYPE = "d435"
CAMERA_INDEX = 0
```

然后直接运行：

```bash
python track.py --mode retarget
```

### 普通拼接双目

设置：

```python
CAMERA_TYPE = "stereo"
CAMERA_INDEX = 0
```

首次使用、更换相机或改变两个镜头的相对位置后执行标定：

```bash
python calibrate.py
```

棋盘默认是 `9×6` 内角点、格长 `23.5 mm`：

- `C`：保存当前有效左右图像对。
- `Q`：结束采样并计算参数。
- 至少采集 `MIN_CALIBRATION_PAIRS` 对图像，默认 10 对。
- 结果写入 `stereo_params.json`。

### 运行模式

```bash
python track.py --mode points
python track.py --mode points --ros
python track.py --mode retarget
python track.py --mode retarget --ros
```

- `points`：输出标准化的 21 个三维关键点。
- `retarget`：将完成初始骨长估计的稳定左手映射到 MMHand。
- `--ros`：在显示结果的同时发布 ROS 2 topic。

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
    D435/普通相机接口、MediaPipe、双目三角化、滤波和标准手归一化

retarget.py
    MMHand URDF 解析、前向运动学、四指映射和拇指数值优化

track.py
    程序入口、实时循环、Web 页面、Viser 和 ROS 发布

calibrate.py
    普通拼接双目的棋盘标定工具

test_hand_capture.py
    几何、滤波、retarget 和本地录像回归测试

assets/mmhand/
    MMHand URDF、STL 和第三方许可证

stereo_params.json
    普通拼接双目的内外参数
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

### 1. 相机和标定参数

`track.py` 根据 `CAMERA_TYPE` 创建相机：

```text
d435
→ RealSenseCamera
→ 左右 Y8 图像 + SDK 出厂内外参数

stereo
→ Camera
→ 拆分 MJPEG 拼接图 + stereo_params.json
```

两种输入都生成相同的 `left、right、timestamp`，并将相同格式的
`K1、D1、K2、D2、R、T` 交给 `StereoProcessor`。

### 2. MediaPipe 和双目三角化

```text
左右图像
→ 左右 MediaPipe Tasks HandLandmarker 21 点
→ 像素坐标去畸变
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
→ 3D One Euro
→ 骨长约束
→ 提取拇指 2 个、四指 12 个屈伸角
→ 将屈伸角限制为非负
→ 角度 One Euro
→ 从骨长和角度恢复 21 点
```

初始骨长样本按 `CALIBRATION_HZ` 限速采集，避免不同相机帧率或处理速度改变标定
时长。100 个有效样本用于计算每根骨长度的中位数。之后每根骨只能在标定长度的
`1±BONE_TOLERANCE` 范围内变化。非负屈伸角允许手指弯曲，但不允许反弯。

`relative_points()` 再将三维点：

- 平移到腕点原点。
- 建立掌局部坐标系。
- 缩放到 `STANDARD_PALM_SIZE=0.086 m`。

### 4. MMHand retarget

Retarget 只处理完成骨长估计、非 stale 的稳定左手。

四指不执行数值优化：

```text
人手 MCP A-A + 机械中位 → MMHand MCP A-A
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

机器人 FK、关节轴和上下限均从 `assets/mmhand/urdf/hand.urdf` 读取。优化使用
上一帧拇指关节角作为初值，但上一帧本身不进入目标函数。

### 5. 状态、显示和输出

`StereoProcessor` 对短暂坏帧返回 stale 结果用于显示，但 stale 数据不会进入
retarget 或 ROS。手丢失、检测到右手、初始骨长估计未完成或求解失败时，不发布
新的机器人目标。

`Viewer` 将左右图像编码为 JPEG，并用两个 Viser 服务显示标准人手和 MMHand。
MMHand 的五个指尖显示 `0.025 m` 长的橙色指腹方向箭头，箭头由各 fingertip
link 的预设局部法向经 FK 转换得到。两个三维窗口的初始
相机均从掌心侧正视手掌；在掌基座校平视角的基础上，相机视角统一顺时针旋转 15°。
这些视角设置不会改变计算坐标、retarget 或 ROS 输出。
`RosOutput` 只在传入 `--ros` 时创建。

## ROS 2 输出

消息类型均为 `std_msgs/msg/Float32MultiArray`。

### `/hand/keypoints`

由 `python track.py --mode points --ros` 发布：

| 字段 | 内容 |
|---|---|
| data | 63 个 float：`x0,y0,z0,...,x20,y20,z20` |
| shape | `21×3` |
| 单位 | m |
| 坐标系 | 腕点原点、掌局部坐标、掌尺寸 0.086 m |
| layout | `mvhc:keypoints:v1:palm_local_m:size=0.086:hand=Left/Right` |

### `/raw_ik_target`

由 `python track.py --mode retarget --ros` 发布：

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
