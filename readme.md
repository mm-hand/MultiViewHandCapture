# MultiView Hand Capture

## 简介

用 D435 双红外或普通拼接双目相机完成 MediaPipe 21 点手部重建、标准手归一化、
MMHand retarget、Web 可视化和可选 ROS 2 发布。仓库是可直接运行的 Python
脚本，不需要构建为 Python 或 ROS 软件包。

主要功能：

- 同时显示左右图像和 MediaPipe 2D 结果。
- 输出毫米制相机坐标与 0.086 m 标准掌尺寸的 21 个三维关键点。
- 对三维轨迹和非负屈伸角分别执行 One Euro 滤波。
- 将稳定左手 retarget 为 MMHand 的 21 个 URDF 关节角。
- 在同一网页显示归一化人手和 retarget 后的完整 MMHand URDF。
- 可发布标准化关键点或机器人关节角 ROS 2 topic。

默认输入是 D435 的两路 `1280×720`、30 FPS Y8 红外图像。程序不启用深度流，
只从 RealSense SDK 读取出厂内参和左右相机外参，然后执行普通双目三角化。也可在
`config.py` 切回单个 `2560×720` MJPEG 拼接双目设备。MMHand URDF 和 STL 位于
`assets/mmhand/`；所有相机、滤波、IK、Web 和 ROS 参数集中在 `config.py`。

核心代码保持扁平：

```text
config.py       配置、关节顺序和 topic 契约
hand_core.py    普通双目/D435、MediaPipe、三角化和人手滤波
retarget.py     MMHand URDF FK 与极简拇指优化
track.py        Web、ROS 和运行主循环
calibrate.py    普通拼接双目标定
```

## 安装

克隆仓库：

```bash
git clone https://github.com/mm-hand/MultiViewHandCapture.git
cd MultiViewHandCapture
```

项目使用固定的 `mmhand` Conda 环境。已有环境时更新：

```bash
mamba env update -n mmhand -f environment.yml --prune
conda activate mmhand
```

首次安装时创建：

```bash
mamba env create -f environment.yml
conda activate mmhand
```

最后安装 MediaPipe：

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  "protobuf>=4.25.3,<5" absl-py attrs flatbuffers sounddevice sentencepiece
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --no-deps mediapipe==0.10.21
```

这里使用清华 PyPI 镜像，并用 `--no-deps` 避免 MediaPipe 为未使用的 Tasks
功能安装 JAX/JAXlib，或覆盖 Conda 环境中已经验证的 OpenCV。
`environment.yml` 也通过清华镜像安装 D435 所需的 `pyrealsense2`。

## Quick Start

在 `config.py` 选择相机；编号也只在这里修改：

```python
CAMERA_TYPE = "d435"       # "d435" 或 "stereo"
CAMERA_INDEX = 0           # 第几个 D435，或普通相机的 OpenCV 编号
```

D435 使用出厂标定，无需标定板，直接运行：

```bash
python track.py --mode points              # 跟踪并显示标准化 21 点
python track.py --mode points --ros        # 同时发布 21 点
python track.py --mode retarget            # 左手 retarget，并显示 MMHand
python track.py --mode retarget --ros      # 同时发布 MMHand 关节角
```

使用普通拼接双目时，先设置 `CAMERA_TYPE = "stereo"`。首次使用、更换相机或改变
两个镜头的相对位置后执行：

```bash
python calibrate.py
```

标定窗口中将 `9×6` 内角点、格长 `23.5 mm` 的棋盘放在不同位置和角度：

- `C`：保存当前有效双目图像对。
- `Q`：结束采样并计算参数。
- 至少采集 10 对；结果写入 `stereo_params.json`。

浏览器打开 `http://localhost:8080`：

- 上方：左右红外/彩色相机和 MediaPipe 叠加。
- 左下：腕点坐标系中的 0.086 m 标准人手。
- 右下：retarget 后的 MMHand URDF。

程序开始后先用 100 个有效帧估计骨长，状态从 `CALIBRATION` 变为
`GESTURE TRACKING` 后才发布机器人角度。基础检查命令：

```bash
python -m unittest -v test_hand_capture.py
```

## 原理

两种相机共用同一条人手重建流程：

```text
左右图像上的 MediaPipe 2D
→ D435 出厂参数或 stereo_params.json
→ 双目三角化
→ 3D One Euro
→ 骨长约束
→ 非负屈伸角
→ 角度 One Euro
→ 21 个 3D 点
```

D435 只打开同步左右红外流，不使用 SDK 生成的深度图。D400 左右红外图默认已经
校正，因此程序直接使用活动流内参、单位旋转和 SDK 给出的米制基线；基线转换成
毫米后复用普通双目的投影矩阵、三角化和 reprojection 检查。

`keypoint_absolute` 是毫米制相机坐标。`keypoint_relative` 以腕点为原点，
使用掌面法向、由小指侧指向食指侧、手指前伸方向组成的掌坐标系，并把平均掌尺寸
归一到 `0.086 m`。

Retarget 只处理稳定、非 stale、完成骨长标定的左手：

```text
标准化 21 点
→ 四指角度直传
→ 拇指五自由度 bounded least squares
→ MMHand 21 个关节角
```

四指 MCP F-E、PIP、DIP 直接使用人手屈伸角。MCP A-A 在人手侧摆角上加
MMHand 的机械中位：小指、无名指、中指、食指分别为
`36°、29°、31°、23°`，不增加额外散开偏置。

拇指优化仅保留三类目标：

1. 人手 Wrist→Thumb TIP 经一个尺度匹配机器人
   Palm-origin→Thumb TIP。
2. 人手 Thumb TIP→其余四指尖的四条向量直接匹配机器人对应向量，不再缩放。
3. MMHand 指腹法向指向距离人手拇指最近的两个指尖所对应的机器人指尖中点。

机器人掌坐标轴与人手掌坐标轴语义相同，掌原点是相对 URDF `palm_1` 的可调三维
偏移。位置残差统一除以 `0.086 m`，使米制位置误差与无量纲方向误差处于相近
数量级，同时避免对指向量接近零时按自身长度归一化导致发散。上一帧关节角只作为
求解初值，不进入损失函数。

## 输出 ROS 话题

加 `--ros` 后，`track.py` 在同一进程内创建 ROS 2 节点
`multiview_hand_capture`，消息类型均为 `std_msgs/msg/Float32MultiArray`。

### `/hand/keypoints`

由 `python track.py --mode points --ros` 发布。

| 字段 | 内容 |
|---|---|
| data | 63 个 float：`x0,y0,z0,...,x20,y20,z20` |
| shape | `21×3` |
| 单位 | m |
| 坐标系 | 腕点原点、掌坐标轴、掌尺寸 0.086 m |
| layout label | `mvhc:keypoints:v1:palm_local_m:size=0.086:hand=Left/Right` |

### `/raw_ik_target`

由 `python track.py --mode retarget --ros` 发布。

| 字段 | 内容 |
|---|---|
| data | 21 个 float，J00...J20 |
| shape | `21` |
| 单位 | degree，URDF 关节角 |
| 顺序 | 小指、无名指、中指、食指、拇指 |
| layout label | `mmhand:J00-J20:urdf_deg` |

可用以下命令检查：

```bash
ros2 topic echo /hand/keypoints
ros2 topic echo /raw_ik_target
```

手丢失、结果 stale、骨长标定未完成、检测到右手或 IK 失败时，不发布
`/raw_ik_target`。
