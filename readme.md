# MultiView Hand Capture

双目 MediaPipe 21 点重建、MMHand Vector retarget、Viser Web 可视化和可选 ROS 2
发布。它是直接运行的脚本，不是 Python/ROS 软件包。

## 环境

环境名固定为 `mmhand`，同时包含 Python 3.12、RoboStack Jazzy 和运行依赖：

```bash
mamba env update -n mmhand -f environment.yml --prune
conda activate mmhand
```

若还没有这个环境，改用：

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

这里使用清华 PyPI 镜像，并特意带 `--no-deps`：本项目只使用 MediaPipe Hands，
不让 pip 为未使用的 Tasks 功能额外安装 JAX/JAXlib，也不让 pip 覆盖环境中已经
验证可用的 conda OpenCV。

## 运行

```bash
python calibrate.py                         # 已有 stereo_params.json 时跳过
python track.py --mode points              # 只跟踪、显示
python track.py --mode points --ros        # 发布标准化 21 点
python track.py --mode retarget            # 左手 → MMHand，并显示 URDF
python track.py --mode retarget --ros      # 同时发布机器人角度
```

浏览器打开 `http://localhost:8080`。上方整行显示左右相机和 MediaPipe
叠加；左下以正对掌面的固定初始视角显示腕点坐标系、0.086 m 标准掌尺寸的
归一化手势，右下显示 retarget 后的完整 MMHand URDF。内部视口使用
8081、8082 端口。

当前相机是单个 2560×720 MJPEG 设备，左右各 1280×720。`Camera.read()` 的
输出已统一为 `(ok, left, right, timestamp)`；以后接 D435 只需实现相同接口，
当前不包含未验证的 D435 代码。相机、标定、滤波、IK、URDF 语义和发布参数均在
`config.py`。

## 数据流

人手重建：

```text
MediaPipe 2D → 双目三角化 → 3D One Euro → 骨长约束
→ 非负屈伸角约束 → 角度 One Euro → 21 个 3D 点
```

`keypoint_absolute` 是毫米制相机坐标；`keypoint_relative` 是以腕点为原点、
掌宽方向固定且掌尺寸归一到 0.086 m 的 21×3 米制坐标。

Retarget 只接受稳定、非 stale、已经完成骨长标定的 `Left`：

```text
21 点 → 腕点坐标系与 MMHand 坐标变换 → 20 个对应人手/机器人向量
→ 每向量中立姿态旋转与长度缩放 → URDF 解析 Jacobian + SLSQP
→ 安全关节限位与指尖间距 → 21 个关节角
```

20 个向量包括腕部到五个指尖、四指各三段指骨和拇指三段指骨。每项使用独立的
固定旋转、scaling 和 weight；旋转把人手中立
向量直接对齐到 MMHand 的机械中位向量，因此 A-A 不需要额外补偿损失。机械
中位为小指、无名指、中指、食指、拇指 A-A 分别 36°、29°、31°、23°、28°，
安全姿态再令四指 A-A 相对中位分别为小指 −8°、无名指 −1.5°、中指 +1.5°、
食指 +6°，并作为 IK 初值和持续边界。

向量统一采用 SmoothL1。人手拇指 MCP/IP 弯曲角直接目标到机器人 Thumb
PIP/DIP；拇指到四指改用带碰撞球半径和 1.5 mm 净空的标量距离，不再优化四个
互相冲突的方向向量。接近对指时，拇指指腹法向朝向人手最近两指所对应的机器人
指尖中点，以消除 CMC 自由度。相邻四指另有单侧安全距离项，最终输出还会拒绝
任何末端球穿入。上一帧角度既作 warm start，也用于小权重时序正则；输入已经
过 One Euro，因此不再重复低通机器人角度。没有通用碰撞引擎、DexPilot、分段、
阻尼最小二乘、Torch 或额外进程。所有映射和参数均在 `config.py`。

## ROS 2 契约

`--mode points --ros` 发布：

- topic：`/hand/keypoints`
- type：`std_msgs/Float32MultiArray`
- data：63 个 float，顺序为 `x0,y0,z0,...`，单位 m
- layout：含 `hand=Left` 或 `hand=Right`

`--mode retarget --ros` 发布：

- topic：`/raw_ik_target`
- type：`std_msgs/Float32MultiArray`
- data：21 个 float，J00...J20 的 URDF 角度，单位 degree
- layout：`mmhand:J00-J20:urdf_deg`

手丢失、stale、右手或 IK 失败时不发布机器人角度。

## 文件

```text
config.py       所有配置和语义映射
hand_core.py    相机、双目重建、人手滤波
retarget.py     Vector 目标、URDF 解析 Jacobian 和 SLSQP
track.py        Web、ROS 和主循环
calibrate.py    双目标定
```

MMHand URDF 和 STL 来自父仓库 `structure` 的 `1326daf`，位于
`assets/mmhand/`。URDF 是关节名、轴向和限位的唯一来源；ROS 输出直接按
小指、无名指、中指、食指、拇指排列为 J00...J20。本地录像位于被 Git 忽略的
`test_data/`。
测试命令：

```bash
python -m unittest -v test_hand_capture.py
```
