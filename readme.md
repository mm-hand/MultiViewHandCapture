# MultiView Hand Capture

双目 MediaPipe 21 点重建、MMHand 数值 retarget、Viser Web 可视化和可选 ROS 2
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
叠加；下方三个独立、可旋转的 3D 视口分别显示相机坐标系原始点、掌心局部
坐标系的 0.086 m 标准人手，以及 retarget 后的完整 MMHand URDF。三个视口
使用独立相机和尺度，不会因放在同一场景而被整体缩小。内部视口使用
`config.py` 中的 8081–8083 端口。

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
标准人手 21 点 → 全手 21 变量有限差分 Jacobian
→ Moore-Penrose pseudoinverse → 关节限位 → 关节 One Euro
```

单个全局目标只包含四类向量：局部骨段、低权重的指根到指尖、低权重的手掌到
指尖、拇指尖到其余四个指尖。没有分段、阻尼最小二乘、参考姿态、接触状态、
碰撞项或 PIP/DIP 耦合。上一帧原始角度只作下一帧初值。

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
- layout：`mmhand:J00-J20:urdf_deg:structure_urdf_v2:v3:73FD45FA`

手丢失、stale、右手或 IK 失败时不发布机器人角度。

## 文件

```text
config.py       所有配置和语义映射
hand_core.py    相机、双目重建、人手滤波
retarget.py     URDF FK 和全局数值 IK
track.py        Web、ROS 和主循环
calibrate.py    双目标定
```

MMHand URDF、contract 和 STL 位于 `assets/mmhand/`。本地录像位于被 Git 忽略
的 `test_data/`。测试命令：

```bash
python -m unittest -v test_hand_capture.py
```
