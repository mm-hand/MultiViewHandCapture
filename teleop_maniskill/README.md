# NERO + MMHand 纯 SAPIEN 在线遥操

这个目录提供一个双进程 pick-and-place 可视化程序：

- `track.py --mode retarget` 从真人左手估计 MMHand 的 J00–J20，通过 localhost UDP 发送目标关节角。
- `sim_pick_place.py` 使用键盘控制 NERO 末端 XYZ，并在高精度 NERO + MMHand URDF 上实时跟随人手。
- `run_sim_teleop.py` 一键启动上述两个进程，关闭 Viewer 或按 `Ctrl-C` 时同时清理它们。

虽然目录名仍是 `teleop_maniskill`，实际运行时只使用 SAPIEN 3，不需要
ManiSkill、UltraDexGrasp、ROS 或 Torch。

## 1. 自包含边界

完整运行边界是 `MultiViewHandCapture/` 项目根目录，不是只复制
`teleop_maniskill/` 这一个子目录。一键启动还会用到根目录的
`track.py`、`config.py`、`hand_core.py`、`retarget.py` 和 `assets/mmhand/`。

项目内已包含：

- `assets/robot/nero_capture_mmhand.urdf` 及全部 NERO DAE/STL 和 MMHand STL。
- `assets/objects/` 下的 bowl、cup、can 和 box OBJ。
- 版本化 UDP 协议、纯 SAPIEN 场景、启动器和测试。

运行时不会从相邻仓库读取 URDF、mesh 或 Python 代码；启动器也会清除
继承的 `PYTHONPATH`/`PYTHONHOME`，并要求两个 Python 解释器都在本项目内。

项目之外仍需要操作系统提供以下能力：

- D435 或已配置的普通双目相机；
- Linux 相机/USB 权限；
- GUI 模式所需的 `DISPLAY` 和可用 Vulkan 驱动。

代码和资产可以随整个项目复制，但 `.venv` 中含有指向本机 Python 的符号链接和
Linux 原生 wheel，**不能假设 `.venv` 可以跨机器或跨系统直接复制**。换机器后应在
新机器上重建 `.venv`。

## 2. 环境配置

### 2.1 当前机器直接运行

当前 checkout 的项目内 `.venv` 已配置完成，实际使用 Python 3.10.12、
NumPy 1.26.4、OpenCV 4.5.4 和 SAPIEN 3.0.3。从项目根目录检查：

```bash
cd /media/blank/OS/Users/DELL/Desktop/SLAI/project/code/MultiViewHandCapture
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=/tmp/mvhc-matplotlib

.venv/bin/python -c \
  "import numpy, cv2, mediapipe, sapien; print(numpy.__version__, cv2.__version__, mediapipe.__version__, sapien.__version__)"
```

无需 `source` 激活环境，后续命令统一显式使用 `.venv/bin/python`，避免意外调用
Conda 或系统 Python。

### 2.2 从零重建低存储 Python 3.10 `.venv`

不要为此创建 `environment.yml` 中的 ROS/Conda 环境。在当前 Ubuntu 机器上，最省空间的
方式是复用已有的系统 NumPy/OpenCV，再只安装项目缺少的 wheel。
当前机器已有所需系统包，不需要再运行 `apt`。其他 Ubuntu 22.04 机器如果缺少
Python venv 或 OpenCV 所需共享库，可先安装：

```bash
sudo apt install python3.10 python3.10-venv libgl1 libglib2.0-0 libportaudio2
```

先确认系统 Python 3.10 能找到它们：

```bash
/usr/bin/python3.10 -c "import numpy, cv2; print(numpy.__version__, cv2.__version__)"
```

如果这条命令成功，可以按当前机器的方式重建：

```bash
cd /media/blank/OS/Users/DELL/Desktop/SLAI/project/code/MultiViewHandCapture
/usr/bin/python3.10 -m venv .venv

VENV_SITE="$PWD/.venv/lib/python3.10/site-packages"
printf '%s\n' \
  /usr/local/lib/python3.10/dist-packages \
  /usr/lib/python3/dist-packages \
  > "$VENV_SITE/mvhc_system_packages.pth"
```

如果是其他机器，上述系统路径不一定存在；不要盲目复制 `.pth` 或旧 `.venv`。
改为在新 `.venv` 内安装基础包：

```bash
.venv/bin/python -m pip install --no-cache-dir \
  "numpy==1.26.4" "opencv-python==4.10.0.84" \
  "matplotlib==3.7.5" "requests>=2.22,<3" "lxml>=4.8" \
  "tqdm>=4,<5" "six>=1.16" "typing-extensions>=4"
```

当前机器采用系统包复用时，NumPy、OpenCV、Matplotlib、requests、lxml、tqdm、
six 和 typing-extensions 已由上述 `.pth` 提供，不需重复安装。然后两种方式
都继续执行：

```bash
.venv/bin/python -m pip install --no-cache-dir \
  "scipy==1.14.1" "absl-py==2.5.0" "attrs==23.2.0" \
  "flatbuffers==25.12.19" "protobuf==4.25.9" \
  "sounddevice==0.5.5" "sentencepiece==0.2.2" \
  "imageio==2.37.4" "msgspec==0.21.1" "networkx==3.4.2" \
  "rich==14.3.4" "websockets==16.1.1" "zstandard==0.25.0"

.venv/bin/python -m pip install --no-cache-dir --no-deps \
  "mediapipe==0.10.21" "pyrealsense2==2.58.3.10794" \
  "viser==1.0.30" "yourdfpy==0.0.60"

.venv/bin/python -m pip install --no-cache-dir --no-deps \
  -r teleop_maniskill/requirements-sapien-minimal.txt
```

`--no-cache-dir` 避免保留 pip wheel 缓存。`mediapipe` 故意使用 `--no-deps`，因为本项目的
手部 solution 不需要 JAX/JAXlib，也不需要在已有 OpenCV 之外再安装一份
`opencv-contrib-python`。SAPIEN 同样按固定的最小依赖表安装，不会引入 ManiSkill 或
Torch。

安装后验证实际导入：

```bash
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=/tmp/mvhc-matplotlib

.venv/bin/python - <<'PY'
import cv2
import mediapipe
import numpy
import pyrealsense2
import sapien
import scipy
import trimesh
import transforms3d
import viser
import yourdfpy

print("Python environment OK")
print("NumPy", numpy.__version__, "OpenCV", cv2.__version__)
print("MediaPipe", mediapipe.__version__, "SAPIEN", sapien.__version__)
PY
```

## 3. 相机配置

在项目根目录的 `config.py` 选择相机：

```python
CAMERA_TYPE = "d435"  # "d435" 或 "stereo"
CAMERA_INDEX = 0
```

- `d435` 直接读取 D435 左右红外相机和出厂标定，不需要棋盘标定。
- `stereo` 读取单路 `2560×720` 左右拼接画面，并使用项目根目录的
  `stereo_params.json`。首次使用或改变镜头位置后先运行 `.venv/bin/python calibrate.py`。

捕捉 Web 页默认为 <http://localhost:8080>。程序先收集 100 个有效样本估计骨长，
连续识别时大约需要 10 秒。只有页面状态进入 `GESTURE TRACKING` 后，有效左手
目标才会驱动仿真 MMHand。

## 4. 启动方式

### 4.1 一键启动（推荐）

在 `MultiViewHandCapture/` 根目录运行：

```bash
.venv/bin/python teleop_maniskill/run_sim_teleop.py
```

默认监听 `127.0.0.1:5557`、启动 bowl case，并使用项目内同一个 `.venv`
启动捕捉和仿真。例如直接启动 cup，并固定物体的 reset 位姿：

```bash
.venv/bin/python teleop_maniskill/run_sim_teleop.py \
  --object-case cup --fixed-object --rotation-speed 45
```

查看全部启动参数：

```bash
.venv/bin/python teleop_maniskill/run_sim_teleop.py --help
```

### 4.2 分开调试

终端 1：启动捕捉、Web 页和 UDP 发布。

```bash
.venv/bin/python track.py --mode retarget --udp 127.0.0.1:5557
```

终端 2：启动纯 SAPIEN 仿真。

```bash
.venv/bin/python teleop_maniskill/sim_pick_place.py \
  --listen 127.0.0.1:5557 --object-case bowl
```

仿真以 100 Hz 运行物理，以 20 Hz 更新控制。默认手臂速度是 `0.12 m/s`，
每个控制步约 `6 mm`，且单步位移不超过 `2 cm`。MMHand 目标的变化率
限制为 `6 rad/s`，即单个 20 Hz 控制步最多变化 `0.30 rad`。为提高圆柱、杯子等物体的夹持稳定性，MMHand 的全部碰撞形状
默认使用静摩擦 `2.0`、动摩擦 `1.0` 的指腹材料；目标物体和桌面仍保持原来的
`0.3/0.3`，因此不会因为这次调整额外粘在桌面上。启动时终端的 `[CONTACT]`
行会打印实际采用的摩擦配置和覆盖的手部碰撞形状数量。

在线遥操采用更适合厘米级手指接触的物理参数：contact offset 为 `3 mm`，solver
position/velocity iterations 为 `25/4`，全部主动关节 friction 为 `0`，动态物体的
linear/angular damping 为 `2/2`。较低物体阻尼使固定手势随 EE 推进时优先推动物体，
而不是让轻质指节明显退让；这些参数不会改变 IK、手部目标角或 URDF 运动学。

调试时可使用其他端口，但发送和接收必须完全一致：

```bash
.venv/bin/python track.py --mode retarget --udp 127.0.0.1:5560
.venv/bin/python teleop_maniskill/sim_pick_place.py --listen 127.0.0.1:5560
```

## 5. 键盘控制

按键需要 SAPIEN Viewer 窗口获得焦点。

| 按键 | 功能 |
|---|---|
| `Up` / `Down` | NERO 末端沿世界 X 正/负方向连续移动 |
| `Left` / `Right` | NERO 末端沿世界 Y 正/负方向连续移动 |
| `U` / `J` | NERO 末端沿世界 Z 正/负方向连续移动 |
| `I` / `K` | EE 绕自身局部 X 轴 roll 正/负旋转 |
| `O` / `L` | EE 绕自身局部 Y 轴 pitch 正/负旋转 |
| `P` / `M` | EE 绕自身局部 Z 轴 yaw 正/负旋转 |
| `1` / `2` / `3` | front / left-rear / right-rear 视角 |
| `4` | bowl OBJ，切换后 reset |
| `5` | cup OBJ，切换后 reset |
| `6` | can OBJ，切换后 reset |
| `7` | box OBJ，切换后 reset |
| `8` / `9` / `0` | procedural cube / cylinder / sphere，切换后 reset |
| `Space` | 暂停/恢复手臂和手部控制 |
| `N` | 切换到张手姿态；下一个有效捕捉帧自动恢复跟随 |
| `R` | 重置机器人和当前物体，并重新设置 self-collision mask |
| `H` | 在终端重新打印按键帮助 |
| `Q` / `Esc` | 退出仿真 |

终端以 5 Hz 显示 `WAIT/LIVE/HOLD`、UDP age、TCP XYZ 和
`contact/grasp/lift/in_place/success`。`hand_err` 与 `thumb_err` 分别表示全部手关节和
拇指关节的最大 `|drive target - actual q|`，可用于判断接触时手势是否被明显推离。

## 6. 物体 case 和自定义 OBJ

### 6.1 内置物体

| case | 类型 | 默认尺寸/scale |
|---|---|---|
| `bowl` | 本地 bowl OBJ | `--object-scale 0.08` |
| `cup` | UltraDexGrasp DGN 物体库中的原始连体 mug OBJ | 约 `11.74 × 8.94 × 6.78 cm` |
| `can` | 带小倒角的圆罐本地 OBJ | 约 `6.08 × 6.08 × 8 cm` |
| `box` | 长方体纸盒本地 OBJ | 约 `6.8 × 5.2 × 8 cm` |
| `cube` | SAPIEN procedural box | `7 × 7 × 7 cm` |
| `cylinder` | SAPIEN procedural cylinder | 半径 `3 cm`，高 `10 cm` |
| `sphere` | SAPIEN procedural sphere | 半径 `3.5 cm` |

cup 是从 UltraDexGrasp 随附 DGN 物体库复制进来的原始
`mujoco_Cole_Hardware_Mug_Classic_Blue` 网格；杯把和杯身属于同一个连通 mesh。
can 和 box 由 `assets/objects/generate_builtin_objects.py` 在本地生成。默认
`--object-scale 0.08`。为与 UltraDexGrasp 的物体加载逻辑一致，所有 OBJ 都使用
`simplified.obj` 作为 visual mesh，并从同一文件生成一个 convex collision；因此 mug
内腔不是可以放入小物体的精确凹碰撞体。cup 目录也保留了上游 COACD URDF，当前运行
时尚未启用它。

默认每次 reset 会在任务范围内随机化物体 XY 和 yaw。需要可复现的固定姿态时使用：

```bash
.venv/bin/python teleop_maniskill/run_sim_teleop.py \
  --object-case box --seed 0 --fixed-object
```

### 6.2 使用自定义 OBJ

将 OBJ 放在 `MultiViewHandCapture/` 内部，例如：

```text
teleop_maniskill/assets/objects/my_object/mesh/simplified.obj
```

然后覆盖 bowl/cup/can/box 中的一个 mesh case：

```bash
.venv/bin/python teleop_maniskill/run_sim_teleop.py \
  --object-case cup \
  --object-mesh-path teleop_maniskill/assets/objects/my_object \
  --object-scale 0.001
```

`--object-mesh-path` 可以直接指向 `.obj`，也可以指向包含
`mesh/simplified.obj` 或 `simplified.obj` 的目录。这个参数只能和
`--object-case bowl|cup|can|box` 搭配，不能覆盖 procedural case。

为了维持自包含，代码会拒绝项目根目录之外的路径。OBJ 应是有限坐标、非空的
三角网格；`--object-scale` 是从 OBJ 原始单位到米的统一缩放。物体太大、太小或
初始高度不正常时，先检查 OBJ bounds 和 scale。

## 7. 常用参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--listen` | `127.0.0.1:5557` | UDP 接收端点；一键启动会同时传给捕捉端 |
| `--arm-speed` | `0.12` | 键盘控制的世界系 TCP 速度，单位 m/s |
| `--rotation-speed` | `45` | EE 局部 roll/pitch/yaw 旋转速度，单位 deg/s |
| `--object-case` | `bowl` | 启动时的物体 case |
| `--object-mesh-path` | 空 | 用项目内 OBJ 覆盖当前 mesh case |
| `--object-scale` | `0.08` | OBJ 统一缩放 |
| `--seed` | `0` | reset 随机数种子 |
| `--fixed-object` | 关 | 固定物体 XY/yaw，不做 reset 随机化 |
| `--headless` | 关 | 不创建 Viewer，只运行 CPU 物理 |
| `--max-steps` | `0` | 20 Hz 控制步数上限；`0` 表示持续运行 |
| `--udp-timeout` | `0.25` | 只用于 `sim_pick_place.py`；超时后进入 `HOLD` |
| `--startup-delay` | `0.75` | 只用于启动器；仿真启动后延迟多少秒再启动捕捉 |

## 8. 验证与测试

所有命令都从 `MultiViewHandCapture/` 根目录执行。

验证高精度 URDF 的 42-link/28-DoF 结构、本地 mesh 路径和 manifest：

```bash
.venv/bin/python teleop_maniskill/prepare_full_fidelity_urdf.py validate
```

运行 teleop 独立回归测试：

```bash
.venv/bin/python -m unittest -v \
  teleop_maniskill.test_full_fidelity_urdf \
  teleop_maniskill.test_object_assets \
  teleop_maniskill.test_standalone_urdf \
  teleop_maniskill.test_teleop_protocol \
  teleop_maniskill.test_sim_pick_place
```

不打开窗口、不连接相机的物理烟雾测试：

```bash
.venv/bin/python teleop_maniskill/sim_pick_place.py \
  --headless --max-steps 20 --object-case cup --fixed-object
```

`--headless` 只验证 CPU 物理、URDF、物体和控制链，不能替代 GUI/Vulkan 验收。
最终人工验收应包含：三视角、全部物体 case、六个方向键、左手张合、
`WAIT/LIVE/HOLD`、暂停、张手和 reset。

## 9. 常见问题

### `ModuleNotFoundError`

确认从项目根目录使用 `.venv/bin/python`，而不是系统 `python` 或 Conda。运行第
2 节的导入检查。若刚复制到新机器，请重建 `.venv`，不要修补从旧机器复制的
Python 符号链接。

### Viewer 不出现、Vulkan 报错或原生层崩溃

GUI 需要有效的 `DISPLAY` 和 Vulkan ICD/显卡驱动。本程序使用 SAPIEN default raster
shader，不会自动尝试 RT；当前设备上 RT 后端不可用，不要手动改成 RT。先用
`--headless --max-steps 20` 区分物理/资产问题与渲染驱动问题。

### `VIDIOC_S_FMT ... Device or resource busy`

相机已被其他 `track.py`、RealSense Viewer 或视频程序占用。先用 `Ctrl-C` 正常关闭
原进程，再启动本程序。

### 仿真显示 `WAIT` 或 `HOLD`

- `WAIT`：从未收到 UDP 包；检查两端的 host/port 是否一致。
- `HOLD`：包超时、当前帧无效、仍在标定、检测到右手或求解失败。确认 Web 页已
  进入 `GESTURE TRACKING`，并且视野中是左手。

`HOLD` 不会把手指瞬间弹回零位，而是保持最后一个有效姿态。

### `Address already in use`

上一个仿真可能仍在占用 UDP 端口。关闭旧进程，或使用新端口；分开启动时一定要
同时修改 `track.py --udp` 和 `sim_pick_place.py --listen`。

### 按键无效或不能继续移动

先点击 Viewer 获取键盘焦点，再检查是否按过 `Space` 进入暂停。如果终端打印
`[IK] target rejected`，表示目标超出当前手臂可达范围；沿相反方向退回即可。

### 自定义 OBJ 被拒绝或尺寸不正常

只接受项目内的 `.obj` 文件。不能将 `--object-mesh-path` 与
`cube/cylinder/sphere` 搭配。确认 mesh 非空、坐标有限，并按模型原始单位调整
`--object-scale`。

### Web 页打不开

检查 `track.py` 是否仍在运行，以及 `config.py` 中的 `WEB_PORT`。默认地址是
<http://localhost:8080>；若端口已被其他程序占用，需先关闭占用者或修改配置。

## 10. 资产来源与许可

- 高精度 NERO + MMHand 组合资产的来源、修改和 SHA-256 manifest 见
  `assets/robot/ASSET_PROVENANCE.md` 和 `assets/robot/MANIFEST.json`。
- NERO 来源仓库随附的 Apache-2.0 文本保存在
  `assets/robot/LICENSE-UltraDexGrasp-Apache-2.0.txt`。
- MMHand 资产随附的 dex-retargeting MIT notice 保存在
  `assets/robot/LICENSE-MMHand-dex-retargeting-MIT.txt`。
- bowl 和 cup 的来源说明见 `assets/objects/ASSET_PROVENANCE.md`；cup 是从
  UltraDexGrasp 随附 DGN 物体库原样复制的 mug。can 和 box 是本地程序化生成的
  原始 mesh。每个目录中都有独立 `ASSET_PROVENANCE.md`。

随附的仓库许可文本不等于 NERO/MMHand OEM CAD 的权属和对外再分发权已经
独立明确。当前 bundle 可用于本地研究；若要公开发布或再分发 OEM mesh，应先向
模型提供方确认权利。自定义 OBJ 的许可和来源由添加者自行记录和负责。
