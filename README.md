# mmhand_teleop

`mmhand_teleop` converts a monocular camera or a MANUS glove into one common
21-landmark hand representation, retargets a left hand to the 21 joints of
MMHand, and exposes a browser dashboard and optional ROS 2 output.

The default input is a V4L2 camera running the CUDA ONNX version of WiLoR. A
camera may be a USB/UVC device or an integrated webcam exposed as
`/dev/videoN`. No depth or multi-camera calibration is required.

## Installation

The WiLoR models use Git LFS. Install the repository and Conda environment with:

```bash
git lfs install
git clone https://github.com/mm-hand/mmhand_teleop.git
cd mmhand_teleop
git lfs pull
mamba env create -f environment.yml
conda activate mmhand
python -m pip install -r input/wilor/requirements.txt
```

The last command installs ONNX Runtime with its CUDA and cuDNN dependencies.
WiLoR requires `CUDAExecutionProvider`. SAPIEN is optional:

```bash
python -m pip install -r simulation/requirements.txt
```

## Configuration

Runtime configuration lives in `config.py`. Select exactly one input:

```python
INPUT_SOURCE = "wilor"  # "wilor" or "manus"
```

For WiLoR, configure any OpenCV source and its requested mode:

```python
CAMERA_DEVICE = 0       # V4L2 index or "/dev/videoN"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
WILOR_DEVICE_ID = 0
```

The camera prints the actual mode negotiated with the device. WiLoR keypoints
and finger-pad directions have independent One Euro settings:

```python
WILOR_POINT_FILTER = (0.5, 1.0, 1.0)
WILOR_DIRECTION_FILTER = (0.5, 0.25, 1.0)
WILOR_THUMB_PAD_ROTATION_DEG = 45.0
```

Each tuple is `(min_cutoff, beta, derivative_cutoff)`. Filters reset when the
hand disappears, an inference result is invalid, or handedness changes.
Thumb-pad alignment and the two independent human-to-robot thumb angle scales
are configured separately:

```python
RETARGET_THUMB_PAD_WEIGHT = 0.25
RETARGET_THUMB_MCP_ANGLE_SCALE = 1.0
RETARGET_THUMB_IP_ANGLE_SCALE = 1.0
RETARGET_THUMB_TO_FINGERTIPS_WEIGHT = 1.0
```

For MANUS, place/build the bundled Core SDK bridge as described in
`input/manus/assets/README.md`, then set `INPUT_SOURCE = "manus"`.
`MANUS_PAD_LOCAL_AXIS` selects the outward pad axis in every Tip node's local
frame and defaults to `(0, 0, -1)`. Handedness comes directly from the SDK
frame or its NodeInfo topology.

## Running

```bash
python track.py
python track.py --ros
python track.py --sim
```

`--ros` and `--sim` are mutually exclusive. Every other option belongs in
`config.py`.

Open <http://localhost:8080>. The dashboard contains the input preview, the
normalized hand, and the retargeted MMHand. WiLoR overlays the detected box and
yellow MANO mesh on the camera image. Its status shows confidence followed by
smoothed tracking FPS. Both hand views show finger-pad directions, thumb angle
arcs, and live angle values in degrees. Human values are the original measured
angles; the scale coefficients affect only retargeting.

Only a ready left hand enters retargeting. A right hand can still be displayed
and published.

## Common input format

Every input returns `input.frame.InputFrame`:

```python
@dataclass(slots=True)
class InputFrame:
    timestamp: float
    points: np.ndarray | None
    handedness: str | None
    ready: bool
    status: str
    finger_pad_directions: np.ndarray | None = None
    preview: np.ndarray | None = None
```

`points` is a finite `21 x 3` array in metres. Its order is wrist followed by
Thumb, Index, Middle, Ring, and Little, with four joints from palm to tip per
finger. The wrist is the origin. The axes are palm-local and the mean distance
from wrist to the four MCP joints is normalized to `0.086 m`.

`finger_pad_directions` is a finite `5 x 3` array in Thumb-to-Little order. Each
row is an outward unit vector in the same palm-local frame. WiLoR produces this
field from selected MANO pad faces. MANUS rotates `MANUS_PAD_LOCAL_AXIS` by the
five Tip-node WORLD quaternions before transforming it into the palm frame.

`timestamp` is the monotonic acquisition time. Invalid or missing data uses
`points=None`, `finger_pad_directions=None`, and `ready=False`. Every frame with
points must also contain five finite unit pad directions; malformed frames raise
an error. Device-specific data never crosses this input boundary. Retargeting
receives the common points, pad directions, and timestamp from the same frame.

## Processing

```text
OpenCV frame -> detector/cached crop -> WiLoR MANO mesh
             -> 21 points + 5 pad normals -> palm-local normalization
             -> One Euro filters -> InputFrame

MANUS Raw Skeleton -> semantic/fallback 25-to-21 mapping
                   + Tip-node WORLD rotations -> 5 pad directions
                   -> palm-local normalization -> InputFrame

InputFrame -> direct four-finger mapping + thumb SLSQP -> MMHand J00-J20
           -> dashboard / ROS 2 / optional SAPIEN simulation
```

WiLoR runs its detector and reconstruction models in FP16, then performs box
and NMS arithmetic in FP32. The MANO mesh is regressed to standard landmarks.
Pad-face normals are averaged, mirrored for handedness, transformed into the
palm frame, and normalized. The thumb normal is rotated by the configured fixed
angle around its IP-to-TIP axis: positive for Left and negative for Right.

Four MMHand fingers are mapped analytically. Five thumb joints are solved with
bounded SLSQP using analytic Jacobians and the previous solution as a warm
start. The human MCP and IP flexion angles are unsigned 3D angles in `[0, pi]`,
independently scaled, and matched directly to MMHand J18/PIP and J19/DIP. Pad
direction and handedness do not affect these angles. Four complete vectors from
the thumb tip to the other fingertips are matched in the palm-local frames. The
other objectives align the thumb tip position and thumb-pad
direction with the local `-Z` direction of MMHand's `5-tip_Link`. The dashboard
reports every weighted term in real time. Final robot angles have a separate
One Euro filter and are clipped to the limits read from the MMHand URDF.

## Robot assets

The robot model is `assets/mmhand/urdf/mmhand_collision_coacd.urdf`. It and its
27 visual meshes plus 162 convex collision meshes come from
[`mm-hand/structure`](https://github.com/mm-hand/structure) commit
`04eb24bedc300419d8556de1e3848c6d4e344d4e`. This repository adds five fixed,
geometry-free virtual fingertip links used by retargeting. Every STL is tracked
with Git LFS, so `git lfs pull` is required after cloning.

## ROS 2

`python track.py --ros` publishes `std_msgs/msg/Float32MultiArray`:

| Topic | Shape | Meaning |
|---|---:|---|
| `/hand/keypoints` | `21 x 3` | palm-local normalized points in metres |
| `/hand/finger_pad_directions` | `5 x 3` | palm-local outward unit vectors |
| `/raw_ik_target` | `21` | MMHand J00-J20 in degrees |

Every frame published on the keypoint topic has a matching direction message.
Keypoint and direction layouts include handedness. The layouts are:

```text
mmhand_teleop:keypoints:v1:palm_local_m:size=0.086
mmhand_teleop:finger_pad_directions:v1:palm_local_unit
mmhand:J00-J20:urdf_deg
```

## Repository structure

```text
input/
  frame.py              common frame, topology, and palm normalization
  wilor/
    camera.py           latest-frame OpenCV capture
    source.py           ONNX inference, mesh geometry, box caching, filtering
    assets/              ONNX/MANO data and license notices
  manus/
    adapter.py          25-to-21 mapping and quaternion pad directions
    source.py           official SDK transport and common-frame source
    assets/              SDK bridge, headers, library, and documentation
assets/mmhand/           MMHand URDF, meshes, and licenses
config.py                all runtime settings and public ROS constants
one_euro.py              ndarray One Euro filter
retarget.py              URDF FK, analytic mapping, thumb optimization
ros.py                   keypoint, pad-direction, and joint publishers
viewer.py                dashboard and both 3D views
track.py                 minimal runtime lifecycle
simulation/              optional SAPIEN grasp simulation
test_hand_capture.py     retarget, viewer, and ROS tests
test_wilor.py            camera, inference geometry, filter tests
test_manus_phase1.py     MANUS mapping and stale-data tests
test_simulation.py       optional headless simulation tests
```

The source implementations end at `InputFrame`. Retargeting, visualization,
ROS, and simulation contain no input-device branches.

## Tests and licenses

```bash
python -m unittest -v test_hand_capture.py test_wilor.py test_manus_phase1.py
python -m unittest -v test_simulation.py
```

Simulation tests skip when SAPIEN is unavailable. WiLoR model terms are in
`input/wilor/assets/WILOR_MODEL_LICENSE.txt`; related notices are in
`input/wilor/assets/THIRD_PARTY_NOTICES.md`. MANUS SDK terms are distributed
with its assets.
