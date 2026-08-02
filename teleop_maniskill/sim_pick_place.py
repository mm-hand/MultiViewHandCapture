#!/usr/bin/env python3
"""Standalone SAPIEN keyboard NERO + live capture-native MMHand teleoperation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import sapien
import trimesh
from transforms3d.quaternions import axangle2quat, mat2quat

if __package__:
    from .prepare_full_fidelity_urdf import (
        ARM_ACTIVE_JOINT_NAMES as ARM_JOINT_NAMES,
        DEFAULT_OUTPUT_URDF as FULL_ROBOT_URDF,
        HAND_ACTIVE_JOINT_NAMES as HAND_JOINT_NAMES,
        validate_urdf,
    )
    from .teleop_protocol import (
        JOINT_NAMES,
        LatestUdpRetargetReceiver,
        RetargetPacket,
        capture_to_sim,
        rate_limit,
    )
else:
    # Direct execution keeps imports rooted in this checkout only.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from teleop_maniskill.prepare_full_fidelity_urdf import (
        ARM_ACTIVE_JOINT_NAMES as ARM_JOINT_NAMES,
        DEFAULT_OUTPUT_URDF as FULL_ROBOT_URDF,
        HAND_ACTIVE_JOINT_NAMES as HAND_JOINT_NAMES,
        validate_urdf,
    )
    from teleop_maniskill.teleop_protocol import (
        JOINT_NAMES,
        LatestUdpRetargetReceiver,
        RetargetPacket,
        capture_to_sim,
        rate_limit,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

SIM_FREQ = 100
CONTROL_FREQ = 20
PHYSICS_DT = 1.0 / SIM_FREQ
CONTROL_DT = 1.0 / CONTROL_FREQ
PHYSICS_STEPS_PER_CONTROL = SIM_FREQ // CONTROL_FREQ
HAND_RATE_LIMIT_RAD_S = 6.0
ARM_STEP_LIMIT_M = 0.02
EE_ROTATION_STEP_LIMIT_RAD = math.radians(10.0)
DEFAULT_ROTATION_SPEED_DEG_S = 45.0
IK_POSITION_TOLERANCE_M = 0.002
IK_ORIENTATION_TOLERANCE_RAD = math.radians(2.0)
SOLVER_POSITION_ITERATIONS = 25
SOLVER_VELOCITY_ITERATIONS = 4
CONTACT_OFFSET_M = 0.003
ACTIVE_JOINT_FRICTION = 0.0
OBJECT_LINEAR_DAMPING = 2.0
OBJECT_ANGULAR_DAMPING = 2.0

# The capture-native hand uses hard collision geometry rather than deformable
# pads.  Give every MMHand collision shape a fingertip-like material while
# leaving the arm, object, and table at the original 0.3/0.3 friction.  PhysX
# combines the two colliding materials, so changing only the hand isolates this
# first grip experiment without making objects stick to the table.
HAND_STATIC_FRICTION = 2.0
HAND_DYNAMIC_FRICTION = 1.0

ACTION_DIM = 24
ARM_ACTION_SLICE = slice(0, 3)
HAND_ACTION_SLICE = slice(3, 24)

ARM_HOME_QPOS = np.asarray(
    [-0.1, 0.55, 0.0, 1.05, 0.0, 0.0, 0.0], dtype=np.float64
)
HAND_OPEN_QPOS = np.zeros(21, dtype=np.float64)
HAND_OPEN_QPOS[[0, 4, 8, 12, 16]] = np.radians((36, 29, 31, 23, 28))

ROBOT_ROOT_POSE = sapien.Pose([0.0, 0.45, 0.714], [0.0, 0.0, 0.0, 1.0])
TCP_LINK_NAME = "capture_hand__base_link"
TABLE_TOP_Z = 0.714
OBJECT_SPAWN_XY = np.asarray([0.55, 0.35], dtype=np.float64)
OBJECT_XY_CENTER = np.asarray([0.56, 0.3533], dtype=np.float64)
OBJECT_XY_SPAN = np.asarray([0.20, 0.40], dtype=np.float64)
PLACE_CENTER_XY = np.asarray([0.20, -0.05], dtype=np.float64)
PLACE_RADIUS_M = 0.09
MESH_OBJECT_PATHS = {
    "bowl": SCRIPT_DIR / "assets/objects/bowl/mesh/simplified.obj",
    "cup": SCRIPT_DIR / "assets/objects/cup/mesh/simplified.obj",
    "can": SCRIPT_DIR / "assets/objects/can/mesh/simplified.obj",
    "box": SCRIPT_DIR / "assets/objects/box/mesh/simplified.obj",
}
MESH_OBJECT_RENDER = {
    "bowl": ([0.85, 0.85, 0.85, 1.0], 0.42, 0.00),
    "cup": ([0.93, 0.52, 0.16, 1.0], 0.38, 0.00),
    "can": ([0.72, 0.12, 0.10, 1.0], 0.30, 0.20),
    "box": ([0.58, 0.34, 0.16, 1.0], 0.62, 0.00),
}
DEFAULT_OBJECT_SCALE = 0.08
PROCEDURAL_OBJECT_CASES = ("cube", "cylinder", "sphere")
OBJECT_CASES = (*MESH_OBJECT_PATHS, *PROCEDURAL_OBJECT_CASES)
OBJECT_HOTKEYS = {
    "4": "bowl",
    "5": "cup",
    "6": "can",
    "7": "box",
    "8": "cube",
    "9": "cylinder",
    "0": "sphere",
}
FINGERTIP_LINK_TO_FINGER = {
    "capture_hand__finger_1_fingertip_1": "index",
    "capture_hand__finger_2_fingertip_1": "middle",
    "capture_hand__finger_3_fingertip_1": "ring",
    "capture_hand__finger_4_fingertip_1": "little",
    "capture_hand__mmhand_thumb_1_finger_7_fingertip_1": "thumb",
}

_CAMERA_TARGET = np.asarray([0.2533333333, 0.2511, TABLE_TOP_Z + 0.28])
CAMERAS = {
    "front": (_CAMERA_TARGET + np.asarray([1.25, 0.0, 0.72]) * 0.85, _CAMERA_TARGET),
    "left-rear": (
        _CAMERA_TARGET + np.asarray([-0.85, 0.90, 0.72]) * 0.85,
        _CAMERA_TARGET,
    ),
    "right-rear": (
        _CAMERA_TARGET + np.asarray([-0.85, -0.90, 0.72]) * 0.85,
        _CAMERA_TARGET,
    ),
}

if tuple(HAND_JOINT_NAMES) != tuple(JOINT_NAMES):
    raise RuntimeError("full-fidelity URDF and UDP hand-joint contracts differ")


@dataclass(frozen=True)
class TaskMetrics:
    contact: bool
    contact_fingers: int
    grasp: bool
    lift: bool
    in_place: bool
    success: bool


def _key_down(window: Any, *keys: str) -> bool:
    for key in keys:
        try:
            if bool(window.key_down(key)):
                return True
        except Exception:
            continue
    return False


def _key_press(window: Any, *keys: str) -> bool:
    for key in keys:
        try:
            if bool(window.key_press(key)):
                return True
        except Exception:
            continue
    return False


def _camera_rpy_from_look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    delta = target - eye
    horizontal = float(np.linalg.norm(delta[:2]))
    return np.asarray(
        [0.0, math.atan2(float(delta[2]), horizontal), -math.atan2(delta[1], delta[0])],
        dtype=np.float64,
    )


def _camera_pose_from_look_at(eye: np.ndarray, target: np.ndarray) -> sapien.Pose:
    """Return a SAPIEN camera pose whose local +X axis looks at the target."""

    forward = np.asarray(target - eye, dtype=np.float64)
    forward /= max(float(np.linalg.norm(forward)), 1e-9)
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    left = np.cross(world_up, forward)
    if float(np.linalg.norm(left)) < 1e-8:
        left = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    up /= max(float(np.linalg.norm(up)), 1e-9)
    rotation = np.stack([forward, left, up], axis=1)
    return sapien.Pose(eye, mat2quat(rotation))


def _finger_from_link(link_name: str) -> str | None:
    return FINGERTIP_LINK_TO_FINGER.get(link_name)


def _render_material(
    base_color: Sequence[float],
    *,
    roughness: float = 0.55,
    metallic: float = 0.0,
    specular: float = 0.35,
) -> sapien.render.RenderMaterial:
    """Create a PBR material without losing an RGBA alpha channel."""

    material = sapien.render.RenderMaterial()
    material.set_base_color(np.asarray(base_color, dtype=np.float32))
    material.set_roughness(float(roughness))
    material.set_metallic(float(metallic))
    material.set_specular(float(specular))
    return material


def _project_object_mesh(path: str | Path) -> Path:
    """Resolve an OBJ kept in this checkout, accepting an object-root path."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    if candidate.is_dir():
        for relative in (Path("mesh/simplified.obj"), Path("simplified.obj")):
            mesh = candidate / relative
            if mesh.is_file():
                candidate = mesh
                break
    resolved = candidate.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError(
            "object mesh must be copied inside the MultiViewHandCapture project: "
            f"{resolved}"
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"object mesh does not exist: {resolved}")
    if resolved.suffix.lower() != ".obj":
        raise ValueError(f"object mesh must be an OBJ file: {resolved}")
    return resolved


def _configure_sapien_defaults(render_enabled: bool) -> None:
    """Configure contact-stable teleoperation physics and raster MSAA."""

    scene_config = sapien.physx.get_scene_config()
    scene_config.gravity = np.asarray([0.0, 0.0, -9.81], dtype=np.float32)
    scene_config.bounce_threshold = 2.0
    scene_config.enable_pcm = True
    scene_config.enable_tgs = True
    scene_config.enable_friction_every_iteration = True
    sapien.physx.set_scene_config(scene_config)

    body_config = sapien.physx.get_body_config()
    body_config.sleep_threshold = 0.005
    body_config.solver_position_iterations = SOLVER_POSITION_ITERATIONS
    body_config.solver_velocity_iterations = SOLVER_VELOCITY_ITERATIONS
    sapien.physx.set_body_config(body_config)

    shape_config = sapien.physx.get_shape_config()
    shape_config.contact_offset = CONTACT_OFFSET_M
    shape_config.rest_offset = 0.0
    sapien.physx.set_shape_config(shape_config)
    sapien.physx.set_default_material(0.3, 0.3, 0.0)

    if render_enabled:
        # The original environment uses SAPIEN's raster/default shader.  RT is
        # intentionally not auto-selected because unsupported Vulkan drivers
        # can fail below Python rather than raising a catchable exception.
        sapien.render.set_viewer_shader_dir("default")
        sapien.render.set_camera_shader_dir("default")
        sapien.render.set_msaa(4)


class StandalonePickPlace:
    """Pure-SAPIEN task with project-local NERO + capture MMHand assets."""

    def __init__(
        self,
        object_case: str = "bowl",
        object_mesh_path: str | Path | None = None,
        object_scale: float = DEFAULT_OBJECT_SCALE,
        headless: bool = False,
        viewer: bool = True,
        seed: int = 0,
        randomize_object: bool = True,
    ):
        if object_case not in OBJECT_CASES:
            raise ValueError(f"object_case must be one of {OBJECT_CASES}")
        self.headless = bool(headless)
        self.render_enabled = not self.headless
        self.viewer_enabled = self.render_enabled and bool(viewer)
        _configure_sapien_defaults(self.render_enabled)
        if not math.isfinite(object_scale) or object_scale <= 0.0:
            raise ValueError("object_scale must be finite and positive")
        if object_mesh_path is not None and object_case not in MESH_OBJECT_PATHS:
            raise ValueError(
                "object_mesh_path can override only a mesh case: "
                f"{tuple(MESH_OBJECT_PATHS)}"
            )
        self.object_mesh_override = (
            None
            if object_mesh_path is None
            else _project_object_mesh(object_mesh_path)
        )
        self.object_mesh_override_case = (
            None if object_mesh_path is None else object_case
        )
        self.object_mesh_path: Path | None = None
        self.object_scale = float(object_scale)
        self.rng = np.random.default_rng(seed)
        self.randomize_object = bool(randomize_object)
        self.scene = self._create_scene()
        self.scene.set_timestep(PHYSICS_DT)

        self.viewer = None
        self.object = None
        self.object_body = None
        self.object_case = object_case
        self.object_rest_z = TABLE_TOP_Z
        self.sensor_cameras: dict[str, Any] = {}
        self._build_world()
        self.robot = self._load_robot()
        self._configure_robot()
        self.set_object_case(object_case)
        self.reset()

        if self.viewer_enabled:
            self._create_viewer()

    def _create_scene(self):
        if self.headless:
            return sapien.Scene([sapien.physx.PhysxCpuSystem()])
        return sapien.Scene()

    def _build_world(self) -> None:
        self.surface_material = self.scene.create_physical_material(0.3, 0.3, 0.0)
        self.object_material = self.scene.create_physical_material(0.3, 0.3, 0.0)
        self.hand_material = self.scene.create_physical_material(
            HAND_STATIC_FRICTION,
            HAND_DYNAMIC_FRICTION,
            0.0,
        )

        self.scene.add_ground(
            0.0,
            render=self.render_enabled,
            material=self.surface_material,
        )

        builder = self.scene.create_actor_builder()
        builder.add_box_collision(
            half_size=[0.90, 0.80, 0.03],
            material=self.surface_material,
        )
        if self.render_enabled:
            builder.add_box_visual(
                half_size=[0.90, 0.80, 0.03],
                material=_render_material(
                    [180 / 255, 170 / 255, 160 / 255, 1.0],
                    roughness=0.72,
                    specular=0.22,
                ),
            )
        builder.initial_pose = sapien.Pose([0.0, 0.45, TABLE_TOP_Z - 0.03])
        self.table = builder.build_kinematic("table")

        if self.render_enabled:
            marker = self.scene.create_actor_builder()
            marker.add_box_visual(
                half_size=[PLACE_RADIUS_M, PLACE_RADIUS_M, 0.001],
                material=_render_material(
                    [0.10, 0.65, 0.95, 1.0],
                    roughness=0.45,
                    specular=0.25,
                ),
            )
            marker.initial_pose = sapien.Pose(
                [PLACE_CENTER_XY[0], PLACE_CENTER_XY[1], TABLE_TOP_Z + 0.002]
            )
            self.place_marker = marker.build_kinematic("place_region")

            self.scene.set_ambient_light([0.30, 0.30, 0.32])
            self.scene.add_directional_light(
                [1.0, 1.0, -1.0],
                [1.0, 1.0, 0.96],
                shadow=True,
                shadow_scale=5.0,
                shadow_map_size=2048,
            )
            self.scene.add_directional_light(
                [0.0, 0.0, -1.0],
                [0.35, 0.38, 0.45],
                shadow=False,
            )
            self._build_sensor_cameras()

    def _build_sensor_cameras(self) -> None:
        """Create the original three 512-square RGB/depth/seg cameras."""

        for name, (eye, target) in CAMERAS.items():
            camera = self.scene.add_camera(
                name,
                width=512,
                height=512,
                fovy=math.radians(40.0),
                near=0.01,
                far=100.0,
            )
            camera.entity_pose = _camera_pose_from_look_at(eye, target)
            self.sensor_cameras[name] = camera

    def _load_robot(self):
        urdf_path = FULL_ROBOT_URDF.resolve()
        if not urdf_path.is_relative_to(PROJECT_ROOT):
            raise RuntimeError(f"full-fidelity robot escaped project root: {urdf_path}")
        if not urdf_path.is_file():
            raise FileNotFoundError(
                "full-fidelity robot asset is missing; expected " f"{urdf_path}"
            )
        validate_urdf(urdf_path)

        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        builder = loader.load_file_as_articulation_builder(str(urdf_path))
        if builder is None:
            raise RuntimeError(f"SAPIEN could not parse {urdf_path}")
        if self.headless:
            # A physics-only scene intentionally has no RenderSystem.
            for link_builder in builder.link_builders:
                link_builder.visual_records.clear()
        robot = builder.build()
        robot.pose = ROBOT_ROOT_POSE
        if self.render_enabled:
            self._polish_mmhand_materials(robot)
        return robot

    @staticmethod
    def _polish_mmhand_materials(robot: Any) -> None:
        """Retain MMHand colors while giving its STL surfaces readable PBR."""

        for link in robot.links:
            if not link.name.startswith("capture_hand__"):
                continue
            body = link.entity.find_component_by_type(
                sapien.render.RenderBodyComponent
            )
            if body is None:
                continue
            for shape in body.render_shapes:
                for part in shape.parts:
                    part.material.roughness = 0.62
                    part.material.specular = 0.32

    def _configure_robot(self) -> None:
        active_joints = list(self.robot.active_joints)
        active_names = [joint.name for joint in active_joints]
        expected = set(ARM_JOINT_NAMES) | set(HAND_JOINT_NAMES)
        if len(active_names) != 28 or set(active_names) != expected:
            raise RuntimeError(
                "full-fidelity robot joint mismatch: "
                f"active={active_names}, missing={sorted(expected - set(active_names))}, "
                f"extra={sorted(set(active_names) - expected)}"
            )

        self.joints_by_name = {joint.name: joint for joint in active_joints}
        self.joint_index = {name: index for index, name in enumerate(active_names)}
        self.arm_indices = np.asarray(
            [self.joint_index[name] for name in ARM_JOINT_NAMES], dtype=np.int64
        )
        self.hand_indices = np.asarray(
            [self.joint_index[name] for name in HAND_JOINT_NAMES], dtype=np.int64
        )
        limits = np.asarray(self.robot.get_qlimits(), dtype=np.float64)
        self.hand_lower = limits[self.hand_indices, 0].copy()
        self.hand_upper = limits[self.hand_indices, 1].copy()
        self.open_hand = np.clip(HAND_OPEN_QPOS, self.hand_lower, self.hand_upper)

        self.home_qpos = np.zeros(self.robot.dof, dtype=np.float64)
        self.home_qpos[self.arm_indices] = ARM_HOME_QPOS
        self.home_qpos[self.hand_indices] = self.open_hand

        for name in ARM_JOINT_NAMES:
            self.joints_by_name[name].set_friction(ACTIVE_JOINT_FRICTION)
            self.joints_by_name[name].set_drive_properties(
                stiffness=1000.0,
                damping=100.0,
                force_limit=1e10,
                mode="force",
            )
        for name in HAND_JOINT_NAMES:
            self.joints_by_name[name].set_friction(ACTIVE_JOINT_FRICTION)
            self.joints_by_name[name].set_drive_properties(
                stiffness=1000.0,
                damping=100.0,
                force_limit=1e10,
                mode="force",
            )

        self.robot_link_names = {link.name for link in self.robot.links}
        self.hand_friction_shape_count = self.apply_hand_friction()
        self.self_collision_shape_count = self.disable_self_collision()
        self.pinocchio = self.robot.create_pinocchio_model()
        link_names = [link.name for link in self.robot.links]
        if TCP_LINK_NAME not in link_names:
            raise RuntimeError(f"TCP link missing from full robot: {TCP_LINK_NAME}")
        self.tcp_link_index = link_names.index(TCP_LINK_NAME)
        self.arm_active_mask = np.zeros(self.robot.dof, dtype=np.int32)
        self.arm_active_mask[self.arm_indices] = 1
        self.target_qpos = self.home_qpos.copy()
        self.tcp_target_local = sapien.Pose()

    def apply_hand_friction(self) -> int:
        """Apply the high-friction pad material to all MMHand collisions."""

        count = 0
        for link in self.robot.links:
            if not link.name.startswith("capture_hand__"):
                continue
            for shape in link.collision_shapes:
                shape.set_physical_material(self.hand_material)
                count += 1
        return count

    def disable_self_collision(self) -> int:
        """Disable collisions among robot links with SAPIEN group-2 bit 30."""

        count = 0
        for link in self.robot.links:
            for shape in link.collision_shapes:
                groups = list(shape.get_collision_groups())
                groups[2] |= 1 << 30
                shape.set_collision_groups(groups)
                count += 1
        return count

    def _set_all_drive_targets(self) -> None:
        for name, index in self.joint_index.items():
            self.joints_by_name[name].set_drive_target(
                float(self.target_qpos[index])
            )

    def _reset_tcp_target(self) -> None:
        self.pinocchio.compute_forward_kinematics(self.target_qpos)
        pose = self.pinocchio.get_link_pose(self.tcp_link_index)
        self.tcp_target_local = sapien.Pose(
            np.asarray(pose.p, dtype=np.float64),
            np.asarray(pose.q, dtype=np.float64),
        )

    def _build_object(self, object_case: str):
        builder = self.scene.create_actor_builder()
        self.object_mesh_path = None
        mesh_volume: float | None = None
        if object_case in MESH_OBJECT_PATHS:
            mesh_path = (
                self.object_mesh_override
                if object_case == self.object_mesh_override_case
                else _project_object_mesh(MESH_OBJECT_PATHS[object_case])
            )
            if mesh_path is None:
                raise RuntimeError(f"mesh path is missing for {object_case}")
            self.object_mesh_path = mesh_path
            mesh = trimesh.load(mesh_path, force="mesh")
            if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
                raise ValueError(f"could not load a triangle mesh: {mesh_path}")
            bounds = np.asarray(mesh.bounds, dtype=np.float64) * self.object_scale
            if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
                raise ValueError(f"mesh has invalid bounds: {mesh_path}")
            scale = [self.object_scale] * 3
            builder.add_convex_collision_from_file(
                filename=str(mesh_path),
                scale=scale,
                material=self.object_material,
                density=100.0,
            )
            if self.render_enabled:
                builder.add_visual_from_file(
                    filename=str(mesh_path),
                    scale=scale,
                    material=_render_material(
                        MESH_OBJECT_RENDER[object_case][0],
                        roughness=MESH_OBJECT_RENDER[object_case][1],
                        metallic=MESH_OBJECT_RENDER[object_case][2],
                        specular=0.35,
                    ),
                )
            self.object_rest_z = TABLE_TOP_Z - float(bounds[0, 2]) + 0.02
            mesh_volume = abs(float(mesh.volume))
        elif object_case == "cube":
            half_size = [0.035, 0.035, 0.035]
            builder.add_box_collision(
                half_size=half_size,
                material=self.object_material,
                density=235.0,
            )
            if self.render_enabled:
                builder.add_box_visual(
                    half_size=half_size,
                    material=_render_material(
                        [0.92, 0.24, 0.18, 1.0], roughness=0.36
                    ),
                )
            rest_height = 0.035
        elif object_case == "cylinder":
            # SAPIEN cylinders point along local X; rotate local X onto world Z.
            local_pose = sapien.Pose(
                q=[math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0]
            )
            builder.add_cylinder_collision(
                pose=local_pose,
                radius=0.030,
                half_length=0.050,
                material=self.object_material,
                density=285.0,
            )
            if self.render_enabled:
                builder.add_cylinder_visual(
                    pose=local_pose,
                    radius=0.030,
                    half_length=0.050,
                    material=_render_material(
                        [0.18, 0.40, 0.92, 1.0], roughness=0.32
                    ),
                )
            rest_height = 0.050
        elif object_case == "sphere":
            builder.add_sphere_collision(
                radius=0.035,
                material=self.object_material,
                density=445.0,
            )
            if self.render_enabled:
                builder.add_sphere_visual(
                    radius=0.035,
                    material=_render_material(
                        [0.18, 0.76, 0.32, 1.0], roughness=0.30
                    ),
                )
            rest_height = 0.035
        else:
            raise ValueError(f"unknown object case: {object_case}")

        if object_case not in MESH_OBJECT_PATHS:
            self.object_rest_z = TABLE_TOP_Z + rest_height + 0.006
        builder.initial_pose = sapien.Pose(
            [OBJECT_SPAWN_XY[0], OBJECT_SPAWN_XY[1], self.object_rest_z]
        )
        actor = builder.build(f"object_{object_case}")
        body = actor.find_component_by_type(
            sapien.physx.PhysxRigidDynamicComponent
        )
        if body is None:
            raise RuntimeError(f"object {object_case} has no dynamic body")
        body.set_linear_damping(OBJECT_LINEAR_DAMPING)
        body.set_angular_damping(OBJECT_ANGULAR_DAMPING)
        if mesh_volume is not None and math.isfinite(mesh_volume):
            body.set_mass(max(min(mesh_volume * 100.0, 0.1), 0.01))
        return actor, body

    def set_object_case(self, object_case: str) -> None:
        if object_case not in OBJECT_CASES:
            raise ValueError(f"object_case must be one of {OBJECT_CASES}")
        if self.object is not None:
            self.scene.remove_entity(self.object)
        self.object_case = object_case
        self.object, self.object_body = self._build_object(object_case)

    def reset(self, object_case: str | None = None) -> None:
        if object_case is not None and object_case != self.object_case:
            self.set_object_case(object_case)
        self.robot.pose = ROBOT_ROOT_POSE
        self.robot.set_qpos(self.home_qpos)
        self.robot.set_qvel(np.zeros(self.robot.dof, dtype=np.float64))
        self.target_qpos = self.home_qpos.copy()
        self._set_all_drive_targets()
        self.disable_self_collision()
        self._reset_tcp_target()

        if self.randomize_object:
            object_xy = OBJECT_XY_CENTER + (
                self.rng.random(2) - 0.5
            ) * OBJECT_XY_SPAN
            yaw = float(self.rng.uniform(0.0, 2.0 * math.pi))
        else:
            object_xy = OBJECT_SPAWN_XY
            yaw = 0.0
        self.object.pose = sapien.Pose(
            [object_xy[0], object_xy[1], self.object_rest_z],
            [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)],
        )
        self.object_body.linear_velocity = np.zeros(3, dtype=np.float32)
        self.object_body.angular_velocity = np.zeros(3, dtype=np.float32)

    def command_tcp_delta(
        self,
        world_delta: Sequence[float],
        local_rotation_delta: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> bool:
        """Move the TCP in world XYZ and rotate about its current local axes.

        ``local_rotation_delta`` is a three-component rotation vector in
        radians.  Its direction is the EE-local rotation axis and its norm is
        the requested angle, so translation and rotation share one IK solve.
        """

        delta = np.asarray(world_delta, dtype=np.float64)
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise ValueError("world_delta must contain three finite values")
        rotation_delta = np.asarray(local_rotation_delta, dtype=np.float64)
        if rotation_delta.shape != (3,) or not np.isfinite(rotation_delta).all():
            raise ValueError(
                "local_rotation_delta must contain three finite values"
            )
        norm = float(np.linalg.norm(delta))
        rotation_norm = float(np.linalg.norm(rotation_delta))
        if norm == 0.0 and rotation_norm == 0.0:
            return True
        if norm > ARM_STEP_LIMIT_M:
            delta *= ARM_STEP_LIMIT_M / norm
        if rotation_norm > EE_ROTATION_STEP_LIMIT_RAD:
            rotation_delta *= EE_ROTATION_STEP_LIMIT_RAD / rotation_norm
            rotation_norm = EE_ROTATION_STEP_LIMIT_RAD

        world_target = self.robot.pose * self.tcp_target_local
        moved_world = sapien.Pose(
            np.asarray(world_target.p, dtype=np.float64) + delta,
            np.asarray(world_target.q, dtype=np.float64),
        )
        if rotation_norm > 0.0:
            moved_world = moved_world * sapien.Pose(
                q=axangle2quat(
                    rotation_delta / rotation_norm,
                    rotation_norm,
                )
            )
        candidate_local = self.robot.pose.inv() * moved_world
        solution, success, _ = self.pinocchio.compute_inverse_kinematics(
            self.tcp_link_index,
            candidate_local,
            initial_qpos=self.target_qpos,
            active_qmask=self.arm_active_mask,
            eps=1e-4,
            max_iterations=200,
            dt=0.1,
            damp=1e-4,
        )
        solution = np.asarray(solution, dtype=np.float64)
        if not bool(success) or solution.shape != (self.robot.dof,):
            return False
        if not np.isfinite(solution[self.arm_indices]).all():
            return False

        qlimits = np.asarray(self.robot.get_qlimits(), dtype=np.float64)
        candidate_qpos = self.target_qpos.copy()
        candidate_qpos[self.arm_indices] = np.clip(
            solution[self.arm_indices],
            qlimits[self.arm_indices, 0],
            qlimits[self.arm_indices, 1],
        )

        # Pinocchio may report success for a solution beyond the URDF limits.
        # Validate the clipped command before advancing the persistent TCP
        # target, otherwise repeated boundary commands create a large virtual
        # target and a noticeable dead zone when the user reverses direction.
        self.pinocchio.compute_forward_kinematics(candidate_qpos)
        achieved_local = self.pinocchio.get_link_pose(self.tcp_link_index)
        position_error = float(
            np.linalg.norm(
                np.asarray(achieved_local.p, dtype=np.float64)
                - np.asarray(candidate_local.p, dtype=np.float64)
            )
        )
        achieved_q = np.asarray(achieved_local.q, dtype=np.float64)
        requested_q = np.asarray(candidate_local.q, dtype=np.float64)
        achieved_q /= np.linalg.norm(achieved_q)
        requested_q /= np.linalg.norm(requested_q)
        quaternion_dot = float(
            np.clip(abs(np.dot(achieved_q, requested_q)), 0.0, 1.0)
        )
        orientation_error = 2.0 * math.acos(quaternion_dot)
        if (
            position_error > IK_POSITION_TOLERANCE_M
            or orientation_error > IK_ORIENTATION_TOLERANCE_RAD
        ):
            return False

        self.target_qpos[self.arm_indices] = candidate_qpos[self.arm_indices]
        for name in ARM_JOINT_NAMES:
            index = self.joint_index[name]
            self.joints_by_name[name].set_drive_target(
                float(self.target_qpos[index])
            )
        self.tcp_target_local = sapien.Pose(
            np.asarray(achieved_local.p, dtype=np.float64),
            np.asarray(achieved_local.q, dtype=np.float64),
        )
        return True

    def command_hand(self, hand_qpos: Sequence[float]) -> np.ndarray:
        target = capture_to_sim(
            hand_qpos,
            lower=self.hand_lower,
            upper=self.hand_upper,
        )
        self.target_qpos[self.hand_indices] = target
        for name, value in zip(HAND_JOINT_NAMES, target):
            self.joints_by_name[name].set_drive_target(float(value))
        return target

    def apply_action(
        self,
        action: Sequence[float],
        local_tcp_rotation_delta: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> bool:
        """Apply 24D translation/hand action plus an optional local EE rotation."""

        value = np.asarray(action, dtype=np.float64)
        if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
            raise ValueError(f"action must contain {ACTION_DIM} finite values")
        ik_success = self.command_tcp_delta(
            value[ARM_ACTION_SLICE],
            local_tcp_rotation_delta,
        )
        self.command_hand(value[HAND_ACTION_SLICE])
        return ik_success

    def step_physics(self, steps: int = PHYSICS_STEPS_PER_CONTROL) -> None:
        if steps < 0:
            raise ValueError("steps cannot be negative")
        for _ in range(steps):
            self.robot.set_qf(
                self.robot.compute_passive_force(
                    gravity=True,
                    coriolis_and_centrifugal=True,
                )
            )
            self.scene.step()

    @property
    def tcp_pose(self) -> sapien.Pose:
        return self.robot.links[self.tcp_link_index].entity_pose

    def metrics(self) -> TaskMetrics:
        contacted_fingers: set[str] = set()
        contact = False
        for record in self.scene.get_contacts():
            entities = (record.bodies[0].entity, record.bodies[1].entity)
            if self.object not in entities:
                continue
            other = entities[1] if entities[0] is self.object else entities[0]
            if other is None or other.name not in self.robot_link_names:
                continue
            pair_impulse = sum(
                (
                    np.asarray(point.impulse, dtype=np.float64)
                    for point in record.points
                ),
                np.zeros(3, dtype=np.float64),
            )
            pair_force = float(np.linalg.norm(pair_impulse)) / PHYSICS_DT
            if pair_force <= 1e-3:
                continue
            contact = True
            finger = _finger_from_link(other.name)
            if finger is not None:
                contacted_fingers.add(finger)

        object_p = np.asarray(self.object.pose.p, dtype=np.float64)
        lift = bool(object_p[2] > self.object_rest_z + 0.08)
        # The rendered target is a square with this half extent.
        xy_in_place = bool(
            np.all(np.abs(object_p[:2] - PLACE_CENTER_XY) <= PLACE_RADIUS_M)
        )
        near_table = abs(float(object_p[2]) - self.object_rest_z) <= 0.06
        in_place = bool(xy_in_place and near_table)
        speed = float(np.linalg.norm(self.object_body.linear_velocity))
        grasp = len(contacted_fingers) >= 2 and (lift or speed <= 0.2)
        success = bool(in_place and speed <= 0.05)
        return TaskMetrics(
            contact=contact,
            contact_fingers=len(contacted_fingers),
            grasp=grasp,
            lift=lift,
            in_place=in_place,
            success=success,
        )

    def _create_viewer(self) -> None:
        sapien.render.set_imgui_ini_filename("/tmp/mvhc-sapien-imgui.ini")
        self.viewer = self.scene.create_viewer()
        self.viewer.window.set_camera_parameters(
            near=0.01,
            far=100.0,
            fovy=math.radians(40.0),
        )
        self.apply_camera("front")
        self.scene.update_render()

    def apply_camera(self, camera_name: str) -> None:
        if self.viewer is None:
            return
        eye, target = CAMERAS[camera_name]
        self.viewer.set_camera_xyz(*eye.tolist())
        self.viewer.set_camera_rpy(
            *_camera_rpy_from_look_at(eye, target).tolist()
        )
        print(f"[CAMERA] {camera_name}", flush=True)

    def render(self) -> None:
        if self.viewer is None:
            return
        self.scene.update_render()
        self.viewer.render()

    def take_multiview_observations(self) -> dict[str, dict[str, np.ndarray]]:
        """Return RGB, camera-space position/depth, and segmentation per view."""

        if not self.render_enabled:
            raise RuntimeError("multiview observations require a render-enabled scene")
        self.scene.update_render()
        observations: dict[str, dict[str, np.ndarray]] = {}
        for name, camera in self.sensor_cameras.items():
            camera.take_picture()
            observations[name] = {
                "rgb": np.asarray(camera.get_picture("Color"), dtype=np.float32)[
                    ..., :3
                ].copy(),
                "position": np.asarray(
                    camera.get_picture("Position"), dtype=np.float32
                ).copy(),
                "segmentation": np.asarray(
                    camera.get_picture("Segmentation")
                ).copy(),
            }
        return observations

    def close(self) -> None:
        if self.viewer is not None and not self.viewer.closed:
            self.viewer.close()
        self.viewer = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pure-SAPIEN NERO + capture-native MMHand viewer. All robot and "
            "object assets are stored or generated inside teleop_maniskill."
        )
    )
    parser.add_argument("--listen", default="127.0.0.1:5557")
    parser.add_argument(
        "--arm-speed",
        type=float,
        default=0.12,
        help="Keyboard TCP speed in world metres per second.",
    )
    parser.add_argument(
        "--rotation-speed",
        type=float,
        default=DEFAULT_ROTATION_SPEED_DEG_S,
        help="Keyboard EE-local roll/pitch/yaw speed in degrees per second.",
    )
    parser.add_argument(
        "--object-case",
        choices=OBJECT_CASES,
        default="bowl",
        help="Project-local OBJ case or one of three procedural objects.",
    )
    parser.add_argument(
        "--object-mesh-path",
        type=Path,
        default=None,
        help=(
            "Override the selected OBJ case. It must live inside this project; "
            "an object root containing mesh/simplified.obj is also accepted."
        ),
    )
    parser.add_argument(
        "--object-scale",
        type=float,
        default=DEFAULT_OBJECT_SCALE,
        help="Uniform scale for the selected/local OBJ mesh.",
    )
    parser.add_argument(
        "--udp-timeout",
        type=float,
        default=0.25,
        help="Latest packet age in seconds before the hand changes to HOLD.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fixed-object",
        action="store_true",
        help="Use the deterministic [0.55, 0.35] object pose instead of reset randomization.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Maximum 20 Hz control steps; 0 runs until closed or interrupted.",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.arm_speed) or args.arm_speed < 0:
        parser.error("--arm-speed must be finite and non-negative")
    if not math.isfinite(args.rotation_speed) or args.rotation_speed < 0:
        parser.error("--rotation-speed must be finite and non-negative")
    if not math.isfinite(args.udp_timeout) or args.udp_timeout <= 0:
        parser.error("--udp-timeout must be finite and positive")
    if not math.isfinite(args.object_scale) or args.object_scale <= 0:
        parser.error("--object-scale must be finite and positive")
    if args.object_mesh_path is not None and args.object_case not in MESH_OBJECT_PATHS:
        parser.error(
            "--object-mesh-path requires --object-case "
            + "|".join(MESH_OBJECT_PATHS)
        )
    if args.max_steps < 0:
        parser.error("--max-steps cannot be negative")
    return args


def _arm_delta(window: Any, arm_speed: float) -> np.ndarray:
    direction = np.asarray(
        [
            float(_key_down(window, "up", "arrow_up"))
            - float(_key_down(window, "down", "arrow_down")),
            float(_key_down(window, "left", "arrow_left"))
            - float(_key_down(window, "right", "arrow_right")),
            float(_key_down(window, "u")) - float(_key_down(window, "j")),
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(direction))
    if norm > 0:
        direction /= norm
    delta = direction * arm_speed * CONTROL_DT
    delta_norm = float(np.linalg.norm(delta))
    if delta_norm > ARM_STEP_LIMIT_M:
        delta *= ARM_STEP_LIMIT_M / delta_norm
    return delta


def _ee_rotation_delta(window: Any, rotation_speed_deg_s: float) -> np.ndarray:
    """Return an EE-local [roll, pitch, yaw] rotation vector in radians."""

    direction = np.asarray(
        [
            float(_key_down(window, "i")) - float(_key_down(window, "k")),
            float(_key_down(window, "o")) - float(_key_down(window, "l")),
            float(_key_down(window, "p")) - float(_key_down(window, "m")),
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(direction))
    if norm > 0.0:
        direction /= norm
    delta = direction * math.radians(rotation_speed_deg_s) * CONTROL_DT
    delta_norm = float(np.linalg.norm(delta))
    if delta_norm > EE_ROTATION_STEP_LIMIT_RAD:
        delta *= EE_ROTATION_STEP_LIMIT_RAD / delta_norm
    return delta


def _packet_age(packet: RetargetPacket | None, now: float) -> float | None:
    if packet is None:
        return None
    return max(0.0, now - float(packet.sent_monotonic))


def _tracking_state(
    packet: RetargetPacket | None,
    now: float,
    timeout: float,
) -> str:
    if packet is None:
        return "WAIT"
    age = _packet_age(packet, now)
    left = (packet.handedness or "").strip().lower() == "left"
    if not packet.valid or not left or age is None or age > timeout:
        return "HOLD"
    return "LIVE"


def _print_status(
    simulation: StandalonePickPlace,
    packet: RetargetPacket | None,
    timeout: float,
    paused: bool,
    step: int,
) -> None:
    now = time.monotonic()
    age = _packet_age(packet, now)
    age_text = "-" if age is None else f"{age * 1000.0:.0f}ms"
    phase = "-" if packet is None else packet.phase
    tcp = np.asarray(simulation.tcp_pose.p, dtype=np.float64)
    actual_qpos = np.asarray(simulation.robot.get_qpos(), dtype=np.float64)
    hand_error = np.abs(
        simulation.target_qpos[simulation.hand_indices]
        - actual_qpos[simulation.hand_indices]
    )
    max_hand_error_deg = math.degrees(float(np.max(hand_error)))
    max_thumb_error_deg = math.degrees(float(np.max(hand_error[16:21])))
    metrics = simulation.metrics()
    print(
        f"[STEP {step:06d}] object={simulation.object_case} "
        f"tracking={_tracking_state(packet, now, timeout)} "
        f"control={'PAUSED' if paused else 'RUN'} age={age_text} phase={phase} "
        f"tcp=({tcp[0]:+.3f},{tcp[1]:+.3f},{tcp[2]:+.3f}) "
        f"hand_err={max_hand_error_deg:.1f}deg "
        f"thumb_err={max_thumb_error_deg:.1f}deg "
        f"contact={int(metrics.contact)} fingers={metrics.contact_fingers} "
        f"grasp={int(metrics.grasp)} lift={int(metrics.lift)} "
        f"in_place={int(metrics.in_place)} success={int(metrics.success)}",
        flush=True,
    )


def _print_help() -> None:
    print(
        "\n[CONTROLS]\n"
        "  Up / Down     : TCP world X +/-\n"
        "  Left / Right  : TCP world Y +/-\n"
        "  U / J         : TCP world Z +/-\n"
        "  I / K         : EE-local roll  +/- (local X)\n"
        "  O / L         : EE-local pitch +/- (local Y)\n"
        "  P / M         : EE-local yaw   +/- (local Z)\n"
        "  1 / 2 / 3     : front / left-rear / right-rear camera\n"
        "  4 / 5 / 6 / 7 : bowl / cup / can / box OBJ (and reset)\n"
        "  8 / 9 / 0     : cube / cylinder / sphere primitive (and reset)\n"
        "  Space         : pause / resume control\n"
        "  N             : open hand until the next valid capture frame\n"
        "  R             : reset robot and current object\n"
        "  H             : print this help\n"
        "  Q / Esc       : quit\n",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receiver: LatestUdpRetargetReceiver | None = None
    simulation: StandalonePickPlace | None = None
    try:
        receiver = LatestUdpRetargetReceiver(args.listen)
        print(f"[UDP] listening on {args.listen}", flush=True)
        simulation = StandalonePickPlace(
            object_case=args.object_case,
            object_mesh_path=args.object_mesh_path,
            object_scale=args.object_scale,
            headless=args.headless,
            seed=args.seed,
            randomize_object=not args.fixed_object,
        )
        print(
            f"[PHYSICS] pure SAPIEN {sapien.__version__}; "
            f"robot=28DOF local full-mesh URDF; self_collision_shapes="
            f"{simulation.self_collision_shape_count}; "
            f"solver={SOLVER_POSITION_ITERATIONS}/{SOLVER_VELOCITY_ITERATIONS}; "
            f"contact_offset={CONTACT_OFFSET_M * 1000.0:.1f}mm; "
            f"object_damping={OBJECT_LINEAR_DAMPING:.1f}/"
            f"{OBJECT_ANGULAR_DAMPING:.1f}",
            flush=True,
        )
        print(
            f"[CONTACT] MMHand friction static={HAND_STATIC_FRICTION:.1f} "
            f"dynamic={HAND_DYNAMIC_FRICTION:.1f}; "
            f"shapes={simulation.hand_friction_shape_count}; "
            "object/table friction=0.3/0.3",
            flush=True,
        )
        print(
            f"[INFO] sim={SIM_FREQ}Hz control={CONTROL_FREQ}Hz "
            f"arm_speed={args.arm_speed:.3f}m/s "
            f"rotation_speed={args.rotation_speed:.1f}deg/s "
            f"hand_rate={HAND_RATE_LIMIT_RAD_S:.1f}rad/s",
            flush=True,
        )
        if simulation.viewer is not None:
            _print_help()

        current_hand = simulation.open_hand.copy()
        target_hand = current_hand.copy()
        latest_packet: RetargetPacket | None = None
        neutral_override = False
        paused = False
        step = 0
        next_tick = time.monotonic()
        next_status = next_tick
        last_ik_warning = float("-inf")

        while args.max_steps <= 0 or step < args.max_steps:
            if simulation.viewer is not None and simulation.viewer.closed:
                break

            now = time.monotonic()
            packet = receiver.poll()
            if packet is not None:
                latest_packet = packet
                age = _packet_age(packet, now)
                is_left = (packet.handedness or "").strip().lower() == "left"
                if (
                    packet.valid
                    and packet.q_rad is not None
                    and is_left
                    and age is not None
                    and age <= args.udp_timeout
                ):
                    try:
                        target_hand = capture_to_sim(
                            packet.q_rad,
                            lower=simulation.hand_lower,
                            upper=simulation.hand_upper,
                        )
                        neutral_override = False
                    except ValueError as exc:
                        print(
                            f"[WARN] rejected hand target at seq={packet.seq}: {exc}",
                            flush=True,
                        )
            if (
                not neutral_override
                and _tracking_state(latest_packet, now, args.udp_timeout) == "HOLD"
            ):
                target_hand = current_hand.copy()

            arm_delta = np.zeros(3, dtype=np.float64)
            ee_rotation_delta = np.zeros(3, dtype=np.float64)
            reset_case: str | None = None
            if simulation.viewer is not None:
                window = simulation.viewer.window
                if _key_press(window, "q", "escape", "esc"):
                    break
                if _key_press(window, "h"):
                    _print_help()
                if _key_press(window, "space", " "):
                    paused = not paused
                    print(
                        f"[CONTROL] {'paused' if paused else 'running'}",
                        flush=True,
                    )
                if _key_press(window, "n"):
                    neutral_override = True
                    target_hand = simulation.open_hand.copy()
                    print(
                        "[HAND] open target selected; next valid frame resumes tracking",
                        flush=True,
                    )
                if _key_press(window, "r"):
                    reset_case = simulation.object_case
                for key, camera_name in zip(("1", "2", "3"), CAMERAS):
                    if _key_press(window, key):
                        simulation.apply_camera(camera_name)
                for key, object_case in OBJECT_HOTKEYS.items():
                    if _key_press(window, key):
                        reset_case = object_case
                if not paused:
                    arm_delta = _arm_delta(window, args.arm_speed)
                    ee_rotation_delta = _ee_rotation_delta(
                        window,
                        args.rotation_speed,
                    )

            if reset_case is not None:
                simulation.reset(reset_case)
                current_hand = simulation.open_hand.copy()
                target_hand = current_hand.copy()
                latest_packet = None
                neutral_override = False
                print(
                    f"[RESET] robot and object={simulation.object_case}",
                    flush=True,
                )

            if not paused:
                current_hand = rate_limit(
                    current_hand,
                    target_hand,
                    max_rate=HAND_RATE_LIMIT_RAD_S,
                    dt=CONTROL_DT,
                )
                action = np.empty(ACTION_DIM, dtype=np.float64)
                action[ARM_ACTION_SLICE] = arm_delta
                action[HAND_ACTION_SLICE] = current_hand
                if not simulation.apply_action(
                    action,
                    local_tcp_rotation_delta=ee_rotation_delta,
                ):
                    now = time.monotonic()
                    if now - last_ik_warning >= 1.0:
                        print(
                            "[IK] target rejected; keeping previous arm target",
                            flush=True,
                        )
                        last_ik_warning = now
                simulation.step_physics()
                step += 1

            simulation.render()
            now = time.monotonic()
            if now >= next_status:
                _print_status(
                    simulation,
                    latest_packet,
                    args.udp_timeout,
                    paused,
                    step,
                )
                next_status = now + 0.2

            next_tick += CONTROL_DT
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -CONTROL_DT:
                next_tick = time.monotonic()
        return 0
    except KeyboardInterrupt:
        print("\n[INFO] interrupted", flush=True)
        return 0
    finally:
        if receiver is not None:
            receiver.close()
        if simulation is not None:
            simulation.close()


if __name__ == "__main__":
    raise SystemExit(main())
