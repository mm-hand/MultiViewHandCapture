import math
import time

import numpy as np
import sapien

from config import ROBOT_JOINT_NAMES, URDF_PATH


_DT = 0.01
_HOME = np.array((0.0, 0.0, 0.20))
_CONTACT_OFFSET = 0.003
_SURFACE_FRICTION = 0.3
_HAND_STATIC_FRICTION = 2.0
_HAND_DYNAMIC_FRICTION = 1.0
_OBJECT_LINEAR_DAMPING = 2.0
_OBJECT_ANGULAR_DAMPING = 2.0
_FINGERTIPS = {
    "finger_1_fingertip_1",
    "finger_2_fingertip_1",
    "finger_3_fingertip_1",
    "finger_4_fingertip_1",
    "mmhand_thumb_1_finger_7_fingertip_1",
}


def _key(window, method, *names):
    for name in names:
        try:
            if getattr(window, method)(name):
                return True
        except Exception:
            pass
    return False


class GraspSimulation:
    """Latest-MMHand SAPIEN scene consuming J00-J20 radians."""

    def __init__(self, headless=False):
        self.headless = headless
        self.rng = np.random.default_rng()

        scene_config = sapien.physx.get_scene_config()
        scene_config.enable_pcm = True
        scene_config.enable_tgs = True
        scene_config.enable_friction_every_iteration = True
        sapien.physx.set_scene_config(scene_config)

        body_config = sapien.physx.get_body_config()
        body_config.sleep_threshold = 0.005
        body_config.solver_position_iterations = 25
        body_config.solver_velocity_iterations = 4
        sapien.physx.set_body_config(body_config)

        shape_config = sapien.physx.get_shape_config()
        shape_config.contact_offset = _CONTACT_OFFSET
        shape_config.rest_offset = 0.0
        sapien.physx.set_shape_config(shape_config)
        sapien.physx.set_default_material(
            _SURFACE_FRICTION, _SURFACE_FRICTION, 0.0
        )

        if headless:
            self.scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
        else:
            sapien.render.set_viewer_shader_dir("default")
            sapien.render.set_camera_shader_dir("default")
            self.scene = sapien.Scene()
        self.scene.set_timestep(_DT)
        self.hand_material = self.scene.create_physical_material(
            _HAND_STATIC_FRICTION, _HAND_DYNAMIC_FRICTION, 0.0
        )
        self.object_material = self.scene.create_physical_material(
            _SURFACE_FRICTION, _SURFACE_FRICTION, 0.0
        )
        self.scene.add_ground(0.0, render=not headless, material=self.object_material)
        self.hand = self._load_hand()
        self.object = None
        self._reset()
        self.viewer = None
        if not headless:
            self.scene.set_ambient_light([0.35] * 3)
            self.scene.add_directional_light([1, 1, -1], [1, 1, 1], shadow=True)
            self.viewer = self.scene.create_viewer()
            self.viewer.window.set_camera_parameters(near=0.01, far=10.0, fovy=0.8)
            self.viewer.set_camera_xyz(0.55, -0.45, 0.40)
            self.viewer.set_camera_rpy(0.0, -0.47, -2.36)
        self.last_time = time.monotonic()
        self.accumulator = 0.0
        self.last_contacts = None

    def _load_hand(self):
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        build_link = loader._build_link

        # SAPIEN rejects the URDF's semidefinite inertia tensors.
        def positive_inertia(link, builder):
            if link.inertial is not None:
                inertia = link.inertial.inertia
                if not np.array_equal(inertia, np.zeros((3, 3))):
                    minimum = np.linalg.eigvalsh(inertia)[0]
                    if minimum <= 0:
                        link.inertial.inertia = inertia + np.eye(3) * (1e-9 - minimum)
            build_link(link, builder)

        loader._build_link = positive_inertia
        builder = loader.load_file_as_articulation_builder(str(URDF_PATH))
        if builder is None:
            raise RuntimeError(f"Cannot load MMHand URDF: {URDF_PATH}")
        if self.headless:
            for link in builder.link_builders:
                link.visual_records.clear()
        hand = builder.build()
        joints = list(hand.active_joints)
        if len(joints) != 21 or {joint.name for joint in joints} != set(ROBOT_JOINT_NAMES):
            raise RuntimeError("MMHand joints do not match ROBOT_JOINT_NAMES")
        self.joints = {joint.name: joint for joint in joints}
        index = {joint.name: i for i, joint in enumerate(joints)}
        limits = np.asarray(hand.get_qlimits(), float)
        self.lower = np.array([limits[index[name], 0] for name in ROBOT_JOINT_NAMES])
        self.upper = np.array([limits[index[name], 1] for name in ROBOT_JOINT_NAMES])
        for joint in joints:
            joint.set_friction(0.0)
            joint.set_drive_properties(
                stiffness=1000.0, damping=100.0, force_limit=1e10, mode="force"
            )
        for link in hand.links:
            for shape in link.collision_shapes:
                shape.set_physical_material(self.hand_material)
                groups = list(shape.get_collision_groups())
                groups[2] |= 1 << 30
                shape.set_collision_groups(groups)
        hand.pose = sapien.Pose(_HOME)
        q = np.clip(np.zeros(21), limits[:, 0], limits[:, 1])
        hand.set_qpos(q)
        hand.set_qvel(np.zeros(21))
        for joint, value in zip(joints, q):
            joint.set_drive_target(float(value))
        return hand

    def _new_cylinder(self):
        self.radius = float(self.rng.uniform(0.025, 0.035))
        self.height = float(self.rng.uniform(0.080, 0.120))
        local = sapien.Pose(q=[math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0])
        builder = self.scene.create_actor_builder()
        builder.add_cylinder_collision(
            pose=local, radius=self.radius, half_length=self.height / 2,
            material=self.object_material, density=300.0,
        )
        if not self.headless:
            material = sapien.render.RenderMaterial()
            material.set_base_color(np.array([0.85, 0.25, 0.15, 1.0], np.float32))
            builder.add_cylinder_visual(
                pose=local, radius=self.radius, half_length=self.height / 2,
                material=material,
            )
        cylinder = builder.build("cylinder")
        cylinder.pose = sapien.Pose([0.10, 0.0, self.height / 2])
        self.object_body = cylinder.find_component_by_type(
            sapien.physx.PhysxRigidDynamicComponent
        )
        if self.object_body is None:
            raise RuntimeError("Cylinder has no dynamic PhysX body")
        self.object_body.set_linear_damping(_OBJECT_LINEAR_DAMPING)
        self.object_body.set_angular_damping(_OBJECT_ANGULAR_DAMPING)
        print(f"cylinder radius={self.radius * 1000:.1f}mm height={self.height * 1000:.1f}mm")
        return cylinder

    def _reset(self):
        self.hand.pose = sapien.Pose(_HOME)
        if self.object is not None:
            self.scene.remove_entity(self.object)
        self.object = self._new_cylinder()

    def _move(self, elapsed):
        window = self.viewer.window
        move = np.array((
            _key(window, "key_down", "up", "arrow_up") - _key(window, "key_down", "down", "arrow_down"),
            _key(window, "key_down", "left", "arrow_left") - _key(window, "key_down", "right", "arrow_right"),
            _key(window, "key_down", "u") - _key(window, "key_down", "j"),
        ), float)
        turn = np.array((
            _key(window, "key_down", "i") - _key(window, "key_down", "k"),
            _key(window, "key_down", "o") - _key(window, "key_down", "l"),
            _key(window, "key_down", "p") - _key(window, "key_down", "m"),
        ), float)
        pose = self.hand.pose
        norm = np.linalg.norm(move)
        position = np.asarray(pose.p) + (move / norm * 0.12 * elapsed if norm else 0)
        norm = np.linalg.norm(turn)
        if norm:
            axis, angle = turn / norm, math.radians(45) * elapsed
            pose = sapien.Pose(position, pose.q) * sapien.Pose(
                q=np.r_[math.cos(angle / 2), axis * math.sin(angle / 2)]
            )
        else:
            pose = sapien.Pose(position, pose.q)
        self.hand.pose = pose

    def _contacts(self):
        fingers = set()
        for contact in self.scene.get_contacts():
            entities = (contact.bodies[0].entity, contact.bodies[1].entity)
            if self.object not in entities:
                continue
            other = entities[1] if entities[0] is self.object else entities[0]
            if other is not None and other.name in _FINGERTIPS and any(
                np.linalg.norm(point.impulse) > 1e-8 for point in contact.points
            ):
                fingers.add(other.name)
        return len(fingers)

    def update(self, robot_joints=None):
        if self.viewer is not None and self.viewer.closed:
            return False
        if robot_joints is not None:
            q = np.asarray(robot_joints, float)
            if q.shape != (21,) or not np.isfinite(q).all():
                raise ValueError("robot_joints must contain 21 finite radians")
            for name, value in zip(ROBOT_JOINT_NAMES, np.clip(q, self.lower, self.upper)):
                self.joints[name].set_drive_target(float(value))
        now = time.monotonic()
        elapsed = min(now - self.last_time, 0.05)
        self.last_time = now
        if self.viewer is not None:
            window = self.viewer.window
            if _key(window, "key_press", "q", "escape", "esc"):
                return False
            if _key(window, "key_press", "r"):
                self._reset()
            self._move(elapsed)
        self.accumulator = min(self.accumulator + elapsed, 5 * _DT)
        while self.accumulator >= _DT:
            self.hand.set_qf(self.hand.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=True
            ))
            self.scene.step()
            self.accumulator -= _DT
        contacts = self._contacts()
        if contacts != self.last_contacts:
            print(f"contacts={contacts}")
            self.last_contacts = contacts
        if self.viewer is not None:
            self.scene.update_render()
            self.viewer.render()
        return True

    def close(self):
        if self.viewer is not None and not self.viewer.closed:
            self.viewer.close()
        self.viewer = None
