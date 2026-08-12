import math
import time

import numpy as np
import sapien

from config import ROBOT_JOINT_NAMES, URDF_PATH


_DT = 0.01
_HAND_CONTROL_DT = 0.05
_HAND_RATE_LIMIT = 6.0
_HAND_MAX_DELTA = _HAND_RATE_LIMIT * _HAND_CONTROL_DT
_ROOT_LINEAR_SPEED = 0.12
_ROOT_ANGULAR_SPEED = math.radians(45.0)
_ROOT_POSITION_GAIN = 20.0
_ROOT_ROTATION_GAIN = 20.0
_ROOT_MAX_LINEAR_SPEED = 0.24
_ROOT_MAX_ANGULAR_SPEED = math.radians(90.0)
_HOME = np.array((0.0, 0.0, 0.20))
_CONTACT_OFFSET = 0.003
_SURFACE_FRICTION = 0.3
_HAND_STATIC_FRICTION = 0.8
_HAND_DYNAMIC_FRICTION = 0.6
_CONTACT_LOG_DT = 1.0
_OBJECT_LINEAR_DAMPING = 2.0
_OBJECT_ANGULAR_DAMPING = 2.0
_JOINT_FRICTION = 0.0
_DRIVE_STIFFNESS = 1000.0
_DRIVE_DAMPING = 100.0
_DRIVE_FORCE_LIMIT = 1e10
_SELF_COLLISION_BIT = 30
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


def _quat_multiply(one, two):
    """Multiply scalar-first quaternions."""
    w1, x1, y1, z1 = np.asarray(one, float)
    w2, x2, y2, z2 = np.asarray(two, float)
    return np.array((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ))


def _quat_rotate(quaternion, vector):
    quaternion = np.asarray(quaternion, float)
    vector_quaternion = np.r_[0.0, np.asarray(vector, float)]
    conjugate = quaternion * np.array((1.0, -1.0, -1.0, -1.0))
    return _quat_multiply(
        _quat_multiply(quaternion, vector_quaternion), conjugate
    )[1:]


def _rotation_error(target, current):
    """Return the shortest world-frame rotation vector current -> target."""
    current_conjugate = np.asarray(current, float) * np.array(
        (1.0, -1.0, -1.0, -1.0)
    )
    error = _quat_multiply(target, current_conjugate)
    error /= max(np.linalg.norm(error), 1e-12)
    if error[0] < 0:
        error = -error
    length = np.linalg.norm(error[1:])
    if length < 1e-12:
        return np.zeros(3)
    angle = 2 * math.atan2(length, max(error[0], 0.0))
    return error[1:] / length * angle


def _clip_norm(vector, maximum):
    vector = np.asarray(vector, float)
    length = np.linalg.norm(vector)
    return vector if length <= maximum else vector * (maximum / length)


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
        # Contact bodies can be exposed through fresh Python wrappers, so
        # object membership against ``self.hand.links`` is not reliable.
        self.hand_link_names = {link.name for link in self.hand.links}
        self.object = None
        self.hand_control_accumulator = 0.0
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
        self.last_contact_log = 0.0

    def _load_hand(self):
        loader = self.scene.create_urdf_loader()
        # The root must remain dynamic so PhysX sees its commanded linear and
        # angular velocities. Moving a fixed root by assigning ``hand.pose``
        # teleports its collision shapes with zero physical velocity.
        loader.fix_root_link = False
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
        self.joint_index = {joint.name: i for i, joint in enumerate(joints)}
        limits = np.asarray(hand.get_qlimits(), float)
        self.lower = np.array([
            limits[self.joint_index[name], 0] for name in ROBOT_JOINT_NAMES
        ])
        self.upper = np.array([
            limits[self.joint_index[name], 1] for name in ROBOT_JOINT_NAMES
        ])
        self.neutral_qpos = np.clip(np.zeros(21), limits[:, 0], limits[:, 1])
        self.neutral_command = np.array([
            self.neutral_qpos[self.joint_index[name]] for name in ROBOT_JOINT_NAMES
        ])
        return hand

    def _configure_hand_physics(self):
        """Reapply the contact and drive settings that make grasping stiff."""
        for link in self.hand.links:
            # The free root is explicitly velocity-controlled. Gravity remains
            # enabled for the cylinder and ground, but not for the robot hand.
            link.disable_gravity = True
        for joint in self.hand.active_joints:
            joint.set_friction(_JOINT_FRICTION)
            joint.set_drive_properties(
                stiffness=_DRIVE_STIFFNESS,
                damping=_DRIVE_DAMPING,
                force_limit=_DRIVE_FORCE_LIMIT,
                mode="force",
            )
            joint.set_drive_velocity_target(0.0)
        shape_count = 0
        for link in self.hand.links:
            for shape in link.collision_shapes:
                shape.set_physical_material(self.hand_material)
                groups = list(shape.get_collision_groups())
                groups[2] |= 1 << _SELF_COLLISION_BIT
                shape.set_collision_groups(groups)
                shape_count += 1
        self.hand_collision_shape_count = shape_count

    def _apply_hand_command(self):
        for name, value in zip(ROBOT_JOINT_NAMES, self.commanded_q):
            self.joints[name].set_drive_target(float(value))

    def command_hand(self, robot_joints):
        """Store a finite, limit-clipped desired J00-J20 pose."""
        q = np.asarray(robot_joints, float)
        if q.shape != (21,) or not np.isfinite(q).all():
            raise ValueError("robot_joints must contain 21 finite radians")
        self.desired_q = np.clip(q, self.lower, self.upper)
        return self.desired_q.copy()

    def _advance_hand_command(self):
        """Advance the drive target by at most 0.30 rad per 20 Hz tick."""
        delta = np.clip(
            self.desired_q - self.commanded_q,
            -_HAND_MAX_DELTA,
            _HAND_MAX_DELTA,
        )
        self.commanded_q = np.clip(
            self.commanded_q + delta,
            self.lower,
            self.upper,
        )
        self._apply_hand_command()
        return self.commanded_q.copy()

    def _reset_hand(self):
        self.target_pose = sapien.Pose(_HOME)
        self.target_linear_velocity = np.zeros(3)
        self.target_angular_velocity = np.zeros(3)
        self.hand.pose = self.target_pose
        self.hand.set_root_linear_velocity(np.zeros(3))
        self.hand.set_root_angular_velocity(np.zeros(3))
        self.hand.set_qpos(self.neutral_qpos.copy())
        self.hand.set_qvel(np.zeros(21))
        self.hand.set_qf(np.zeros(21))
        self.desired_q = self.neutral_command.copy()
        self.commanded_q = self.neutral_command.copy()
        self.hand_control_accumulator = 0.0
        self._configure_hand_physics()
        self._apply_hand_command()

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
        print(
            f"cylinder radius={self.radius * 1000:.1f}mm "
            f"height={self.height * 1000:.1f}mm",
            flush=True,
        )
        return cylinder

    def _reset(self):
        self._reset_hand()
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
        pose = self.target_pose
        norm = np.linalg.norm(move)
        self.target_linear_velocity = (
            move / norm * _ROOT_LINEAR_SPEED if norm else np.zeros(3)
        )
        position = np.asarray(pose.p) + self.target_linear_velocity * elapsed
        norm = np.linalg.norm(turn)
        if norm:
            axis, angle = turn / norm, _ROOT_ANGULAR_SPEED * elapsed
            self.target_angular_velocity = (
                _quat_rotate(pose.q, axis) * _ROOT_ANGULAR_SPEED
            )
            pose = sapien.Pose(position, pose.q) * sapien.Pose(
                q=np.r_[math.cos(angle / 2), axis * math.sin(angle / 2)]
            )
        else:
            self.target_angular_velocity = np.zeros(3)
            pose = sapien.Pose(position, pose.q)
        self.target_pose = pose

    def _apply_root_velocity(self):
        pose = self.hand.pose
        linear = (
            self.target_linear_velocity
            + _ROOT_POSITION_GAIN * (np.asarray(self.target_pose.p) - np.asarray(pose.p))
        )
        angular = (
            self.target_angular_velocity
            + _ROOT_ROTATION_GAIN * _rotation_error(self.target_pose.q, pose.q)
        )
        linear = _clip_norm(linear, _ROOT_MAX_LINEAR_SPEED)
        angular = _clip_norm(angular, _ROOT_MAX_ANGULAR_SPEED)
        self.hand.set_root_linear_velocity(linear)
        self.hand.set_root_angular_velocity(angular)
        return linear, angular

    def _contacts(self):
        fingers = set()
        point_count = 0
        for contact in self.scene.get_contacts():
            entities = (contact.bodies[0].entity, contact.bodies[1].entity)
            if self.object not in entities:
                continue
            other = entities[1] if entities[0] is self.object else entities[0]
            if other is None or other.name not in self.hand_link_names:
                continue
            active_points = sum(
                1 for point in contact.points
                if np.linalg.norm(point.impulse) > 1e-8
            )
            point_count += active_points
            if other.name in _FINGERTIPS and active_points:
                fingers.add(other.name)
        return len(fingers), point_count

    def update(self, robot_joints=None):
        if self.viewer is not None and self.viewer.closed:
            return False
        if robot_joints is not None:
            self.command_hand(robot_joints)
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
        self.hand_control_accumulator = min(
            self.hand_control_accumulator + elapsed,
            _HAND_CONTROL_DT,
        )
        if self.hand_control_accumulator >= _HAND_CONTROL_DT - 1e-12:
            self._advance_hand_command()
            self.hand_control_accumulator = max(
                0.0,
                self.hand_control_accumulator - _HAND_CONTROL_DT,
            )
        self.accumulator = min(self.accumulator + elapsed, 5 * _DT)
        while self.accumulator >= _DT:
            self.step_physics()
            self.accumulator -= _DT
        contact_state = self._contacts()
        if (contact_state != self.last_contacts
                or now - self.last_contact_log >= _CONTACT_LOG_DT):
            fingertip_count, point_count = contact_state
            print(
                f"contacts={fingertip_count} contact_points={point_count}",
                flush=True,
            )
            self.last_contacts = contact_state
            self.last_contact_log = now
        if self.viewer is not None:
            self.scene.update_render()
            self.viewer.render()
        return True

    def step_physics(self, steps=1):
        if not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        for _ in range(steps):
            self._apply_root_velocity()
            self.hand.set_qf(self.hand.compute_passive_force(
                gravity=False,
                coriolis_and_centrifugal=True,
            ))
            self.scene.step()

    def close(self):
        if self.viewer is not None and not self.viewer.closed:
            self.viewer.close()
        self.viewer = None
