"""Fast headless tests for the project-local, pure-SAPIEN teleoperation scene.

Run with the environment stored in this checkout:

    .venv/bin/python -m unittest -v teleop_maniskill.test_sim_pick_place

These tests intentionally create neither a GUI nor a UDP socket.  They verify
the full-mesh robot's physics, action contract, local objects, and task directly.
"""

from __future__ import annotations

from contextlib import redirect_stderr
import io
import math
from pathlib import Path
import unittest

import numpy as np
import sapien

from . import sim_pick_place as sim
from .prepare_full_fidelity_urdf import (
    ARM_ACTIVE_JOINT_NAMES as ARM_JOINT_NAMES,
    DEFAULT_OUTPUT_URDF as FULL_ROBOT_URDF,
    HAND_ACTIVE_JOINT_NAMES as HAND_JOINT_NAMES,
    validate_urdf,
)
from .teleop_protocol import JOINT_NAMES, capture_to_sim


def _qpos(simulation: sim.StandalonePickPlace) -> np.ndarray:
    return np.asarray(simulation.robot.get_qpos(), dtype=np.float64).copy()


def _render_component_count(simulation: sim.StandalonePickPlace) -> int:
    return sum(
        "Render" in type(component).__name__
        for entity in simulation.scene.get_entities()
        for component in entity.components
    )


def _quaternion_distance(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    return 2.0 * np.arccos(np.clip(abs(np.dot(first, second)), 0.0, 1.0))


class _KeyWindow:
    def __init__(self, *keys: str):
        self.keys = set(keys)

    def key_down(self, key: str) -> bool:
        return key in self.keys


class TestStandalonePickPlace(unittest.TestCase):
    """Exercise one shared physics scene; every test resets its state."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.simulation = sim.StandalonePickPlace(
            object_case="cube",
            headless=True,
            randomize_object=False,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.simulation.close()

    def setUp(self) -> None:
        self.simulation.reset("cube")

    def test_task_contact_filter_counts_only_the_five_fingertips(self) -> None:
        for link_name, finger in sim.FINGERTIP_LINK_TO_FINGER.items():
            with self.subTest(link=link_name):
                self.assertEqual(sim._finger_from_link(link_name), finger)
        self.assertIsNone(
            sim._finger_from_link("capture_hand__finger_1_distal_phalanx_1")
        )
        self.assertIsNone(sim._finger_from_link("capture_hand__palm_1"))

    def test_original_table_goal_and_normal_camera_profile_are_preserved(self) -> None:
        self.assertEqual(sim.TABLE_TOP_Z, 0.714)
        np.testing.assert_allclose(sim.OBJECT_XY_CENTER, [0.56, 0.3533])
        np.testing.assert_allclose(sim.OBJECT_XY_SPAN, [0.20, 0.40])
        np.testing.assert_allclose(sim.PLACE_CENTER_XY, [0.20, -0.05])
        self.assertEqual(sim.PLACE_RADIUS_M, 0.09)
        expected_eyes = {
            "front": [1.3158333333, 0.2511, 1.606],
            "left-rear": [-0.4691666667, 1.0161, 1.606],
            "right-rear": [-0.4691666667, -0.5139, 1.606],
        }
        for name, expected in expected_eyes.items():
            with self.subTest(camera=name):
                eye, target = sim.CAMERAS[name]
                np.testing.assert_allclose(eye, expected, atol=1e-9)
                np.testing.assert_allclose(
                    target, [0.2533333333, 0.2511, 0.994], atol=1e-9
                )
        self.assertEqual(
            sim.OBJECT_HOTKEYS,
            {
                "4": "bowl",
                "5": "cup",
                "6": "can",
                "7": "box",
                "8": "cube",
                "9": "cylinder",
                "0": "sphere",
            },
        )

    def test_headless_robot_and_explicit_24d_action_layout(self) -> None:
        simulation = self.simulation
        active_names = [joint.name for joint in simulation.robot.active_joints]

        self.assertTrue(simulation.headless)
        self.assertFalse(simulation.render_enabled)
        self.assertIsNone(simulation.viewer)
        self.assertEqual(_render_component_count(simulation), 0)
        with self.assertRaises(RuntimeError):
            _ = simulation.scene.render_system

        self.assertEqual(sim.SIM_FREQ, 100)
        self.assertEqual(sim.CONTROL_FREQ, 20)
        self.assertEqual(sim.PHYSICS_STEPS_PER_CONTROL, 5)
        self.assertAlmostEqual(
            sim.HAND_RATE_LIMIT_RAD_S * sim.CONTROL_DT,
            0.30,
        )
        self.assertEqual(sim.DEFAULT_ROTATION_SPEED_DEG_S, 45.0)
        self.assertAlmostEqual(
            math.degrees(sim.EE_ROTATION_STEP_LIMIT_RAD),
            10.0,
        )
        self.assertEqual(
            simulation.robot.get_solver_position_iterations(),
            sim.SOLVER_POSITION_ITERATIONS,
        )
        self.assertEqual(
            simulation.robot.get_solver_velocity_iterations(),
            sim.SOLVER_VELOCITY_ITERATIONS,
        )
        self.assertAlmostEqual(
            sapien.physx.get_shape_config().contact_offset,
            sim.CONTACT_OFFSET_M,
        )
        self.assertEqual(sim.ACTION_DIM, 24)
        self.assertEqual(
            (sim.ARM_ACTION_SLICE.start, sim.ARM_ACTION_SLICE.stop),
            (0, 3),
        )
        self.assertEqual(
            (sim.HAND_ACTION_SLICE.start, sim.HAND_ACTION_SLICE.stop),
            (3, 24),
        )

        self.assertEqual(simulation.robot.dof, 28)
        self.assertEqual(len(active_names), 28)
        self.assertEqual(
            set(active_names),
            set(ARM_JOINT_NAMES) | set(HAND_JOINT_NAMES),
        )
        self.assertEqual(tuple(HAND_JOINT_NAMES), tuple(JOINT_NAMES))
        self.assertEqual(simulation.arm_indices.shape, (7,))
        self.assertEqual(simulation.hand_indices.shape, (21,))

        # SAPIEN's active-qpos order is intentionally not capture order.  This
        # guards against accidentally replacing the name-based mapping with a
        # contiguous qpos slice.
        raw_hand_order = tuple(
            active_names[index]
            for index in range(len(active_names))
            if index not in set(simulation.arm_indices.tolist())
        )
        self.assertNotEqual(raw_hand_order, tuple(HAND_JOINT_NAMES))
        self.assertEqual(
            [active_names[index] for index in simulation.hand_indices],
            list(HAND_JOINT_NAMES),
        )
        for joint in simulation.robot.active_joints:
            with self.subTest(joint_friction=joint.name):
                self.assertAlmostEqual(
                    joint.get_friction(),
                    sim.ACTIVE_JOINT_FRICTION,
                )

        urdf_path = FULL_ROBOT_URDF.resolve()
        self.assertTrue(urdf_path.is_relative_to(sim.PROJECT_ROOT.resolve()))
        self.assertTrue(urdf_path.is_file())
        self.assertEqual(validate_urdf(urdf_path)["mesh_references"], 57)

    def test_mmhand_collision_shapes_use_high_friction_material(self) -> None:
        simulation = self.simulation
        hand_shape_count = 0

        for link in simulation.robot.links:
            for shape in link.collision_shapes:
                material = shape.get_physical_material()
                if link.name.startswith("capture_hand__"):
                    hand_shape_count += 1
                    self.assertAlmostEqual(
                        material.get_static_friction(),
                        sim.HAND_STATIC_FRICTION,
                    )
                    self.assertAlmostEqual(
                        material.get_dynamic_friction(),
                        sim.HAND_DYNAMIC_FRICTION,
                    )

        self.assertGreater(hand_shape_count, 0)
        self.assertEqual(
            simulation.hand_friction_shape_count,
            hand_shape_count,
        )

        object_shape = simulation.object_body.get_collision_shapes()[0]
        object_material = object_shape.get_physical_material()
        self.assertAlmostEqual(object_material.get_static_friction(), 0.3)
        self.assertAlmostEqual(object_material.get_dynamic_friction(), 0.3)

    def test_synthetic_capture_target_is_applied_by_hand_joint_name(self) -> None:
        simulation = self.simulation
        fractions = np.linspace(0.15, 0.85, 21, dtype=np.float64)
        capture_target = (
            simulation.hand_lower
            + fractions * (simulation.hand_upper - simulation.hand_lower)
        )
        expected = capture_to_sim(
            capture_target,
            lower=simulation.hand_lower,
            upper=simulation.hand_upper,
        )
        arm_before = simulation.target_qpos[simulation.arm_indices].copy()

        action = np.zeros(sim.ACTION_DIM, dtype=np.float64)
        action[sim.HAND_ACTION_SLICE] = capture_target
        self.assertTrue(simulation.apply_action(action))

        np.testing.assert_allclose(
            simulation.target_qpos[simulation.arm_indices],
            arm_before,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            simulation.target_qpos[simulation.hand_indices],
            expected,
            rtol=0.0,
            atol=1e-12,
        )
        for name, value in zip(HAND_JOINT_NAMES, expected):
            with self.subTest(joint=name):
                raw_index = simulation.joint_index[name]
                self.assertEqual(
                    simulation.robot.active_joints[raw_index].name,
                    name,
                )
                drive_target = np.asarray(
                    simulation.joints_by_name[name].get_drive_target(),
                    dtype=np.float64,
                )
                np.testing.assert_allclose(
                    drive_target,
                    [value],
                    rtol=0.0,
                    atol=1e-7,
                )

        for bad_action in (
            np.zeros(sim.ACTION_DIM - 1),
            np.full(sim.ACTION_DIM, np.nan),
        ):
            with self.subTest(shape=bad_action.shape):
                with self.assertRaises(ValueError):
                    simulation.apply_action(bad_action)
        for bad_rotation in (np.zeros(2), [0.0, 0.0, float("nan")]):
            with self.subTest(rotation=bad_rotation):
                with self.assertRaises(ValueError):
                    simulation.apply_action(
                        action,
                        local_tcp_rotation_delta=bad_rotation,
                    )

    def test_six_world_xyz_ik_commands_move_in_requested_direction(self) -> None:
        simulation = self.simulation
        command_size = 0.006

        for axis, axis_name in enumerate("XYZ"):
            for direction in (-1.0, 1.0):
                with self.subTest(axis=axis_name, direction=direction):
                    simulation.reset("cube")
                    before = np.asarray(
                        simulation.tcp_pose.p,
                        dtype=np.float64,
                    ).copy()
                    delta = np.zeros(3, dtype=np.float64)
                    delta[axis] = direction * command_size

                    self.assertTrue(simulation.command_tcp_delta(delta))
                    target_world = simulation.robot.pose * simulation.tcp_target_local
                    target_displacement = (
                        np.asarray(target_world.p, dtype=np.float64) - before
                    )
                    self.assertGreater(
                        direction * target_displacement[axis],
                        command_size - 2e-4,
                    )

                    simulation.step_physics(20)
                    after = np.asarray(
                        simulation.tcp_pose.p,
                        dtype=np.float64,
                    )
                    self.assertGreater(
                        direction * (after[axis] - before[axis]),
                        1e-4,
                        f"world {axis_name} command moved in the wrong direction",
                    )

    def test_six_local_ee_rotation_commands_move_in_requested_direction(self) -> None:
        simulation = self.simulation
        command_angle = math.radians(2.0)

        for axis, axis_name in enumerate("XYZ"):
            for direction in (-1.0, 1.0):
                with self.subTest(axis=axis_name, direction=direction):
                    simulation.reset("cube")
                    before = simulation.robot.pose * simulation.tcp_target_local
                    actual_before_q = np.asarray(
                        simulation.tcp_pose.q,
                        dtype=np.float64,
                    ).copy()
                    rotation_delta = np.zeros(3, dtype=np.float64)
                    rotation_delta[axis] = direction * command_angle

                    self.assertTrue(
                        simulation.command_tcp_delta(
                            np.zeros(3, dtype=np.float64),
                            rotation_delta,
                        )
                    )
                    after = simulation.robot.pose * simulation.tcp_target_local
                    expected = before * sapien.Pose(
                        q=[
                            math.cos(command_angle / 2.0),
                            *(direction * math.sin(command_angle / 2.0)
                              * np.eye(3, dtype=np.float64)[axis]),
                        ]
                    )

                    np.testing.assert_allclose(
                        after.p,
                        before.p,
                        rtol=0.0,
                        atol=sim.IK_POSITION_TOLERANCE_M,
                    )
                    self.assertLess(
                        _quaternion_distance(after.q, expected.q),
                        sim.IK_ORIENTATION_TOLERANCE_RAD,
                    )
                    self.assertGreater(
                        _quaternion_distance(after.q, before.q),
                        math.radians(1.0),
                    )

                    simulation.step_physics(20)
                    actual_after_q = np.asarray(
                        simulation.tcp_pose.q,
                        dtype=np.float64,
                    )
                    self.assertGreater(
                        _quaternion_distance(actual_after_q, actual_before_q),
                        math.radians(0.25),
                    )
                    self.assertLess(
                        _quaternion_distance(actual_after_q, after.q),
                        _quaternion_distance(actual_before_q, after.q),
                    )

    def test_rotation_keys_generate_local_axis_rotation_vectors(self) -> None:
        step = math.radians(sim.DEFAULT_ROTATION_SPEED_DEG_S) * sim.CONTROL_DT
        mappings = {
            "i": (0, 1.0),
            "k": (0, -1.0),
            "o": (1, 1.0),
            "l": (1, -1.0),
            "p": (2, 1.0),
            "m": (2, -1.0),
        }
        for key, (axis, direction) in mappings.items():
            with self.subTest(key=key):
                expected = np.zeros(3, dtype=np.float64)
                expected[axis] = direction * step
                np.testing.assert_allclose(
                    sim._ee_rotation_delta(
                        _KeyWindow(key),
                        sim.DEFAULT_ROTATION_SPEED_DEG_S,
                    ),
                    expected,
                    rtol=0.0,
                    atol=1e-12,
                )

        diagonal = sim._ee_rotation_delta(
            _KeyWindow("i", "o", "p"),
            sim.DEFAULT_ROTATION_SPEED_DEG_S,
        )
        self.assertAlmostEqual(float(np.linalg.norm(diagonal)), step)

    def test_fixed_hand_pose_pushes_object_without_large_thumb_deflection(self) -> None:
        simulation = self.simulation
        simulation.reset("cube")
        thumb_link_name = next(
            name
            for name, finger in sim.FINGERTIP_LINK_TO_FINGER.items()
            if finger == "thumb"
        )
        thumb_link = next(
            link for link in simulation.robot.links if link.name == thumb_link_name
        )
        thumb_indices = simulation.hand_indices[16:21]
        initial_thumb_q = _qpos(simulation)[thumb_indices]
        object_start = (
            np.asarray(thumb_link.entity_pose.p, dtype=np.float64)
            + np.asarray([0.065, 0.0, 0.0], dtype=np.float64)
        )

        simulation.object_body.disable_gravity = True
        try:
            simulation.object.pose = sapien.Pose(object_start)
            simulation.object_body.linear_velocity = np.zeros(3, dtype=np.float32)
            simulation.object_body.angular_velocity = np.zeros(3, dtype=np.float32)
            max_thumb_error = 0.0

            for _ in range(12):
                self.assertTrue(simulation.command_tcp_delta([0.006, 0.0, 0.0]))
                simulation.command_hand(simulation.open_hand)
                simulation.step_physics()
                thumb_q = _qpos(simulation)[thumb_indices]
                max_thumb_error = max(
                    max_thumb_error,
                    float(np.max(np.abs(thumb_q - initial_thumb_q))),
                )

            object_delta = (
                np.asarray(simulation.object.pose.p, dtype=np.float64)
                - object_start
            )
            self.assertGreater(object_delta[0], 0.035)
            self.assertLess(max_thumb_error, math.radians(2.0))
            for name, expected in zip(HAND_JOINT_NAMES[16:21], simulation.open_hand[16:21]):
                self.assertAlmostEqual(
                    float(simulation.joints_by_name[name].get_drive_target()[0]),
                    float(expected),
                    places=6,
                )
        finally:
            simulation.object_body.disable_gravity = False
            simulation.reset("cube")

    def test_reaching_joint_limit_does_not_create_virtual_tcp_dead_zone(self) -> None:
        simulation = self.simulation
        simulation.reset("cube")
        downward = np.asarray([0.0, 0.0, -0.006], dtype=np.float64)

        rejected = False
        for _ in range(120):
            if not simulation.command_tcp_delta(downward):
                rejected = True
                break
            simulation.pinocchio.compute_forward_kinematics(
                simulation.target_qpos
            )
            achieved = simulation.pinocchio.get_link_pose(
                simulation.tcp_link_index
            )
            error = np.linalg.norm(
                np.asarray(achieved.p) - np.asarray(simulation.tcp_target_local.p)
            )
            self.assertLess(error, sim.IK_POSITION_TOLERANCE_M)

        self.assertTrue(rejected, "the repeated downward command never hit a limit")
        before_reverse = np.asarray(
            (simulation.robot.pose * simulation.tcp_target_local).p,
            dtype=np.float64,
        ).copy()
        self.assertTrue(simulation.command_tcp_delta(-downward))
        after_reverse = np.asarray(
            (simulation.robot.pose * simulation.tcp_target_local).p,
            dtype=np.float64,
        )
        self.assertGreater(after_reverse[2] - before_reverse[2], 0.004)

        limits = np.asarray(simulation.robot.get_qlimits(), dtype=np.float64)
        commanded_arm = simulation.target_qpos[simulation.arm_indices]
        np.testing.assert_array_less(
            limits[simulation.arm_indices, 0] - 1e-9,
            commanded_arm,
        )
        np.testing.assert_array_less(
            commanded_arm,
            limits[simulation.arm_indices, 1] + 1e-9,
        )

    def test_seven_local_objects_have_mass_and_reset_cleanly(self) -> None:
        simulation = self.simulation
        expected_shape_types = {
            "bowl": "PhysxCollisionShapeConvexMesh",
            "cup": "PhysxCollisionShapeConvexMesh",
            "can": "PhysxCollisionShapeConvexMesh",
            "box": "PhysxCollisionShapeConvexMesh",
            "cube": "PhysxCollisionShapeBox",
            "cylinder": "PhysxCollisionShapeCylinder",
            "sphere": "PhysxCollisionShapeSphere",
        }
        previous_object = None

        for object_case in sim.OBJECT_CASES:
            with self.subTest(object_case=object_case):
                simulation.set_object_case(object_case)
                if previous_object is not None:
                    self.assertIsNot(simulation.object, previous_object)
                    self.assertNotIn(
                        previous_object,
                        simulation.scene.get_entities(),
                    )
                previous_object = simulation.object

                self.assertEqual(simulation.object_case, object_case)
                self.assertEqual(simulation.object.name, f"object_{object_case}")
                self.assertGreater(float(simulation.object_body.mass), 0.0)
                self.assertTrue(
                    np.isfinite(float(simulation.object_body.mass))
                )
                self.assertAlmostEqual(
                    simulation.object_body.get_linear_damping(),
                    sim.OBJECT_LINEAR_DAMPING,
                )
                self.assertAlmostEqual(
                    simulation.object_body.get_angular_damping(),
                    sim.OBJECT_ANGULAR_DAMPING,
                )
                shapes = simulation.object_body.get_collision_shapes()
                self.assertEqual(len(shapes), 1)
                self.assertEqual(
                    type(shapes[0]).__name__,
                    expected_shape_types[object_case],
                )
                if object_case in sim.MESH_OBJECT_PATHS:
                    self.assertEqual(
                        simulation.object_mesh_path,
                        sim.MESH_OBJECT_PATHS[object_case].resolve(),
                    )

                simulation.object.pose = sapien.Pose([0.1, -0.2, 1.4])
                simulation.object_body.linear_velocity = np.asarray(
                    [1.0, -2.0, 3.0],
                    dtype=np.float32,
                )
                changed_qpos = simulation.home_qpos.copy()
                changed_qpos[simulation.arm_indices[0]] += 0.2
                simulation.robot.set_qpos(changed_qpos)
                simulation.reset()

                expected_object_p = np.asarray(
                    [
                        sim.OBJECT_SPAWN_XY[0],
                        sim.OBJECT_SPAWN_XY[1],
                        simulation.object_rest_z,
                    ],
                    dtype=np.float64,
                )
                np.testing.assert_allclose(
                    simulation.object.pose.p,
                    expected_object_p,
                    rtol=0.0,
                    atol=1e-7,
                )
                np.testing.assert_allclose(
                    simulation.object_body.linear_velocity,
                    np.zeros(3),
                    rtol=0.0,
                    atol=1e-7,
                )
                np.testing.assert_allclose(
                    _qpos(simulation),
                    simulation.home_qpos,
                    rtol=0.0,
                    atol=1e-7,
                )

    def test_self_collision_mask_and_100_tick_hold_are_stable(self) -> None:
        simulation = self.simulation

        for cycle in ("initial", "after-reset"):
            with self.subTest(cycle=cycle):
                simulation.reset("cube")
                shape_count = 0
                for link in simulation.robot.links:
                    for shape in link.collision_shapes:
                        shape_count += 1
                        groups = shape.get_collision_groups()
                        self.assertNotEqual(groups[2] & (1 << 30), 0)
                self.assertGreater(shape_count, 0)
                self.assertEqual(
                    simulation.self_collision_shape_count,
                    shape_count,
                )

                before = _qpos(simulation)
                simulation.step_physics(100)
                drift = float(np.max(np.abs(_qpos(simulation) - before)))
                self.assertLess(
                    drift,
                    1e-3,
                    f"{cycle}: max qpos drift was {drift:.9g} rad",
                )

    def test_metrics_cover_initial_lift_and_placed_states(self) -> None:
        simulation = self.simulation
        initial = simulation.metrics()
        self.assertFalse(initial.contact)
        self.assertEqual(initial.contact_fingers, 0)
        self.assertFalse(initial.grasp)
        self.assertFalse(initial.lift)
        self.assertFalse(initial.in_place)
        self.assertFalse(initial.success)

        simulation.object.pose = sapien.Pose(
            [
                sim.OBJECT_SPAWN_XY[0],
                sim.OBJECT_SPAWN_XY[1],
                simulation.object_rest_z + 0.081,
            ]
        )
        lifted = simulation.metrics()
        self.assertTrue(lifted.lift)
        self.assertFalse(lifted.in_place)
        self.assertFalse(lifted.success)

        simulation.object.pose = sapien.Pose(
            [
                sim.PLACE_CENTER_XY[0],
                sim.PLACE_CENTER_XY[1],
                simulation.object_rest_z,
            ]
        )
        simulation.object_body.linear_velocity = np.zeros(3, dtype=np.float32)
        placed = simulation.metrics()
        self.assertFalse(placed.lift)
        self.assertTrue(placed.in_place)
        self.assertTrue(placed.success)

        simulation.object.pose = sapien.Pose(
            [
                sim.PLACE_CENTER_XY[0] + 0.09,
                sim.PLACE_CENTER_XY[1] + 0.09,
                simulation.object_rest_z,
            ]
        )
        self.assertTrue(
            simulation.metrics().in_place,
            "the rendered square corner must match the square success region",
        )


class TestCommandLine(unittest.TestCase):
    def test_mesh_resolver_accepts_local_object_root_and_rejects_external(self) -> None:
        cup_root = sim.MESH_OBJECT_PATHS["cup"].parents[1]
        self.assertEqual(
            sim._project_object_mesh(cup_root),
            sim.MESH_OBJECT_PATHS["cup"].resolve(),
        )
        with self.assertRaisesRegex(ValueError, "must be copied inside"):
            sim._project_object_mesh("/usr/bin/python3")

    def test_standalone_cli_defaults_and_case_selection(self) -> None:
        defaults = sim.parse_args(["--headless"])
        self.assertEqual(defaults.listen, "127.0.0.1:5557")
        self.assertEqual(defaults.object_case, "bowl")
        self.assertIsNone(defaults.object_mesh_path)
        self.assertEqual(defaults.object_scale, 0.08)
        self.assertEqual(defaults.seed, 0)
        self.assertFalse(defaults.fixed_object)
        self.assertEqual(defaults.arm_speed, 0.12)
        self.assertEqual(defaults.rotation_speed, 45.0)
        self.assertEqual(defaults.udp_timeout, 0.25)
        self.assertTrue(defaults.headless)
        self.assertEqual(
            set(vars(defaults)),
            {
                "listen",
                "arm_speed",
                "rotation_speed",
                "object_case",
                "object_mesh_path",
                "object_scale",
                "udp_timeout",
                "headless",
                "seed",
                "fixed_object",
                "max_steps",
            },
        )

        selected = sim.parse_args(
            [
                "--headless",
                "--object-case",
                "cup",
                "--arm-speed",
                "0.2",
                "--rotation-speed",
                "60",
                "--max-steps",
                "3",
            ]
        )
        self.assertEqual(selected.object_case, "cup")
        self.assertEqual(selected.arm_speed, 0.2)
        self.assertEqual(selected.rotation_speed, 60.0)
        self.assertEqual(selected.max_steps, 3)

    def test_invalid_cli_values_are_rejected(self) -> None:
        for argv in (
            ["--object-case", "mesh"],
            ["--object-case", "cube", "--object-mesh-path", "object.obj"],
            ["--arm-speed", "-0.1"],
            ["--rotation-speed", "-0.1"],
            ["--udp-timeout", "0"],
            ["--object-scale", "0"],
            ["--max-steps", "-1"],
        ):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    sim.parse_args(argv)


if __name__ == "__main__":
    unittest.main()
