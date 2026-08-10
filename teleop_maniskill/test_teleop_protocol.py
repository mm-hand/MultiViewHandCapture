import json
import contextlib
import io
import os
from pathlib import Path
import socket
import time
import unittest

import numpy as np

from config import MCP_AA_NEUTRAL_DEG, ROBOT_JOINT_NAMES
from teleop_maniskill.run_sim_teleop import (
    DEFAULT_PYTHON,
    PROJECT_ROOT,
    build_commands,
    child_environment,
    parse_args,
    require_project_path,
    wait_for_processes,
)
from teleop_maniskill.teleop_protocol import (
    JOINT_NAMES,
    SCHEMA,
    LatestUdpRetargetReceiver,
    UdpRetargetSender,
    capture_to_sim,
    parse_endpoint,
    rate_limit,
)


class LauncherTests(unittest.TestCase):
    class _Process:
        def __init__(self, *codes):
            self.codes = list(codes)
            self.poll_count = 0

        def poll(self):
            self.poll_count += 1
            if len(self.codes) > 1:
                return self.codes.pop(0)
            return self.codes[0]

    def test_default_interpreters_use_the_same_project_virtualenv(self):
        args = parse_args([])
        self.assertEqual(args.capture_python, DEFAULT_PYTHON)
        self.assertEqual(args.sim_python, DEFAULT_PYTHON)

        checked = require_project_path(
            args.capture_python, "capture Python", executable=True
        )
        self.assertEqual(checked, DEFAULT_PYTHON.absolute())
        self.assertIn(".venv", checked.parts)

    def test_child_environment_removes_external_python_search_paths(self):
        old_pythonpath = os.environ.get("PYTHONPATH")
        old_pythonhome = os.environ.get("PYTHONHOME")
        try:
            os.environ["PYTHONPATH"] = "/outside/checkout"
            os.environ["PYTHONHOME"] = "/outside/python"
            environment = child_environment("/tmp/mvhc-test-matplotlib")
        finally:
            if old_pythonpath is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = old_pythonpath
            if old_pythonhome is None:
                os.environ.pop("PYTHONHOME", None)
            else:
                os.environ["PYTHONHOME"] = old_pythonhome

        self.assertNotIn("PYTHONPATH", environment)
        self.assertNotIn("PYTHONHOME", environment)
        self.assertEqual(environment["PYTHONNOUSERSITE"], "1")

    def test_commands_only_use_project_files_and_supported_sim_options(self):
        args = parse_args(
            [
                "--listen",
                "127.0.0.1:6001",
                "--arm-speed",
                "0.2",
                "--rotation-speed",
                "60",
                "--object-case",
                "sphere",
                "--object-scale",
                "0.08",
                "--seed",
                "0",
                "--max-steps",
                "12",
                "--headless",
            ]
        )
        sim_command, capture_command = build_commands(args)

        for command in (sim_command, capture_command):
            for path_text in command[:2]:
                Path(path_text).relative_to(PROJECT_ROOT)

        self.assertEqual(
            sim_command[2:],
            [
                "--listen",
                "127.0.0.1:6001",
                "--arm-speed",
                "0.2",
                "--rotation-speed",
                "60.0",
                "--object-case",
                "sphere",
                "--object-scale",
                "0.08",
                "--seed",
                "0",
                "--max-steps",
                "12",
                "--headless",
            ],
        )
        self.assertEqual(
            capture_command[2:],
            ["--mode", "retarget", "--udp", "127.0.0.1:6001"],
        )
        joined = " ".join(sim_command + capture_command).lower()
        self.assertNotIn("--object-mesh-path", joined)
        self.assertIn("--object-scale 0.08", joined)
        self.assertNotIn("pythonpath", joined)

    def test_external_interpreter_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "inside the project root"):
            require_project_path(
                Path("/usr/bin/python3"), "simulation Python", executable=True
            )

    def test_object_case_is_restricted_to_builtin_cases(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--object-case", "external-mesh"])
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--object-case",
                        "cube",
                        "--object-mesh-path",
                        "teleop_maniskill/assets/objects/bowl",
                    ]
                )

    def test_custom_object_mesh_is_forwarded_only_from_this_project(self):
        object_root = PROJECT_ROOT / "teleop_maniskill/assets/objects/bowl"
        args = parse_args(
            ["--object-mesh-path", str(object_root), "--fixed-object"]
        )
        sim_command, _ = build_commands(args)
        self.assertIn("--object-mesh-path", sim_command)
        self.assertIn(str(object_root), sim_command)
        self.assertIn("--fixed-object", sim_command)

        external = parse_args(["--object-mesh-path", "/usr/bin/python3"])
        with self.assertRaisesRegex(ValueError, "inside the project root"):
            build_commands(external)

    def test_capture_failure_keeps_simulation_alive_until_it_exits(self):
        simulation = self._Process(None, None, 0)
        capture = self._Process(3)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = wait_for_processes(
                {"simulation": simulation, "capture": capture},
                poll_interval=0.0,
            )

        self.assertEqual(code, 0)
        self.assertEqual(capture.poll_count, 1)
        self.assertGreaterEqual(simulation.poll_count, 3)
        self.assertIn("capture exited with code 3", output.getvalue())
        self.assertIn("simulation remains open in HOLD", output.getvalue())


def unused_endpoint():
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return f"127.0.0.1:{port}"


def packet(seq=1, **changes):
    payload = {
        "schema": SCHEMA,
        "seq": seq,
        "sent_monotonic": 12.5,
        "valid": True,
        "phase": "GESTURE TRACKING",
        "handedness": "Left",
        "joint_names": list(JOINT_NAMES),
        "q_rad": np.linspace(0.0, 1.0, 21).tolist(),
        "quality": {"reprojection_error": 1.25},
    }
    payload.update(changes)
    return payload


def poll_until(receiver, timeout=0.25):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = receiver.poll()
        if result is not None:
            return result
        time.sleep(0.002)
    return None


class EndpointTests(unittest.TestCase):
    def test_parse_endpoint(self):
        self.assertEqual(parse_endpoint("127.0.0.1:5557"), ("127.0.0.1", 5557))
        for endpoint in ("localhost", ":5557", "localhost:", "localhost:0", "localhost:65536"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                parse_endpoint(endpoint)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.endpoint = unused_endpoint()
        self.receiver = LatestUdpRetargetReceiver(self.endpoint)
        self.sender = UdpRetargetSender(self.endpoint)

    def tearDown(self):
        self.sender.close()
        self.receiver.close()

    def test_valid_and_invalid_tracking_round_trip(self):
        q = np.linspace(-0.2, 0.8, 21)
        self.assertTrue(
            self.sender.send(
                7,
                123.25,
                True,
                "GESTURE TRACKING",
                "Left",
                ROBOT_JOINT_NAMES,
                q,
                {"rms": 0.01},
            )
        )
        received = poll_until(self.receiver)
        self.assertIsNotNone(received)
        self.assertEqual(received.schema, SCHEMA)
        self.assertEqual(received.seq, 7)
        self.assertEqual(received.joint_names, tuple(ROBOT_JOINT_NAMES))
        np.testing.assert_allclose(received.q_rad, q)
        self.assertEqual(received.quality, {"rms": 0.01})

        self.assertTrue(
            self.sender.send(
                8,
                123.3,
                False,
                "CALIBRATING",
                "Left",
                ROBOT_JOINT_NAMES,
                None,
                {"rejected_reason": "calibration"},
            )
        )
        received = poll_until(self.receiver)
        self.assertFalse(received.valid)
        self.assertIsNone(received.q_rad)

        self.assertTrue(
            self.sender.send(
                9,
                123.4,
                False,
                "GESTURE TRACKING",
                "Right",
                ROBOT_JOINT_NAMES,
                None,
                {"rejected_reason": "right_hand"},
            )
        )
        received = poll_until(self.receiver)
        self.assertFalse(received.valid)
        self.assertEqual(received.handedness, "Right")

    def test_sender_rejects_invalid_contract_and_oversize_json(self):
        base = (1, 1.0, True, "GESTURE TRACKING", "Left")
        self.assertFalse(
            self.sender.send(*base, ROBOT_JOINT_NAMES, np.zeros(20), {})
        )
        self.assertFalse(
            self.sender.send(*base, ("wrong",) * 21, np.zeros(21), {})
        )
        bad = np.zeros(21)
        bad[2] = np.nan
        self.assertFalse(self.sender.send(*base, ROBOT_JOINT_NAMES, bad, {}))
        self.assertFalse(
            self.sender.send(
                1,
                1.0,
                False,
                "WAITING",
                None,
                ROBOT_JOINT_NAMES,
                np.zeros(21),
                {},
            )
        )
        self.assertFalse(
            self.sender.send(
                1,
                1.0,
                True,
                "GESTURE TRACKING",
                "Right",
                ROBOT_JOINT_NAMES,
                np.zeros(21),
                {},
            )
        )
        self.assertFalse(
            self.sender.send(
                1,
                1.0,
                True,
                "CALIBRATION",
                "Left",
                ROBOT_JOINT_NAMES,
                np.zeros(21),
                {},
            )
        )
        self.assertFalse(
            self.sender.send(
                1,
                1.0,
                False,
                "WAITING",
                None,
                ROBOT_JOINT_NAMES,
                None,
                {"detail": "x" * 9000},
            )
        )

    def test_receiver_drops_bad_schema_shape_nonfinite_and_names(self):
        raw = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        host_port = parse_endpoint(self.endpoint)
        malformed = (
            packet(1, schema="wrong"),
            packet(2, q_rad=[0.0] * 20),
            packet(3, q_rad=[0.0] * 20 + [float("nan")]),
            packet(4, joint_names=["wrong"] * 21),
        )
        try:
            for payload in malformed:
                raw.sendto(json.dumps(payload).encode(), host_port)
            raw.sendto(json.dumps(packet(5)).encode(), host_port)
        finally:
            raw.close()
        received = poll_until(self.receiver)
        self.assertIsNotNone(received)
        self.assertEqual(received.seq, 5)

    def test_poll_drains_and_discards_out_of_order_sequences(self):
        raw = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        host_port = parse_endpoint(self.endpoint)
        try:
            for seq in (4, 10, 7):
                raw.sendto(json.dumps(packet(seq)).encode(), host_port)
            received = poll_until(self.receiver)
            self.assertEqual(received.seq, 10)

            for seq in (9, 10):
                raw.sendto(json.dumps(packet(seq)).encode(), host_port)
            time.sleep(0.01)
            self.assertIsNone(self.receiver.poll())

            raw.sendto(json.dumps(packet(11)).encode(), host_port)
            received = poll_until(self.receiver)
            self.assertEqual(received.seq, 11)
        finally:
            raw.close()

    def test_empty_poll_is_nonblocking_for_timeout_hold(self):
        started = time.monotonic()
        self.assertIsNone(self.receiver.poll())
        self.assertLess(time.monotonic() - started, 0.05)


class MappingTests(unittest.TestCase):
    def test_neutral_keeps_same_urdf_coordinates(self):
        q = np.zeros(21)
        q[[0, 4, 8, 12]] = np.radians(MCP_AA_NEUTRAL_DEG)
        np.testing.assert_allclose(capture_to_sim(q), q, atol=1e-15)

    def test_same_urdf_order_and_signs_are_identity(self):
        q = np.arange(1.0, 22.0)
        np.testing.assert_allclose(capture_to_sim(q), q)

    def test_open_fist_and_single_index_keep_capture_directions(self):
        open_capture = np.zeros(21)
        open_capture[[0, 4, 8, 12]] = np.radians(MCP_AA_NEUTRAL_DEG)
        fist_capture = open_capture.copy()
        fist_capture[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]] = 0.5
        fist_capture[20] = 1.0

        open_target = capture_to_sim(open_capture)
        fist_target = capture_to_sim(fist_capture)
        np.testing.assert_allclose(open_target, open_capture, atol=1e-15)
        np.testing.assert_allclose(fist_target, fist_capture, atol=1e-15)

        index_only = open_capture.copy()
        index_only[13:16] = (0.1, 0.2, 0.3)
        index_target = capture_to_sim(index_only)
        np.testing.assert_allclose(index_target, index_only, atol=1e-15)

    def test_controller_limit_clipping_keeps_joint_order(self):
        q = np.linspace(-1.0, 1.0, 21)
        lower = np.linspace(-0.75, -0.25, 21)
        upper = np.linspace(0.25, 0.75, 21)
        clipped = capture_to_sim(q, lower=lower, upper=upper)
        np.testing.assert_allclose(clipped, np.clip(q, lower, upper))

    def test_aa_and_dip_clipping_uses_radians(self):
        q = np.zeros(21)
        q[12] = np.radians(90)
        q[15] = np.radians(-90)
        lower = np.full(21, -0.2)
        upper = np.full(21, 0.1)
        mapped = capture_to_sim(q, lower=lower, upper=upper)
        self.assertAlmostEqual(mapped[12], 0.1)
        self.assertAlmostEqual(mapped[15], -0.2)

        degrees_mistaken_for_radians = np.zeros(21)
        degrees_mistaken_for_radians[[0, 4, 8, 12]] = MCP_AA_NEUTRAL_DEG
        self.assertGreater(
            np.max(np.abs(capture_to_sim(degrees_mistaken_for_radians))),
            20.0,
        )

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            capture_to_sim(np.zeros(20))
        with self.assertRaises(ValueError):
            capture_to_sim(np.r_[np.zeros(20), np.inf])

    def test_rate_limit(self):
        current = np.asarray((0.0, 1.0, -1.0))
        target = np.asarray((1.0, -1.0, -1.1))
        np.testing.assert_allclose(
            rate_limit(current, target, max_rate=3.0, dt=0.05),
            (0.15, 0.85, -1.1),
        )
        np.testing.assert_allclose(
            rate_limit(current, target, max_rate=3.0, dt=0.0),
            current,
        )
        with self.assertRaises(ValueError):
            rate_limit(current, target, max_rate=-1.0, dt=0.05)


if __name__ == "__main__":
    unittest.main()
