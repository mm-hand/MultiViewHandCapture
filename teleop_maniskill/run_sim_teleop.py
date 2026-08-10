#!/usr/bin/env python3
"""Launch the self-contained capture and SAPIEN teleoperation processes."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
from collections.abc import Sequence


TELEOP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TELEOP_DIR.parent
DEFAULT_PYTHON = PROJECT_ROOT / ".venv/bin/python"
MESH_OBJECT_CASES = ("bowl", "cup", "can", "box")
OBJECT_CASES = (*MESH_OBJECT_CASES, "cube", "cylinder", "sphere")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start the MultiViewHandCapture retarget stream and the NERO + "
            "MMHand SAPIEN viewer from this project and its local .venv."
        )
    )
    parser.add_argument("--listen", default="127.0.0.1:5557")
    parser.add_argument("--arm-speed", type=float, default=0.12)
    parser.add_argument(
        "--rotation-speed",
        type=float,
        default=45.0,
        help="EE-local roll/pitch/yaw speed in degrees per second.",
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
        help="Project-local OBJ or object root overriding the selected OBJ case.",
    )
    parser.add_argument(
        "--object-scale",
        type=float,
        default=0.08,
        help="Uniform scale for the selected/local OBJ mesh.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed-object", action="store_true")
    parser.add_argument(
        "--capture-python",
        type=Path,
        default=DEFAULT_PYTHON,
        help="Project-local Python used for track.py.",
    )
    parser.add_argument(
        "--sim-python",
        type=Path,
        default=DEFAULT_PYTHON,
        help="Project-local Python used for the SAPIEN simulator.",
    )
    parser.add_argument("--startup-delay", type=float, default=0.75)
    parser.add_argument("--headless", action="store_true", help="Run the simulator without a GUI.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Simulator control steps; 0 runs until the viewer is closed.",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.arm_speed) or args.arm_speed < 0.0:
        parser.error("--arm-speed must be finite and non-negative")
    if not math.isfinite(args.rotation_speed) or args.rotation_speed < 0.0:
        parser.error("--rotation-speed must be finite and non-negative")
    if not math.isfinite(args.object_scale) or args.object_scale <= 0.0:
        parser.error("--object-scale must be finite and positive")
    if not math.isfinite(args.startup_delay) or args.startup_delay < 0.0:
        parser.error("--startup-delay must be finite and non-negative")
    if args.object_mesh_path is not None and args.object_case not in MESH_OBJECT_CASES:
        parser.error(
            "--object-mesh-path requires --object-case "
            + "|".join(MESH_OBJECT_CASES)
        )
    if args.max_steps < 0:
        parser.error("--max-steps cannot be negative")
    return args


def require_path(path: Path, label: str, executable: bool = False) -> Path:
    # Do not resolve interpreter symlinks: .venv/bin/python commonly points to
    # /usr/bin/python, but invoking the symlink is what activates the venv.
    path = Path(os.path.abspath(path.expanduser()))
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if executable and not os.access(path, os.X_OK):
        raise PermissionError(f"{label} is not executable: {path}")
    return path


def require_project_path(path: Path, label: str, executable: bool = False) -> Path:
    """Validate an invoked file is addressed through this project tree.

    The path is deliberately checked without resolving symlinks.  A virtual
    environment's ``bin/python`` commonly resolves to a system interpreter,
    while invoking it through ``.venv`` is what selects the local environment.
    """

    checked = require_path(path, label, executable=executable)
    try:
        checked.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"{label} must be inside the project root {PROJECT_ROOT}: {checked}"
        ) from exc
    return checked


def child_environment(mpl_dir: str) -> dict[str, str]:
    """Return a clean child environment without inherited Python search paths."""

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["MPLCONFIGDIR"] = mpl_dir
    return env


def build_commands(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    """Build validated simulator and capture commands.

    Every executable script and interpreter path passed to a child is required
    to be project-local.  This prevents an accidentally configured launcher
    from reaching into a sibling checkout.
    """

    capture_python = require_project_path(
        args.capture_python, "capture Python", executable=True
    )
    sim_python = require_project_path(
        args.sim_python, "simulation Python", executable=True
    )
    track_script = require_project_path(PROJECT_ROOT / "track.py", "tracking script")
    sim_script = require_project_path(
        TELEOP_DIR / "sim_pick_place.py", "simulation script"
    )

    sim_command = [
        str(sim_python),
        str(sim_script),
        "--listen",
        args.listen,
        "--arm-speed",
        str(args.arm_speed),
        "--rotation-speed",
        str(args.rotation_speed),
        "--object-case",
        args.object_case,
        "--object-scale",
        str(args.object_scale),
        "--seed",
        str(args.seed),
        "--max-steps",
        str(max(args.max_steps, 0)),
    ]
    if args.object_mesh_path is not None:
        mesh_path = require_project_path(args.object_mesh_path, "object mesh")
        sim_command.extend(["--object-mesh-path", str(mesh_path)])
    if args.fixed_object:
        sim_command.append("--fixed-object")
    if args.headless:
        sim_command.append("--headless")

    capture_command = [
        str(capture_python),
        str(track_script),
        "--mode",
        "retarget",
        "--udp",
        args.listen,
    ]
    return sim_command, capture_command


def start_process(command: list[str], env: dict[str, str], label: str) -> subprocess.Popen:
    print(f"[LAUNCH] {label}: {shlex.join(command)}", flush=True)
    return subprocess.Popen(
        command, cwd=PROJECT_ROOT, env=env, start_new_session=True
    )


def stop_processes(processes: dict[str, subprocess.Popen]) -> None:
    alive = {name: process for name, process in processes.items() if process.poll() is None}
    for name, process in alive.items():
        print(f"[STOP] interrupting {name}", flush=True)
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + 5.0
    while alive and time.monotonic() < deadline:
        alive = {name: process for name, process in alive.items() if process.poll() is None}
        if alive:
            time.sleep(0.05)

    for name, process in alive.items():
        print(f"[STOP] terminating {name}", flush=True)
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + 2.0
    while alive and time.monotonic() < deadline:
        alive = {name: process for name, process in alive.items() if process.poll() is None}
        if alive:
            time.sleep(0.05)

    for name, process in alive.items():
        print(f"[STOP] killing unresponsive {name}", flush=True)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for process in alive.values():
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass


def wait_for_processes(
    processes: dict[str, subprocess.Popen], poll_interval: float = 0.2
) -> int:
    """Keep the simulator alive when only capture fails.

    The arm keyboard controls and Viewer remain useful without a live hand
    stream, and the receiver already represents this state as HOLD.  Only the
    simulation/Viewer lifetime controls the launcher lifetime.
    """

    capture_exit_reported = False
    while True:
        simulation_code = processes["simulation"].poll()
        if simulation_code is not None:
            print(
                f"[EXIT] simulation exited with code {simulation_code}",
                flush=True,
            )
            return simulation_code

        capture = processes.get("capture")
        if capture is not None and not capture_exit_reported:
            capture_code = capture.poll()
            if capture_code is not None:
                print(
                    f"[WARN] capture exited with code {capture_code}; "
                    "simulation remains open in HOLD. Fix the capture error "
                    "above, then restart the launcher.",
                    flush=True,
                )
                capture_exit_reported = True
        time.sleep(poll_interval)


def main() -> int:
    args = parse_args()
    sim_command, capture_command = build_commands(args)
    sim_env = child_environment("/tmp/mvhc-sapien-matplotlib")
    capture_env = child_environment("/tmp/mvhc-matplotlib")

    processes: dict[str, subprocess.Popen] = {}
    try:
        processes["simulation"] = start_process(sim_command, sim_env, "simulation")
        time.sleep(max(args.startup_delay, 0.0))
        simulation_code = processes["simulation"].poll()
        if simulation_code is not None:
            print(f"[ERROR] simulation exited during startup with code {simulation_code}")
            return simulation_code or 1

        processes["capture"] = start_process(capture_command, capture_env, "capture")
        print(
            "[INFO] Close the SAPIEN viewer or press Ctrl-C here to stop both "
            "processes. Capture errors leave the Viewer open in HOLD."
        )
        return wait_for_processes(processes)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.", flush=True)
        return 0
    finally:
        stop_processes(processes)


if __name__ == "__main__":
    raise SystemExit(main())
