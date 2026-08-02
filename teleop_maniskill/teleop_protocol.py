"""Small, dependency-light UDP protocol for standalone MMHand teleoperation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import socket
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "mvhc.retarget.v1"
MAX_JSON_BYTES = 8 * 1024
# Keep the wire contract local to this package so the simulator can be copied
# or launched without importing capture-side configuration modules.
JOINT_NAMES = (
    "Little_MCP_AA",
    "Little_MCP_FE",
    "finger_4_distal_phalanx_1_PIP_Joint",
    "finger_4_fingertip_1_DIP_Joint",
    "Ring_MCP_AA",
    "Ring_MCP_FE",
    "finger_3_distal_phalanx_1_PIP_Joint",
    "finger_3_fingertip_1_DIP_Joint",
    "Middle_MCP_AA",
    "Middle_MCP_FE",
    "finger_2_distal_phalanx_1_PIP_Joint",
    "finger_2_fingertip_1_DIP_Joint",
    "Index_MCP_AA",
    "Index_MCP_FE",
    "finger_1_distal_phalanx_1_PIP_Joint",
    "finger_1_fingertip_1_DIP_Joint",
    "Thumb_MCP_AA",
    "Thumb_MCP_FE",
    "mmhand_thumb_1_finger_7_distal_phalanx_1_PIP_Joint",
    "mmhand_thumb_1_finger_7_fingertip_1_DIP_Joint",
    "Thumb_CMC",
)


@dataclass(frozen=True)
class RetargetPacket:
    """A validated retarget sample received from the capture process."""

    schema: str
    seq: int
    sent_monotonic: float
    valid: bool
    phase: str
    handedness: str | None
    joint_names: tuple[str, ...]
    q_rad: np.ndarray | None
    quality: dict[str, Any]


def parse_endpoint(endpoint: str) -> tuple[str, int]:
    """Parse a ``host:port`` UDP endpoint."""

    if not isinstance(endpoint, str):
        raise TypeError("endpoint must be a string")
    host, separator, port_text = endpoint.rpartition(":")
    if not separator or not host or not port_text:
        raise ValueError("endpoint must have the form host:port")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError("endpoint port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ValueError("endpoint port must be between 1 and 65535")
    return host, port


def _json_object(
    seq: int,
    sent_monotonic: float,
    valid: bool,
    phase: str,
    handedness: str | None,
    joint_names: Sequence[str],
    q_rad: Sequence[float] | np.ndarray | None,
    quality: Mapping[str, Any],
) -> dict[str, Any] | None:
    if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
        return None
    if (
        isinstance(sent_monotonic, bool)
        or not isinstance(sent_monotonic, (int, float))
        or not math.isfinite(float(sent_monotonic))
        or sent_monotonic < 0
    ):
        return None
    if not isinstance(valid, bool):
        return None
    if not isinstance(phase, str):
        return None
    if handedness is not None and not isinstance(handedness, str):
        return None
    if not isinstance(joint_names, (list, tuple)) or tuple(joint_names) != JOINT_NAMES:
        return None
    if not isinstance(quality, Mapping):
        return None
    if valid and (handedness != "Left" or not phase.startswith("GESTURE")):
        return None

    if valid:
        try:
            q = np.asarray(q_rad, dtype=float)
        except (TypeError, ValueError):
            return None
        if q.shape != (21,) or not np.isfinite(q).all():
            return None
        q_json: list[float] | None = q.tolist()
    else:
        if q_rad is not None:
            return None
        q_json = None

    return {
        "schema": SCHEMA,
        "seq": seq,
        "sent_monotonic": float(sent_monotonic),
        "valid": valid,
        "phase": phase,
        "handedness": handedness,
        "joint_names": list(JOINT_NAMES),
        "q_rad": q_json,
        "quality": dict(quality),
    }


def _decode_packet(data: bytes) -> RetargetPacket | None:
    if len(data) >= MAX_JSON_BYTES:
        return None
    try:
        payload = json.loads(
            data.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        return None

    required = {
        "schema",
        "seq",
        "sent_monotonic",
        "valid",
        "phase",
        "handedness",
        "joint_names",
        "q_rad",
        "quality",
    }
    if set(payload) != required:
        return None
    normalized = _json_object(
        payload["seq"],
        payload["sent_monotonic"],
        payload["valid"],
        payload["phase"],
        payload["handedness"],
        payload["joint_names"],
        payload["q_rad"],
        payload["quality"],
    )
    if normalized is None:
        return None
    return RetargetPacket(
        schema=SCHEMA,
        seq=normalized["seq"],
        sent_monotonic=normalized["sent_monotonic"],
        valid=normalized["valid"],
        phase=normalized["phase"],
        handedness=normalized["handedness"],
        joint_names=JOINT_NAMES,
        q_rad=(
            None
            if normalized["q_rad"] is None
            else np.asarray(normalized["q_rad"], dtype=float)
        ),
        quality=normalized["quality"],
    )


class UdpRetargetSender:
    """Non-blocking sender for the latest hand-retarget sample."""

    def __init__(self, endpoint: str):
        self.endpoint = parse_endpoint(endpoint)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)

    def send(
        self,
        seq: int,
        sent_monotonic: float,
        valid: bool,
        phase: str,
        handedness: str | None,
        joint_names: Sequence[str],
        q_rad: Sequence[float] | np.ndarray | None,
        quality: Mapping[str, Any],
    ) -> bool:
        payload = _json_object(
            seq,
            sent_monotonic,
            valid,
            phase,
            handedness,
            joint_names,
            q_rad,
            quality,
        )
        if payload is None:
            return False
        try:
            data = json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError):
            return False
        if len(data) >= MAX_JSON_BYTES:
            return False
        try:
            return self.socket.sendto(data, self.endpoint) == len(data)
        except (BlockingIOError, OSError):
            return False

    def close(self) -> None:
        self.socket.close()


class LatestUdpRetargetReceiver:
    """Drain a UDP socket and return only the newest valid sequence number."""

    def __init__(self, endpoint: str):
        self.endpoint = parse_endpoint(endpoint)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.endpoint)
        self.socket.setblocking(False)
        self._last_seq = -1

    def poll(self) -> RetargetPacket | None:
        newest = None
        while True:
            try:
                data, _ = self.socket.recvfrom(65535)
            except BlockingIOError:
                break
            except OSError:
                return newest
            packet = _decode_packet(data)
            if (
                packet is not None
                and packet.seq > self._last_seq
                and (newest is None or packet.seq > newest.seq)
            ):
                newest = packet
        if newest is not None:
            self._last_seq = newest.seq
        return newest

    def close(self) -> None:
        self.socket.close()


def capture_to_sim(
    q_rad: Sequence[float] | np.ndarray,
    lower: Sequence[float] | np.ndarray | None = None,
    upper: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Pass through same-URDF J00..J20 radians and apply joint limits."""

    q = np.asarray(q_rad, dtype=float)
    if q.shape != (21,) or not np.isfinite(q).all():
        raise ValueError("q_rad must contain 21 finite radians")

    lower_array = (
        np.full(21, -np.inf)
        if lower is None
        else np.asarray(lower, dtype=float)
    )
    upper_array = (
        np.full(21, np.inf)
        if upper is None
        else np.asarray(upper, dtype=float)
    )
    if lower_array.shape != (21,) or upper_array.shape != (21,):
        raise ValueError("lower and upper must each contain 21 values")
    if np.isnan(lower_array).any() or np.isnan(upper_array).any():
        raise ValueError("lower and upper cannot contain NaN")
    if np.any(lower_array > upper_array):
        raise ValueError("lower cannot exceed upper")
    return np.clip(q, lower_array, upper_array)


def rate_limit(
    current: Sequence[float] | np.ndarray,
    target: Sequence[float] | np.ndarray,
    max_rate: float | Sequence[float] | np.ndarray,
    dt: float,
) -> np.ndarray:
    """Limit element-wise target motion to ``max_rate * dt``."""

    current_array = np.asarray(current, dtype=float)
    target_array = np.asarray(target, dtype=float)
    rate_array = np.asarray(max_rate, dtype=float)
    if current_array.shape != target_array.shape:
        raise ValueError("current and target must have the same shape")
    if not np.isfinite(current_array).all() or not np.isfinite(target_array).all():
        raise ValueError("current and target must be finite")
    if not math.isfinite(dt) or dt < 0:
        raise ValueError("dt must be finite and non-negative")
    try:
        rate_array = np.broadcast_to(rate_array, current_array.shape)
    except ValueError as exc:
        raise ValueError("max_rate must be scalar or match the target shape") from exc
    if not np.isfinite(rate_array).all() or np.any(rate_array < 0):
        raise ValueError("max_rate must be finite and non-negative")
    step = rate_array * dt
    return current_array + np.clip(target_array - current_array, -step, step)
