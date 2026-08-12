"""MANUS Core 3.1.1 input."""

import ctypes
import ctypes.util
from pathlib import Path
import threading
import time

import numpy as np

from config import (
    MANUS_ENDPOINT,
    MANUS_GLOVE_ID_TO_HANDEDNESS,
    MANUS_POSITION_SCALE_TO_M,
    MANUS_SDK_BRIDGE_PATH,
    MANUS_SDK_VERSION,
    MANUS_STALE_SECONDS,
)
from input.frame import InputFrame
from .adapter import MANUS_TO_STANDARD21, adapt_raw_skeleton, handedness_from_node_info

_ZMQ_PULL = 7
_ZMQ_DONTWAIT = 1
_ZMQ_CONFLATE = 54
_ZMQ_EAGAIN = 11
_MAX_MESSAGE_BYTES = 32768
_MAX_SDK_NODES = 64


class _SdkNode(ctypes.Structure):
    _fields_ = (
        ("node_id", ctypes.c_uint32),
        ("parent_id", ctypes.c_uint32),
        ("side", ctypes.c_int32),
        ("chain_type", ctypes.c_int32),
        ("finger_joint_type", ctypes.c_int32),
        ("position", ctypes.c_float * 3),
        ("rotation_wxyz", ctypes.c_float * 4),
    )


class _SdkFrame(ctypes.Structure):
    _fields_ = (
        ("sequence", ctypes.c_uint64),
        ("publish_time", ctypes.c_uint64),
        ("glove_id", ctypes.c_uint32),
        ("node_count", ctypes.c_uint32),
        ("side", ctypes.c_int32),
        ("nodes", _SdkNode * _MAX_SDK_NODES),
    )


def _sdk_frame_to_packet(frame, received_at):
    """Copy the C bridge ABI into Python without changing any transform."""
    count = int(frame.node_count)
    if count <= 0 or count > _MAX_SDK_NODES:
        raise ValueError(f"Invalid MANUS SDK node count: {count}")
    positions = np.empty((count, 3), dtype=float)
    node_ids = []
    node_info = []
    for row in range(count):
        node = frame.nodes[row]
        positions[row] = node.position
        node_ids.append(int(node.node_id))
        node_info.append(
            {
                "nodeId": int(node.node_id),
                "parentId": int(node.parent_id),
                "side": int(node.side),
                "chainType": int(node.chain_type),
                "fingerJointType": int(node.finger_joint_type),
            }
        )
    return {
        "glove_id": str(frame.glove_id),
        "positions": positions,
        "node_ids": node_ids,
        "node_info": node_info,
        "handedness": {1: "Left", 2: "Right"}.get(int(frame.side)),
        "position_scale_to_m": 1.0,
        "received_at": float(received_at),
        "sdk_publish_time": int(frame.publish_time),
        "coordinate_mode": "WORLD/GLOBAL",
        "calibrated": None,
    }


class OfficialSdkTransport:
    """Read Raw Skeleton callbacks from official Core SDK 3.1.1 Integrated."""

    def __init__(self, bridge_path=MANUS_SDK_BRIDGE_PATH, clock=time.monotonic):
        path = Path(bridge_path)
        if not path.is_file():
            raise RuntimeError(
                f"MANUS SDK bridge is missing: {path}; run `make -C input/manus/assets`"
            )
        self._lib = ctypes.CDLL(str(path))
        self._configure_api()
        self._clock = clock
        self._last_sequence = 0
        self._closed = False
        self._connected = False
        self._init_error = None
        if self._lib.manus_bridge_initialize() != 0:
            self._init_error = self._error()
            return
        self._stop = threading.Event()
        self._connect_thread = threading.Thread(
            target=self._connect_loop, name="manus-sdk-connect", daemon=True
        )
        self._connect_thread.start()

    def _configure_api(self):
        self._lib.manus_bridge_initialize.restype = ctypes.c_int
        self._lib.manus_bridge_connect.restype = ctypes.c_int
        self._lib.manus_bridge_poll.argtypes = (ctypes.POINTER(_SdkFrame),)
        self._lib.manus_bridge_poll.restype = ctypes.c_int
        self._lib.manus_bridge_last_error.restype = ctypes.c_char_p
        self._lib.manus_bridge_last_code.restype = ctypes.c_int
        self._lib.manus_bridge_shutdown.restype = None

    def _error(self):
        message = self._lib.manus_bridge_last_error()
        return message.decode("utf-8", errors="replace") if message else "unknown SDK error"

    def _connect_loop(self):
        while not self._stop.is_set():
            self._connected = self._lib.manus_bridge_connect() == 0
            self._stop.wait(1.0)

    def read(self):
        if self._init_error is not None:
            raise RuntimeError(self._init_error)
        if not self._connected:
            return None
        frame = _SdkFrame()
        result = self._lib.manus_bridge_poll(ctypes.byref(frame))
        if result < 0:
            raise RuntimeError(self._error())
        if result == 0 or frame.sequence == self._last_sequence:
            return None
        self._last_sequence = frame.sequence
        return _sdk_frame_to_packet(frame, self._clock())

    def close(self):
        if self._closed:
            return
        self._closed = True
        stop = getattr(self, "_stop", None)
        if stop is not None:
            stop.set()
        thread = getattr(self, "_connect_thread", None)
        if thread is not None:
            thread.join(timeout=3.0)
        if self._init_error is None:
            self._lib.manus_bridge_shutdown()


def _normalize_glove_id(value):
    return str(value).strip().lower().removeprefix("0x")


def parse_legacy_bridge_message(message, received_at=None):
    """Parse the deployed ``gloveId + 25*(xyz+xyzw)`` CSV protocol."""
    text = message.decode("utf-8") if isinstance(message, bytes) else str(message)
    fields = text.strip().split(",")
    if len(fields) not in (176, 352):
        raise ValueError(f"Unexpected MANUS bridge field count: {len(fields)}")
    received_at = time.monotonic() if received_at is None else float(received_at)
    packets = []
    for offset in range(0, len(fields), 176):
        glove_id = _normalize_glove_id(fields[offset])
        values = np.asarray(fields[offset + 1 : offset + 176], dtype=float).reshape(25, 7)
        if not np.isfinite(values).all():
            raise ValueError("MANUS bridge frame contains non-finite values")
        packets.append(
            {
                "glove_id": glove_id,
                "positions": values[:, :3],
                "received_at": received_at,
                "coordinate_mode": "WORLD/GLOBAL",
            }
        )
    return packets


class LegacyZmqTransport:
    """Small libzmq PULL wrapper, avoiding a mandatory pyzmq dependency."""

    def __init__(self, endpoint=MANUS_ENDPOINT):
        library = ctypes.util.find_library("zmq")
        if library is None:
            raise RuntimeError("libzmq is required for the MANUS bridge")
        self._lib = ctypes.CDLL(library, use_errno=True)
        self._configure_api()
        self._context = self._lib.zmq_ctx_new()
        if not self._context:
            raise RuntimeError("Could not create MANUS ZMQ context")
        self._socket = self._lib.zmq_socket(self._context, _ZMQ_PULL)
        if not self._socket:
            self._lib.zmq_ctx_term(self._context)
            raise RuntimeError("Could not create MANUS ZMQ PULL socket")
        conflation = ctypes.c_int(1)
        if self._lib.zmq_setsockopt(
            self._socket, _ZMQ_CONFLATE, ctypes.byref(conflation), ctypes.sizeof(conflation)
        ) != 0:
            self.close()
            raise RuntimeError("Could not enable MANUS latest-frame conflation")
        endpoint_bytes = endpoint.encode("utf-8")
        if self._lib.zmq_connect(self._socket, endpoint_bytes) != 0:
            self.close()
            raise RuntimeError(f"Could not connect to MANUS bridge: {endpoint}")

    def _configure_api(self):
        void_p = ctypes.c_void_p
        self._lib.zmq_ctx_new.restype = void_p
        self._lib.zmq_ctx_term.argtypes = (void_p,)
        self._lib.zmq_socket.argtypes = (void_p, ctypes.c_int)
        self._lib.zmq_socket.restype = void_p
        self._lib.zmq_setsockopt.argtypes = (void_p, ctypes.c_int, void_p, ctypes.c_size_t)
        self._lib.zmq_connect.argtypes = (void_p, ctypes.c_char_p)
        self._lib.zmq_recv.argtypes = (void_p, void_p, ctypes.c_size_t, ctypes.c_int)
        self._lib.zmq_close.argtypes = (void_p,)
        self._lib.zmq_errno.restype = ctypes.c_int

    def read(self):
        buffer = ctypes.create_string_buffer(_MAX_MESSAGE_BYTES)
        size = self._lib.zmq_recv(self._socket, buffer, len(buffer), _ZMQ_DONTWAIT)
        if size < 0:
            error = self._lib.zmq_errno()
            if error == _ZMQ_EAGAIN:
                return None
            raise RuntimeError(f"MANUS bridge receive failed (ZMQ errno {error})")
        if size >= len(buffer):
            raise ValueError("MANUS bridge message exceeds receive buffer")
        return parse_legacy_bridge_message(buffer.raw[:size])

    def close(self):
        socket, context = getattr(self, "_socket", None), getattr(self, "_context", None)
        self._socket = self._context = None
        if socket:
            self._lib.zmq_close(socket)
        if context:
            self._lib.zmq_ctx_term(context)


class ManusSource:
    """Read fresh WORLD Raw Skeleton positions as common input frames."""

    def __init__(
        self,
        transport=None,
        *,
        stale_seconds=MANUS_STALE_SECONDS,
        glove_id_to_handedness=None,
        clock=time.monotonic,
    ):
        self.transport = OfficialSdkTransport() if transport is None else transport
        self.stale_seconds = float(stale_seconds)
        self.glove_id_to_handedness = {
            _normalize_glove_id(key): value
            for key, value in (
                MANUS_GLOVE_ID_TO_HANDEDNESS
                if glove_id_to_handedness is None
                else glove_id_to_handedness
            ).items()
        }
        self.clock = clock
        self.last_received_at = None
        self._waiting_reported = False
        self._stale_reported = False
        self._first_frame_reported = False
        print("Input source: MANUS")
        print(f"MANUS SDK: official Core SDK {MANUS_SDK_VERSION} Integrated")
        print("MANUS coordinate mode: WORLD/GLOBAL")
        print("p_UseWorldCoordinates: true")
        print("MANUS -> Standard21 fallback mapping: " + str(MANUS_TO_STANDARD21.tolist()))
        print("CMC/root frame implementation: compute_cmc_frame; origin=point1; rotation=R_world_from_cmc")

    def _handedness(self, packet):
        explicit = packet.get("handedness")
        if explicit in ("Left", "Right"):
            return explicit
        semantic = handedness_from_node_info(packet.get("node_info"))
        if semantic is not None:
            return semantic
        return self.glove_id_to_handedness.get(_normalize_glove_id(packet.get("glove_id", "")))

    def _select_packet(self, packets):
        packets = packets if isinstance(packets, (list, tuple)) else [packets]
        for packet in packets:
            if self._handedness(packet) == "Left":
                return packet
        return packets[0] if packets else None

    def _empty_frame(self, timestamp, status):
        return InputFrame(
            timestamp=timestamp,
            points=None,
            handedness=None,
            ready=False,
            status=status,
            preview=None,
        )

    def read(self):
        now = self.clock()
        try:
            packets = self.transport.read()
        except (RuntimeError, ValueError) as error:
            return self._empty_frame(now, f"MANUS INVALID · {error}")
        if packets is None:
            if self.last_received_at is None:
                if not self._waiting_reported:
                    self._waiting_reported = True
                    return self._empty_frame(now, "MANUS WAITING · Core/glove data unavailable")
                return None
            if now - self.last_received_at > self.stale_seconds and not self._stale_reported:
                self._stale_reported = True
                return self._empty_frame(now, "MANUS STALE")
            return None

        packet = self._select_packet(packets)
        if packet is None:
            return self._empty_frame(now, "MANUS WAITING")
        received_at = float(packet.get("received_at", now))
        if now - received_at > self.stale_seconds:
            self.last_received_at, self._stale_reported = received_at, True
            return self._empty_frame(now, "MANUS STALE")
        if packet.get("coordinate_mode", "WORLD/GLOBAL") != "WORLD/GLOBAL":
            return self._empty_frame(now, "MANUS INVALID · bridge is not WORLD/GLOBAL")

        handedness = self._handedness(packet)
        try:
            adapted = adapt_raw_skeleton(
                packet["positions"],
                node_info=packet.get("node_info"),
                node_ids=packet.get("node_ids"),
                scale_to_m=packet.get("position_scale_to_m", MANUS_POSITION_SCALE_TO_M),
            )
        except (KeyError, TypeError, ValueError) as error:
            return self._empty_frame(now, f"MANUS INVALID · {error}")

        self.last_received_at, self._stale_reported = received_at, False
        self._waiting_reported = False
        calibrated = packet.get("calibrated")
        ready = handedness in ("Left", "Right") and calibrated is not False
        calibration_text = "calibration unknown" if calibrated is None else (
            "calibrated" if calibrated else "not calibrated"
        )
        status = f"MANUS TRACKING · {handedness or 'unknown side'} · {calibration_text}"
        if not self._first_frame_reported:
            print(f"MANUS raw node count: {np.asarray(packet['positions']).shape[0]}")
            print(f"MANUS handedness: {handedness or 'Unknown (configure glove ID or provide NodeInfo)'}")
            print(f"MANUS mapping source: {adapted.mapping_source}")
            print(f"MANUS -> Standard21 mapping: {adapted.mapping.tolist()}")
            self._first_frame_reported = True
        return InputFrame(
            timestamp=received_at,
            points=adapted.points,
            handedness=handedness,
            ready=ready,
            status=status,
            finger_pad_directions=None,
            preview=None,
        )

    def close(self):
        self.transport.close()
