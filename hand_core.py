import json
import os
import threading
import time

import cv2
import mediapipe as mp
import numpy as np

from config import (
    ANGLE_FILTER,
    BONE_TOLERANCE,
    CALIBRATION_FRAMES,
    CALIBRATION_HZ,
    D435_FPS,
    D435_HEIGHT,
    D435_WIDTH,
    FINGER_CHAINS,
    FINGER_PERPENDICULAR_COSINE,
    FULL_WIDTH,
    HAND_LANDMARKER_PATH,
    HAND_SWITCH_FRAMES,
    HEIGHT,
    MAX_DEPTH_MM,
    MAX_HAND_RADIUS,
    MAX_REPROJECTION_ERROR,
    MP_DETECTION_CONFIDENCE,
    MP_PRESENCE_CONFIDENCE,
    MP_TRACKING_CONFIDENCE,
    PARAMS_PATH,
    POINT_FILTER,
    ROTATE_LEFT,
    ROTATE_RIGHT,
    STANDARD_PALM_SIZE,
    STALE_FRAMES,
    THUMB_EXTENSION_THRESHOLDS,
)

EPS = 1e-8


def rotate_image(image, angle):
    if angle == 0:
        return image
    if angle in (90, -270):
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle in (-90, 270):
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if abs(angle) == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), -angle, 1)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    size = (int(height * sine + width * cosine), int(height * cosine + width * sine))
    matrix[:, 2] += (size[0] / 2 - width / 2, size[1] / 2 - height / 2)
    return cv2.warpAffine(image, matrix, size)


def split_stereo(frame):
    if frame is None or frame.shape[1] != FULL_WIDTH:
        return None, None
    middle = FULL_WIDTH // 2
    return rotate_image(frame[:, :middle], ROTATE_LEFT), rotate_image(frame[:, middle:], ROTATE_RIGHT)


class Camera:
    def __init__(self, index, width=FULL_WIDTH, height=HEIGHT):
        backend = cv2.CAP_V4L2 if os.name == "posix" else cv2.CAP_ANY
        self.capture = cv2.VideoCapture(index, backend)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open camera {index}")
        self.ok, self.frame = self.capture.read()
        self.timestamp = time.monotonic()
        self.lock, self.running = threading.Lock(), True
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        while self.running:
            ok, frame = self.capture.read()
            if ok and frame is not None:
                with self.lock:
                    self.ok, self.frame, self.timestamp = ok, frame, time.monotonic()
            else:
                time.sleep(0.005)

    def read(self):
        with self.lock:
            if not self.ok or self.frame is None:
                return False, None, None, None
            left, right = split_stereo(self.frame.copy())
            return left is not None, left, right, self.timestamp

    def close(self):
        self.running = False
        self.thread.join(timeout=1)
        self.capture.release()


class RealSenseCamera:
    def __init__(self, index, width=D435_WIDTH, height=D435_HEIGHT, fps=D435_FPS):
        try:
            import pyrealsense2 as rs
        except ImportError as error:
            raise RuntimeError("Install pyrealsense2 to use CAMERA_TYPE='d435'") from error

        devices = rs.context().query_devices()
        if not 0 <= index < len(devices):
            raise RuntimeError(f"RealSense camera {index} not found ({len(devices)} connected)")
        device = devices[index]
        serial = device.get_info(rs.camera_info.serial_number)
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        for stream in (1, 2):
            config.enable_stream(rs.stream.infrared, stream, width, height, rs.format.y8, fps)
        try:
            profile = self.pipeline.start(config)
        except Exception as error:
            raise RuntimeError(f"Unable to open D435 {width}x{height}@{fps}: {error}") from error

        sensor = profile.get_device().first_depth_sensor()
        if sensor.supports(rs.option.emitter_enabled):
            sensor.set_option(rs.option.emitter_enabled, 0)
        left = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        right = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
        self.params = self._params(left, right)
        baseline = np.linalg.norm(self.params["T"])
        print(f"D435 {serial}: {width}x{height}@{fps}, baseline {baseline:.2f} mm")

    @staticmethod
    def _params(left, right):
        def matrix(profile):
            value = profile.get_intrinsics()
            return [[value.fx, 0, value.ppx], [0, value.fy, value.ppy], [0, 0, 1]]

        extrinsics = left.get_extrinsics_to(right)
        return {
            "K1": matrix(left),
            "D1": np.zeros(5),
            "K2": matrix(right),
            "D2": np.zeros(5),
            "R": np.eye(3),
            "T": np.asarray(extrinsics.translation) * 1000,
        }

    def read(self):
        try:
            frames = self.pipeline.wait_for_frames(1000)
        except RuntimeError:
            return False, None, None, None
        left, right = frames.get_infrared_frame(1), frames.get_infrared_frame(2)
        if not left or not right:
            return False, None, None, None
        return (
            True,
            np.asanyarray(left.get_data()),
            np.asanyarray(right.get_data()),
            time.monotonic(),
        )

    def close(self):
        self.pipeline.stop()


class OneEuro:
    def __init__(self, min_cutoff, beta, derivative_cutoff):
        self.min_cutoff, self.beta, self.derivative_cutoff = (
            min_cutoff,
            beta,
            derivative_cutoff,
        )
        self.reset()

    @staticmethod
    def _alpha(dt, cutoff):
        value = 2 * np.pi * cutoff * dt
        return value / (value + 1)

    def __call__(self, value, timestamp):
        value = np.asarray(value, float)
        if self.value is None:
            self.value, self.derivative, self.timestamp = value.copy(), np.zeros_like(value), timestamp
            return value.copy()
        dt = max(timestamp - self.timestamp, 1e-6)
        derivative = (value - self.value) / dt
        alpha = self._alpha(dt, self.derivative_cutoff)
        derivative = alpha * derivative + (1 - alpha) * self.derivative
        alpha = self._alpha(dt, self.min_cutoff + self.beta * np.abs(derivative))
        self.value = alpha * value + (1 - alpha) * self.value
        self.derivative, self.timestamp = derivative, timestamp
        return self.value.copy()

    def reset(self):
        self.value = self.derivative = self.timestamp = None


def _unit(vector, fallback=(1.0, 0.0, 0.0)):
    norm = np.linalg.norm(vector)
    if norm > EPS:
        return vector / norm
    fallback = np.asarray(fallback, float)
    return fallback / max(np.linalg.norm(fallback), EPS)


def _safe_unit(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm > EPS else None


def _rotate(vector, axis, degrees):
    angle, axis = np.radians(degrees), _unit(axis)
    return (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * np.dot(axis, vector) * (1 - np.cos(angle))
    )


def _project(vector, axis, fallback):
    return _unit(vector - axis * np.dot(vector, axis), fallback)


def _signed_angle(start, end, axis):
    start, end = _project(start, axis, start), _project(end, axis, start)
    return np.degrees(
        np.arctan2(np.dot(np.cross(start, end), axis), np.clip(np.dot(start, end), -1, 1))
    )


def _finger_frames(points, handedness):
    forward = _unit(points[9] - points[0], (0, 1, 0))
    thumbward = _unit(points[5] - points[17], (1, 0, 0))
    normal = _unit(np.cross(thumbward, forward), (0, 0, 1))
    normal = -normal if handedness == "Left" else normal
    parent = _unit(points[2] - points[1])
    palmward = points[9] - points[2]
    palmward -= parent * np.dot(palmward, parent)
    frames = [((2, 3, 4), parent, _unit(np.cross(parent, _unit(palmward, normal)), normal))]
    for mcp in (5, 9, 13, 17):
        proximal = _unit(points[mcp + 1] - points[mcp])
        neutral = _unit(
            proximal - normal * np.dot(proximal, normal), _unit(points[mcp] - points[0])
        )
        frames.append(((mcp, mcp + 1, mcp + 2, mcp + 3), neutral, _unit(np.cross(neutral, normal))))
    return frames


def extract_angles(points, handedness):
    angles = []
    for chain, direction, axis in _finger_frames(points, handedness):
        for start, end in zip(chain[:-1], chain[1:]):
            measured = _project(points[end] - points[start], axis, direction)
            angles.append(_signed_angle(direction, measured, axis))
            direction = measured
    return np.asarray(angles)


def apply_angles(points, handedness, angles):
    result, offset = points.copy(), 0
    for chain, direction, axis in _finger_frames(points, handedness):
        for position, (start, end) in enumerate(zip(chain[:-1], chain[1:])):
            measured = _project(result[end] - result[start], axis, direction)
            change = angles[offset] - _signed_angle(direction, measured, axis)
            origin = result[start].copy()
            for child in chain[position + 1 :]:
                result[child] = origin + _rotate(result[child] - origin, axis, change)
            direction = _project(result[end] - result[start], axis, direction)
            offset += 1
    return result


def enforce_lengths(points, lengths):
    result = points.copy()
    for chain in FINGER_CHAINS:
        for position, (start, end) in enumerate(zip(chain[:-1], chain[1:])):
            vector = result[end] - result[start]
            nominal = lengths[(start, end)]
            target = np.clip(
                np.linalg.norm(vector),
                nominal * (1 - BONE_TOLERANCE),
                nominal * (1 + BONE_TOLERANCE),
            )
            shift = result[start] + _unit(vector) * target - result[end]
            result[list(chain[position + 1 :])] += shift
    return result


def relative_points(points):
    centered = points - points[0]
    palm_size = np.mean([np.linalg.norm(centered[index]) for index in (5, 9, 13, 17)])
    if palm_size < EPS:
        return None
    points = centered * STANDARD_PALM_SIZE / palm_size
    z_axis = _unit(points[9])
    x_axis = _unit(np.cross(points[5] - points[17], z_axis))
    y_axis = _unit(np.cross(z_axis, x_axis))
    return points @ np.column_stack((x_axis, y_axis, z_axis))


def fingertip_pad_directions(points, handedness, thumb_inside=None, previous_fingers=None):
    """Return Thumb-to-Little unit pad directions and the thumb hysteresis state."""
    points = np.asarray(points, float)
    if points.shape != (21, 3) or not np.isfinite(points).all() or handedness not in ("Left", "Right"):
        return None, thumb_inside

    ey = _safe_unit(points[5] - points[17])
    if ey is None:
        return None, thumb_inside
    ex = _safe_unit(points[9] - points[0] - ey * np.dot(points[9] - points[0], ey))
    normal = None if ex is None else _safe_unit(np.cross(ex, ey))
    if normal is None:
        return None, thumb_inside
    normal *= 1 if handedness == "Left" else -1
    center = points[[0, 5, 9, 13, 17]].mean(axis=0)
    previous = None if previous_fingers is None else np.asarray(previous_fingers, float)
    if previous is not None and (previous.shape != (4, 3) or not np.isfinite(previous).all()):
        previous = None

    def candidate(start, tip):
        axis = _safe_unit(points[tip] - points[start])
        if axis is None:
            return None, None
        direction = _safe_unit(normal - axis * np.dot(normal, axis))
        return axis, direction

    thumb_axis, thumb = candidate(3, 4)
    if thumb is None:
        return None, thumb_inside
    lengths = np.linalg.norm(points[2:5] - points[1:4], axis=1)
    if np.any(lengths <= EPS):
        return None, thumb_inside
    extension = np.linalg.norm(points[4] - points[1]) / lengths.sum()
    low, high = THUMB_EXTENSION_THRESHOLDS
    inside = extension < (low + high) / 2 if thumb_inside is None else thumb_inside
    if extension <= low:
        inside = True
    elif extension >= high:
        inside = False
    if inside:
        inward = points[5] - points[4]
        inward -= thumb_axis * np.dot(inward, thumb_axis)
        if np.dot(thumb, inward) < 0:
            thumb = -thumb

    directions = [thumb]
    for index, (mcp, tip) in enumerate(((5, 8), (9, 12), (13, 16), (17, 20))):
        axis = _safe_unit(points[tip] - points[mcp])
        if axis is not None and abs(np.dot(axis, normal)) >= FINGER_PERPENDICULAR_COSINE:
            direction = None if previous is None else _safe_unit(previous[index])
            if direction is None:
                return None, thumb_inside
            directions.append(direction)
            continue
        axis, direction = candidate(mcp, tip)
        if direction is None:
            return None, thumb_inside
        inward = center - points[tip]
        inward -= axis * np.dot(inward, axis)
        directions.append(direction if np.dot(direction, inward) >= 0 else -direction)
    return np.asarray(directions), inside


class Kinematics:
    def __init__(self, calibration_frames=CALIBRATION_FRAMES, calibration_hz=CALIBRATION_HZ):
        self.points_filter = OneEuro(*POINT_FILTER)
        self.angle_filter = OneEuro(*ANGLE_FILTER)
        self.calibration_frames = calibration_frames
        self.calibration_period = 1 / calibration_hz
        self.samples = {edge: [] for chain in FINGER_CHAINS for edge in zip(chain[:-1], chain[1:])}
        self.count, self.lengths, self.last_sample = 0, None, None

    def update(self, points, handedness, timestamp):
        points = self.points_filter(points, timestamp)
        if self.lengths is None:
            if self.last_sample is None or timestamp - self.last_sample >= self.calibration_period - 1e-6:
                for (start, end), values in self.samples.items():
                    values.append(np.linalg.norm(points[end] - points[start]))
                self.count += 1
                self.last_sample = timestamp
                if self.count >= self.calibration_frames:
                    self.lengths = {edge: float(np.median(values)) for edge, values in self.samples.items()}
            return points, f"CALIBRATION ({self.count}/{self.calibration_frames})"
        points = enforce_lengths(points, self.lengths)
        angles = self.angle_filter(np.maximum(extract_angles(points, handedness), 0), timestamp)
        points = apply_angles(points, handedness, angles)
        return points, "GESTURE TRACKING"

    def reset_filters(self):
        self.points_filter.reset()
        self.angle_filter.reset()


class StableHandedness:
    def __init__(self, switch_frames=HAND_SWITCH_FRAMES):
        self.switch_frames = switch_frames
        self.stable = self.candidate = None
        self.count = 0

    def update(self, label):
        if self.stable is None:
            self.stable = label
        elif label == self.stable:
            self.candidate, self.count = None, 0
        else:
            if label != self.candidate:
                self.candidate, self.count = label, 1
            else:
                self.count += 1
            if self.count >= self.switch_frames:
                self.stable, self.candidate, self.count = label, None, 0
        return self.stable


class HandDetector:
    def __init__(self, model_path=HAND_LANDMARKER_PATH):
        if not model_path.is_file():
            raise FileNotFoundError(f"MediaPipe hand model not found: {model_path}")
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=MP_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MP_PRESENCE_CONFIDENCE,
            min_tracking_confidence=MP_TRACKING_CONFIDENCE,
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.last_timestamp_ms = -1

    def detect(self, rgb, timestamp):
        timestamp_ms = max(int(round(timestamp * 1000)), self.last_timestamp_ms + 1)
        self.last_timestamp_ms = timestamp_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self.detector.detect_for_video(image, timestamp_ms)
        if not result.hand_landmarks or not result.handedness:
            return None, None
        points = np.array([[point.x, point.y] for point in result.hand_landmarks[0]])
        category = result.handedness[0][0]
        return points, category.category_name

    def close(self):
        self.detector.close()


def geometry_error(points, reprojection_error):
    if not np.isfinite(points).all():
        return "non-finite"
    if reprojection_error > MAX_REPROJECTION_ERROR:
        return "reprojection"
    if np.any(points[:, 2] > MAX_DEPTH_MM):
        return "depth"
    if np.max(np.linalg.norm(points - points[0], axis=1)) > MAX_HAND_RADIUS:
        return "hand-size"
    return None


class StereoProcessor:
    def __init__(self, params=None):
        params = json.loads(PARAMS_PATH.read_text()) if params is None else params
        self.K1, self.D1 = np.asarray(params["K1"]), np.asarray(params["D1"])
        self.K2, self.D2 = np.asarray(params["K2"]), np.asarray(params["D2"])
        rotation, translation = np.asarray(params["R"]), np.asarray(params["T"])
        self.P1 = self.K1 @ np.c_[np.eye(3), np.zeros(3)]
        self.P2 = self.K2 @ np.c_[rotation, translation]
        self.left_detector, self.right_detector = HandDetector(), HandDetector()
        self.handedness, self.kinematics = StableHandedness(), Kinematics()
        self.thumb_modes, self.finger_directions = {}, {}
        self.last, self.bad_frames = None, 0

    @staticmethod
    def _empty():
        return {
            "found": False,
            "stale": False,
            "handedness": None,
            "keypoint_absolute": None,
            "keypoint_relative": None,
            "fingertip_directions": None,
            "image_left": None,
            "image_right": None,
            "px_left": None,
            "px_right": None,
            "phase": "WAITING",
            "quality": {"reprojection_error": None, "rejected_reason": None},
        }

    def _reject(self, output, reason, reprojection=None):
        self.bad_frames += 1
        output["quality"] = {"reprojection_error": reprojection, "rejected_reason": reason}
        if self.last is not None and self.bad_frames <= STALE_FRAMES:
            for key in ("handedness", "keypoint_absolute", "keypoint_relative", "fingertip_directions"):
                output[key] = self.last[key]
            output.update(found=True, stale=True, phase=f"STALE ({self.bad_frames}/{STALE_FRAMES})")
        elif self.bad_frames > STALE_FRAMES:
            self.last = None
            self.thumb_modes.clear()
            self.finger_directions.clear()
            self.kinematics.reset_filters()
        return output

    def process(self, left, right, timestamp=None):
        output = self._empty()
        if left is None or right is None:
            return self._reject(output, "frame-size")
        timestamp = time.monotonic() if timestamp is None else timestamp
        conversion = cv2.COLOR_GRAY2RGB if left.ndim == 2 else cv2.COLOR_BGR2RGB
        left_rgb, right_rgb = cv2.cvtColor(left, conversion), cv2.cvtColor(right, conversion)
        output["image_left"], output["image_right"] = left_rgb, right_rgb
        left_norm, label = self.left_detector.detect(left_rgb, timestamp)
        right_norm, _ = self.right_detector.detect(right_rgb, timestamp)
        if left_norm is None or right_norm is None:
            return self._reject(output, "detection")

        label = self.handedness.update(label)
        left_points = left_norm * (left.shape[1], left.shape[0])
        right_points = right_norm * (right.shape[1], right.shape[0])
        output["px_left"], output["px_right"] = left_points, right_points
        left_ud = cv2.undistortPoints(left_points[:, None], self.K1, self.D1, P=self.K1).reshape(-1, 2)
        right_ud = cv2.undistortPoints(right_points[:, None], self.K2, self.D2, P=self.K2).reshape(-1, 2)
        homogeneous = cv2.triangulatePoints(self.P1, self.P2, left_ud.T, right_ud.T)
        if np.any(np.abs(homogeneous[3]) < EPS):
            return self._reject(output, "triangulation")
        points = (homogeneous[:3] / homogeneous[3]).T
        projection = np.c_[points, np.ones(21)].T
        p1, p2 = self.P1 @ projection, self.P2 @ projection
        p1, p2 = (p1[:2] / p1[2]).T, (p2[:2] / p2[2]).T
        reprojection = float(
            np.mean(np.r_[np.linalg.norm(p1 - left_ud, axis=1), np.linalg.norm(p2 - right_ud, axis=1)])
        )
        reason = geometry_error(points, reprojection)
        if reason:
            return self._reject(output, reason, reprojection)

        points, phase = self.kinematics.update(points, label, timestamp)
        relative = relative_points(points)
        directions = None
        if relative is not None:
            directions, thumb_inside = fingertip_pad_directions(
                relative, label, self.thumb_modes.get(label), self.finger_directions.get(label)
            )
            if directions is not None:
                self.thumb_modes[label] = thumb_inside
                self.finger_directions[label] = directions[1:].copy()
        output.update(
            found=True,
            handedness=label,
            keypoint_absolute=points,
            keypoint_relative=relative,
            fingertip_directions=directions,
            phase=f"{phase} - {label} Hand",
            quality={"reprojection_error": reprojection, "rejected_reason": None},
        )
        self.bad_frames, self.last = 0, output
        return output

    def close(self):
        self.left_detector.close()
        self.right_detector.close()
