import os
import threading
import time

import cv2
import numpy as np

from config import (
    D435_FPS,
    D435_HEIGHT,
    D435_WIDTH,
    FULL_WIDTH,
    HEIGHT,
    ROTATE_LEFT,
    ROTATE_RIGHT,
)


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
    size = int(height * sine + width * cosine), int(height * cosine + width * sine)
    matrix[:, 2] += size[0] / 2 - width / 2, size[1] / 2 - height / 2
    return cv2.warpAffine(image, matrix, size)


def split_stereo(frame):
    if frame is None or frame.shape[1] != FULL_WIDTH:
        return None, None
    middle = FULL_WIDTH // 2
    return rotate_image(frame[:, :middle], ROTATE_LEFT), rotate_image(
        frame[:, middle:], ROTATE_RIGHT
    )


class StereoCamera:
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
            raise RuntimeError("Install pyrealsense2 to use Intel RealSense D435") from error

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
