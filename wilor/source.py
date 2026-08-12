"""Small D435-color WiLoR ONNX source; no stereo or PyTorch dependencies."""

from dataclasses import dataclass
import threading
import time

import cv2
import numpy as np

from config import (
    WILOR_ASSET_DIR,
    WILOR_CAMERA_INDEX,
    WILOR_CONFIDENCE,
    WILOR_CROP_FACTOR,
    WILOR_DETECT_EVERY,
    WILOR_DEVICE_ID,
    WILOR_FPS,
    WILOR_HEIGHT,
    WILOR_IOU,
    WILOR_WIDTH,
)
from hand import HandFrame, relative_hand

MEAN = np.array((0.485, 0.456, 0.406), np.float32)[:, None, None]
STD = np.array((0.229, 0.224, 0.225), np.float32)[:, None, None]
TIP_VERTICES = np.array((744, 320, 443, 554, 671))
JOINT_ORDER = np.array((0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18,
                        10, 11, 12, 19, 7, 8, 9, 20))
# Four outward-facing volar triangles per distal phalanx, Thumb to Little.
PAD_FACE_ROWS = np.array(((1298, 1303, 1304, 1297),
                          (500, 464, 497, 498),
                          (698, 697, 731, 732),
                          (950, 951, 952, 972),
                          (1184, 1183, 1182, 1204)))


class LatestColorFrame:
    def __init__(self, index, width, height, fps):
        try:
            import pyrealsense2 as rs
        except ImportError as error:
            raise RuntimeError("Install pyrealsense2 to use WiLoR") from error
        devices = rs.context().query_devices()
        if not 0 <= index < len(devices):
            raise RuntimeError(f"RealSense camera {index} not found ({len(devices)} connected)")
        self.rs = rs
        self.serial = devices[index].get_info(rs.camera_info.serial_number)
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(config)
        self.condition = threading.Condition()
        self.frame = None
        self.timestamp = 0.0
        self.sequence = 0
        self.error = None
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        try:
            while self.running:
                color = self.pipeline.wait_for_frames(1000).get_color_frame()
                if color:
                    with self.condition:
                        self.frame = np.asanyarray(color.get_data()).copy()
                        self.timestamp = time.perf_counter()
                        self.sequence += 1
                        self.condition.notify_all()
        except Exception as error:
            if self.running:
                with self.condition:
                    self.error = error
                    self.condition.notify_all()

    def read(self, after, timeout=2.0):
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.sequence <= after and self.error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for a D435 color frame")
                self.condition.wait(remaining)
            if self.error is not None:
                raise RuntimeError(f"D435 color capture failed: {self.error}")
            return self.sequence, self.frame.copy(), self.timestamp

    def close(self):
        self.running = False
        self.thread.join(timeout=1.2)
        self.pipeline.stop()


def make_session(path, device_id):
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("Install onnxruntime-gpu to use WiLoR") from error
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls(directory="")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("WiLoR requires ONNX Runtime CUDAExecutionProvider")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    session = ort.InferenceSession(
        str(path), options,
        providers=[("CUDAExecutionProvider", {"device_id": device_id}),
                   "CPUExecutionProvider"],
    )
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError("WiLoR CUDA session creation failed")
    return session


def letterbox(frame, size=640):
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    resized = cv2.resize(frame, (round(width * scale), round(height * scale)))
    x = (size - resized.shape[1]) / 2
    y = (size - resized.shape[0]) / 2
    left, top = round(x - 0.1), round(y - 0.1)
    canvas = cv2.copyMakeBorder(
        resized, top, size - resized.shape[0] - top,
        left, size - resized.shape[1] - left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None]), scale, left, top


def box_iou(one, many):
    intersection_size = np.maximum(np.minimum(one[2:], many[:, 2:]) -
                                   np.maximum(one[:2], many[:, :2]), 0)
    intersection = intersection_size[:, 0] * intersection_size[:, 1]
    one_area = np.prod(np.maximum(one[2:] - one[:2], 0))
    many_area = np.prod(np.maximum(many[:, 2:] - many[:, :2], 0), axis=1)
    return intersection / np.maximum(one_area + many_area - intersection, 1e-9)


def nms(boxes, scores, threshold):
    order, keep = scores.argsort()[::-1], []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        order = order[1:][box_iou(boxes[current], boxes[order[1:]]) <= threshold]
    return keep


class Detector:
    def __init__(self, session, confidence, iou):
        self.session, self.confidence, self.iou = session, confidence, iou
        self.dtype = np.float16 if "float16" in session.get_inputs()[0].type else np.float32

    def __call__(self, frame):
        tensor, scale, pad_x, pad_y = letterbox(frame)
        values = self.session.run(None, {"images": tensor.astype(self.dtype) / 255})[0][0].T
        scores = values[:, 4:6].max(1)
        labels = values[:, 4:6].argmax(1).astype(np.int32)
        valid = scores >= self.confidence
        values, scores, labels = values[valid], scores[valid], labels[valid]
        if not len(values):
            return []
        boxes = np.c_[values[:, :2] - values[:, 2:4] / 2,
                      values[:, :2] + values[:, 2:4] / 2]
        selected = []
        for label in (0, 1):
            indices = np.flatnonzero(labels == label)
            selected.extend(indices[nms(boxes[indices], scores[indices], self.iou)])
        height, width = frame.shape[:2]
        detections = []
        for index in sorted(selected, key=lambda item: scores[item], reverse=True)[:1]:
            box = boxes[index].astype(np.float32)
            box[[0, 2]] = (box[[0, 2]] - pad_x) / scale
            box[[1, 3]] = (box[[1, 3]] - pad_y) / scale
            box[[0, 2]] = np.clip(box[[0, 2]], 0, width - 1)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, height - 1)
            if np.all(box[2:] - box[:2] >= 4):
                detections.append((box, int(labels[index]), float(scores[index])))
        return detections


@dataclass
class Track:
    box: np.ndarray
    handedness: int
    score: float
    velocity: np.ndarray
    missed: int = 0


class Tracker:
    def __init__(self):
        self.track = None

    def update(self, detections=None):
        if detections is None:
            if self.track is not None:
                self.track.box += self.track.velocity
            return [] if self.track is None else [self.track]
        if not detections:
            if self.track is not None:
                self.track.missed += 1
                self.track.box += self.track.velocity
                if self.track.missed > 2:
                    self.track = None
            return [] if self.track is None else [self.track]
        box, handedness, score = detections[0]
        if self.track is None or self.track.handedness != handedness:
            self.track = Track(box.copy(), handedness, score, np.zeros(4, np.float32))
        else:
            delta = box - self.track.box
            self.track.velocity = 0.55 * self.track.velocity + 0.45 * delta
            self.track.box = 0.35 * self.track.box + 0.65 * box
            self.track.score, self.track.missed = score, 0
        return [self.track]


def crop_hands(frame, tracks, factor, dtype):
    crops, centers, box_sizes = [], [], []
    height, width = frame.shape[:2]
    for track in tracks:
        center = (track.box[:2] + track.box[2:]) / 2
        box_width, box_height = factor * (track.box[2:] - track.box[:2])
        box_height = max(box_height, box_width * 256 / 192)
        box_width = max(box_width, box_height * 192 / 256)
        box_size = max(box_width, box_height, 1)
        image, center_x = frame, center[0]
        if track.handedness == 0:
            image, center_x = frame[:, ::-1], width - center_x - 1
        scale = 256 / box_size
        transform = np.array(((scale, 0, 128 - scale * center_x),
                              (0, scale, 128 - scale * center[1])), np.float32)
        patch = cv2.warpAffine(image, transform, (256, 256))
        rgb = patch[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255
        crops.append(((rgb - MEAN) / STD)[:, :, 32:-32].astype(dtype))
        centers.append(center)
        box_sizes.append(box_size)
    return (np.ascontiguousarray(np.stack(crops)), np.asarray(centers),
            np.asarray(box_sizes))


class Reconstructor:
    def __init__(self, session, factor):
        self.session, self.factor = session, factor
        self.dtype = np.float16 if "float16" in session.get_inputs()[0].type else np.float32

    def __call__(self, frame, tracks):
        crops, centers, box_sizes = crop_hands(frame, tracks, self.factor, self.dtype)
        vertices, camera = self.session.run(None, {"images": crops})
        vertices, camera = vertices.astype(np.float32), camera.astype(np.float32)
        count = len(tracks)
        if (vertices.shape != (count, 778, 3) or camera.shape != (count, 3)
                or not np.isfinite(vertices).all() or not np.isfinite(camera).all()):
            raise ValueError("Invalid WiLoR model output")
        multiplier = 2 * np.asarray([track.handedness for track in tracks]) - 1
        camera[:, 1] *= multiplier
        width, height = frame.shape[1], frame.shape[0]
        focal = 5000 / 256 * max(width, height)
        scaled_box = box_sizes * camera[:, 0] + 1e-9
        if np.any(scaled_box <= 0):
            raise ValueError("Invalid WiLoR camera scale")
        translation = np.stack((
            2 * (centers[:, 0] - width / 2) / scaled_box + camera[:, 1],
            2 * (centers[:, 1] - height / 2) / scaled_box + camera[:, 2],
            2 * focal / scaled_box,
        ), axis=1)
        return vertices, translation.astype(np.float32)


def mesh_hand(vertices, right, regressor, faces):
    vertices = np.asarray(vertices, np.float32)
    if vertices.shape != (778, 3) or not np.isfinite(vertices).all():
        raise ValueError("WiLoR vertices must be finite with shape (778, 3)")
    joints = np.vstack((regressor @ vertices, vertices[TIP_VERTICES]))[JOINT_ORDER]
    triangles = vertices[faces[PAD_FACE_ROWS]]
    normals = np.cross(triangles[:, :, 1] - triangles[:, :, 0],
                       triangles[:, :, 2] - triangles[:, :, 0]).sum(1)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    if np.any(lengths < 1e-8):
        raise ValueError("Degenerate WiLoR finger-pad surface")
    normals /= lengths
    if not right:
        joints[:, 0] *= -1
        normals[:, 0] *= -1
    points, directions = relative_hand(joints, normals)
    if points is None:
        raise ValueError("Degenerate WiLoR hand")
    return points, directions, "Right" if right else "Left"


def draw_mesh(frame, vertices, translation, right, faces):
    vertices = np.asarray(vertices, np.float32).copy()
    vertices[:, 0] *= 1 if right else -1
    points_3d = vertices + np.asarray(translation, np.float32)
    depth = np.maximum(points_3d[:, 2], 1e-5)
    focal = 5000 / 256 * max(frame.shape[:2])
    points = np.stack((focal * points_3d[:, 0] / depth + frame.shape[1] / 2,
                       focal * points_3d[:, 1] / depth + frame.shape[0] / 2), axis=1)
    triangles = np.clip(points[faces], -32768, 32767).astype(np.int32)
    overlay = frame.copy()
    color = (92, 150, 255) if right else (255, 145, 92)
    cv2.fillPoly(overlay, triangles, color, lineType=cv2.LINE_AA)
    cv2.polylines(overlay, triangles, True, (45, 55, 80), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.67, frame, 0.33, 0, dst=frame)


def preview(frame, tracks, mesh=None):
    frame = frame.copy()
    if mesh is not None:
        draw_mesh(frame, *mesh)
    for track in tracks:
        x1, y1, x2, y2 = track.box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 230, 255), 2)
        cv2.putText(frame, "R" if track.handedness else "L", (x1, max(20, y1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 230, 255), 2)
    canvas = np.zeros((360, 1280, 3), np.uint8)
    canvas[:, 400:880] = cv2.resize(frame, (480, 360))
    return canvas


class WilorSource:
    def __init__(self):
        paths = {name: WILOR_ASSET_DIR / name for name in (
            "detector_fp16.onnx", "wilor_fp16.onnx", "joint_regressor.npy",
            "mano_faces.npy",
        )}
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing WiLoR assets: " + ", ".join(missing))
        detector = make_session(paths["detector_fp16.onnx"], WILOR_DEVICE_ID)
        model = make_session(paths["wilor_fp16.onnx"], WILOR_DEVICE_ID)
        self.detector = Detector(detector, WILOR_CONFIDENCE, WILOR_IOU)
        self.reconstructor = Reconstructor(model, WILOR_CROP_FACTOR)
        self.regressor = np.load(paths["joint_regressor.npy"], allow_pickle=False)
        self.faces = np.load(paths["mano_faces.npy"], allow_pickle=False)
        if self.regressor.shape != (16, 778) or self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("Invalid WiLoR geometry assets")
        self.tracker = Tracker()
        self.camera = LatestColorFrame(
            WILOR_CAMERA_INDEX, WILOR_WIDTH, WILOR_HEIGHT, WILOR_FPS
        )
        self.sequence = self.frame_index = 0
        print(f"WiLoR D435 {self.camera.serial}: {WILOR_WIDTH}x{WILOR_HEIGHT}@{WILOR_FPS}; CUDA")

    def read(self):
        self.sequence, frame, timestamp = self.camera.read(self.sequence)
        detect = self.frame_index % WILOR_DETECT_EVERY == 0 or self.tracker.track is None
        tracks = self.tracker.update(self.detector(frame) if detect else None)
        self.frame_index += 1
        if not tracks:
            image = preview(frame, tracks)
            return HandFrame(timestamp, None, None, False, "WILOR WAITING",
                             finger_pad_directions=None, preview=image)
        try:
            vertices, translations = self.reconstructor(frame, tracks)
            points, directions, handedness = mesh_hand(
                vertices[0], tracks[0].handedness, self.regressor, self.faces
            )
        except ValueError as error:
            image = preview(frame, tracks)
            return HandFrame(timestamp, None, None, False, f"WILOR INVALID: {error}",
                             finger_pad_directions=None, preview=image)
        image = preview(frame, tracks, (
            vertices[0], translations[0], tracks[0].handedness, self.faces
        ))
        return HandFrame(
            timestamp, points, handedness, True,
            f"WILOR {handedness} {tracks[0].score:.2f}",
            finger_pad_directions=directions, preview=image,
        )

    def close(self):
        self.camera.close()
