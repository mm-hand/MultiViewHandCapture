import time

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

from config import CAMERA_INDEX
from hand_core import Camera, StereoProcessor


class Viewer:
    def __init__(self):
        plt.ion()
        self.edges = list(mp.solutions.hands.HAND_CONNECTIONS)
        self.figure = plt.figure(figsize=(18, 10))
        layout = gridspec.GridSpec(2, 3, height_ratios=(1, 2))
        self.image_axes = [self.figure.add_subplot(layout[0, i]) for i in (0, 1)]
        self.images, self.dots_2d, self.lines_2d = [], [], []
        for axis, title in zip(self.image_axes, ("Left camera", "Right camera")):
            axis.set_title(title)
            axis.axis("off")
            self.images.append(axis.imshow(np.zeros((360, 640, 3), np.uint8)))
            self.dots_2d.append(axis.plot([], [], "ro", ms=3)[0])
            self.lines_2d.append([axis.plot([], [], "g-", lw=1)[0] for _ in self.edges])

        info = self.figure.add_subplot(layout[0, 2])
        info.axis("off")
        info.text(0.5, 0.5, "Close this window or press Ctrl+C to exit", ha="center")
        self.axes_3d = [self.figure.add_subplot(layout[1, i], projection="3d") for i in range(3)]
        self.dots_3d, self.lines_3d = [], []
        for i, (axis, title, view) in enumerate(
            zip(self.axes_3d, ("Absolute front", "Absolute side", "Palm relative"), ((-90, -90), (0, 0), (0, 45)))
        ):
            axis.set_title(title)
            axis.view_init(*view)
            limits = ((-150, 150), (-150, 150), (100, 500)) if i < 2 else ((-0.12, 0.12), (-0.12, 0.12), (-0.04, 0.16))
            axis.set(xlim=limits[0], ylim=limits[1], zlim=limits[2])
            self.dots_3d.append(axis.scatter([], [], [], c="r", s=25))
            self.lines_3d.append([axis.plot([], [], [], "b-", lw=1)[0] for _ in self.edges])

    def _image(self, index, image, points):
        if image is None:
            return
        height, width = image.shape[:2]
        image = cv2.resize(image, (640, 360))
        self.images[index].set_data(image)
        points = None if points is None else points * (640 / width, 360 / height)
        self.dots_2d[index].set_data([], [])
        if points is not None:
            self.dots_2d[index].set_data(points[:, 0], points[:, 1])
        for line, edge in zip(self.lines_2d[index], self.edges):
            xy = ([], []) if points is None else points[list(edge)].T
            line.set_data(*xy)

    def _skeleton(self, index, points):
        xyz = ([], [], []) if points is None else points.T
        self.dots_3d[index]._offsets3d = xyz
        for line, edge in zip(self.lines_3d[index], self.edges):
            if points is None:
                line.set_data([], [])
                line.set_3d_properties([])
            else:
                line.set_data(points[list(edge), 0], points[list(edge), 1])
                line.set_3d_properties(points[list(edge), 2])

    def update(self, result):
        quality = result["quality"]
        rejected = quality["rejected_reason"]
        suffix = f" | rejected: {rejected}" if rejected else ""
        self.figure.suptitle(f"{result['phase']}{suffix}")
        self._image(0, result["image_left"], result["px_left"])
        self._image(1, result["image_right"], result["px_right"])
        self._skeleton(0, result["keypoint_absolute"])
        self._skeleton(1, result["keypoint_absolute"])
        self._skeleton(2, result["keypoint_relative"])
        plt.pause(0.001)


def main():
    camera, processor, viewer = Camera(CAMERA_INDEX), StereoProcessor(), Viewer()
    try:
        while plt.fignum_exists(viewer.figure.number):
            ok, frame = camera.read()
            if ok:
                viewer.update(processor.process_frame(frame))
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        processor.close()
        plt.close("all")


if __name__ == "__main__":
    main()
