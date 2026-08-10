import argparse
import time

import numpy as np

from camera import RealSenseCamera, StereoCamera
from config import CAMERA_INDEX, CAMERA_TYPE
from hand_core import StereoProcessor
from retarget import Retargeter, RetargetWorker
from ros import RosOutput
from viewer import Viewer


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ros", action="store_true")
    return parser.parse_args(argv)


def _capture():
    if CAMERA_TYPE == "d435":
        camera = RealSenseCamera(CAMERA_INDEX)
        params = camera.params
    elif CAMERA_TYPE == "stereo":
        camera, params = StereoCamera(CAMERA_INDEX), None
    else:
        raise ValueError("CAMERA_TYPE must be 'd435' or 'stereo'")
    try:
        return camera, StereoProcessor(params)
    except Exception:
        camera.close()
        raise


def main():
    args = _parse_args()
    retargeter = Retargeter()
    camera, processor = _capture()
    viewer = Viewer(retargeter.model)
    ros = RosOutput() if args.ros else None
    worker = RetargetWorker(retargeter)
    last_timestamp = None
    try:
        while True:
            ok, left, right, timestamp = camera.read()
            if not ok or timestamp == last_timestamp:
                time.sleep(0.002)
                continue
            last_timestamp = timestamp
            result = processor.process(left, right, timestamp)
            valid = result["found"] and not result["stale"] and result["keypoint_relative"] is not None
            if valid and ros is not None:
                ros.points(result["keypoint_relative"], result["handedness"])
            if valid and result["handedness"] == "Left" and result["phase"].startswith("GESTURE"):
                worker.submit(result["keypoint_relative"], timestamp)
            else:
                worker.pause()
            output = worker.poll()
            robot, losses = (None, None) if output is None else output
            if ros is not None and robot is not None:
                ros.joints(np.degrees(robot))
            viewer.update(result, robot, losses)
    except KeyboardInterrupt:
        pass
    finally:
        worker.close()
        camera.close()
        processor.close()
        viewer.close()
        if ros is not None:
            ros.close()


if __name__ == "__main__":
    main()
