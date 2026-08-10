import argparse
import time

import numpy as np

from config import INPUT_SOURCE
from retarget import Retargeter, RetargetWorker
from ros import RosOutput
from viewer import Viewer


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ros", action="store_true")
    return parser.parse_args(argv)


def _source():
    if INPUT_SOURCE in ("stereo", "d435"):
        from vision.source import VisionSource

        return VisionSource(INPUT_SOURCE)
    raise ValueError(f"Unsupported INPUT_SOURCE: {INPUT_SOURCE}")


def main():
    args = _parse_args()
    retargeter = Retargeter()
    source = _source()
    viewer = Viewer(retargeter.model)
    ros = RosOutput() if args.ros else None
    worker = RetargetWorker(retargeter)
    try:
        while True:
            frame = source.read()
            if frame is None:
                time.sleep(0.002)
                continue
            if frame.points is not None and ros is not None:
                ros.points(frame.points, frame.handedness)
            if frame.ready and frame.handedness == "Left":
                worker.submit(frame.points, frame.timestamp)
            else:
                worker.pause()
            output = worker.poll()
            robot, losses = (None, None) if output is None else output
            if ros is not None and robot is not None:
                ros.joints(np.degrees(robot))
            viewer.update(frame, robot, losses)
    except KeyboardInterrupt:
        pass
    finally:
        worker.close()
        source.close()
        viewer.close()
        if ros is not None:
            ros.close()


if __name__ == "__main__":
    main()
