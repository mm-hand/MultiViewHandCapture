import argparse
import time

import numpy as np

from config import CAMERA_TYPE
from retarget import Retargeter, RetargetWorker
from ros import RosOutput
from viewer import Viewer


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("vision", "manus", "wilor"), default="vision")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--ros", action="store_true")
    output.add_argument("--sim", action="store_true")
    return parser.parse_args(argv)


def _source(source_name):
    if source_name == "vision":
        from vision.source import VisionSource

        return VisionSource(CAMERA_TYPE)
    if source_name == "manus":
        from manus.source import ManusSource

        return ManusSource()
    if source_name == "wilor":
        from wilor.source import WilorSource

        return WilorSource()
    raise ValueError(f"Unsupported source: {source_name}")


def main():
    args = _parse_args()
    retargeter = Retargeter()
    source = _source(args.source)
    viewer = Viewer(retargeter.model)
    ros = RosOutput() if args.ros else None
    worker = RetargetWorker(retargeter)
    simulation = None
    try:
        if args.sim:
            from simulation import GraspSimulation

            simulation = GraspSimulation()
        while True:
            frame = source.read()
            if frame is None:
                if simulation is not None and not simulation.update(None):
                    break
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
            if simulation is not None and not simulation.update(robot):
                break
    except KeyboardInterrupt:
        pass
    finally:
        if simulation is not None:
            simulation.close()
        worker.close()
        source.close()
        viewer.close()
        if ros is not None:
            ros.close()


if __name__ == "__main__":
    main()
