import json
import time

import cv2
import numpy as np

from config import (
    BOARD_SIZE,
    CAMERA_INDEX,
    INPUT_SOURCE,
    MIN_CALIBRATION_PAIRS,
    PARAMS_PATH,
    SQUARE_SIZE,
)
from .camera import StereoCamera


def collect():
    grid = np.zeros((np.prod(BOARD_SIZE), 3), np.float32)
    grid[:, :2] = np.mgrid[: BOARD_SIZE[0], : BOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE
    objects, lefts, rights, size = [], [], [], None
    camera = StereoCamera(CAMERA_INDEX)
    criteria = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001
    print(f"棋盘格内角点 {BOARD_SIZE}，格长 {SQUARE_SIZE} mm；C 采样，Q 完成。")
    time.sleep(1)
    try:
        while True:
            ok, left, right, _ = camera.read()
            if not ok:
                continue
            size = left.shape[1], left.shape[0]
            gray = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in (left, right)]
            found = [cv2.findChessboardCorners(image, BOARD_SIZE) for image in gray]
            views = []
            for image, (ready, corners) in zip((left, right), found):
                view = image.copy()
                if ready:
                    cv2.drawChessboardCorners(view, BOARD_SIZE, corners, True)
                views.append(view)
            ready = found[0][0] and found[1][0]
            view = np.hstack(views)
            cv2.putText(
                view,
                f"Pairs: {len(objects)}  {'READY' if ready else 'Searching'}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0) if ready else (0, 0, 255),
                2,
            )
            cv2.imshow("Stereo calibration", cv2.resize(view, None, fx=0.5, fy=0.5))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c") and ready:
                objects.append(grid.copy())
                lefts.append(cv2.cornerSubPix(gray[0], found[0][1], (11, 11), (-1, -1), criteria))
                rights.append(cv2.cornerSubPix(gray[1], found[1][1], (11, 11), (-1, -1), criteria))
                print(f"已采集 {len(objects)} 对")
    finally:
        camera.close()
        cv2.destroyAllWindows()
    return objects, lefts, rights, size


def solve(objects, lefts, rights, size):
    left_rms, k1, d1, _, _ = cv2.calibrateCamera(objects, lefts, size, None, None)
    right_rms, k2, d2, _, _ = cv2.calibrateCamera(objects, rights, size, None, None)
    result = cv2.stereoCalibrate(
        objects,
        lefts,
        rights,
        k1,
        d1,
        k2,
        d2,
        size,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
        flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3,
    )
    rms, k1, d1, k2, d2, rotation, translation = result[:7]
    print(
        f"单目 RMS: {left_rms:.4f}, {right_rms:.4f}；"
        f"双目 RMS: {rms:.4f}；基线: {np.linalg.norm(translation):.2f} mm"
    )
    return {
        "K1": k1.tolist(),
        "D1": d1.tolist(),
        "K2": k2.tolist(),
        "D2": d2.tolist(),
        "R": rotation.tolist(),
        "T": translation.tolist(),
        "rms": float(rms),
        "rotated_width": size[0],
        "rotated_height": size[1],
    }


def main():
    if INPUT_SOURCE != "stereo":
        raise RuntimeError("Set INPUT_SOURCE='stereo' in config.py before calibration")
    objects, lefts, rights, size = collect()
    if len(objects) < MIN_CALIBRATION_PAIRS:
        raise RuntimeError(f"至少需要 {MIN_CALIBRATION_PAIRS} 对有效棋盘格图像")
    PARAMS_PATH.write_text(json.dumps(solve(objects, lefts, rights, size), indent=2))
    print(f"已保存到 {PARAMS_PATH}")


if __name__ == "__main__":
    main()
