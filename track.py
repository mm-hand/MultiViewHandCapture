import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import threading
import os
import sys
import time
import json
from joint_constraints import apply_joint_constraints
from config import CAMERA_INDEX, FULL_WIDTH, HEIGHT, CALIBRATION_FRAMES

# =================== 0. Load Stereo Calibration Parameters ===================
JSON_PATH = "stereo_params.json"

if not os.path.exists(JSON_PATH):
    print(f"❌ ERROR: Calibration file not found {JSON_PATH}")
    print("Please run calibrate.py first!")
    sys.exit()

print(f"✅ Loading calibration file: {JSON_PATH}")
with open(JSON_PATH, 'r') as f:
    params = json.load(f)

# Convert list to numpy arrays
K1 = np.array(params["K1"], dtype=np.float64)
D1 = np.array(params["D1"], dtype=np.float64)
K2 = np.array(params["K2"], dtype=np.float64)
D2 = np.array(params["D2"], dtype=np.float64)
R  = np.array(params["R"],  dtype=np.float64)
T  = np.array(params["T"],  dtype=np.float64)

# Projection matrices
P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = K2 @ np.hstack((R, T))

print("Stereo calibration parameters loaded.")

# =================== 1. One Euro Filter Class ===================
class OneEuroFilter:
    """Smooths noisy data while keeping responsiveness."""
    def __init__(self, freq, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        if freq <= 0:
            raise ValueError("Frequency must be positive.")
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = None
        self.is_init = False
        self.t_e = 1.0 / self.freq

    def _smoothing_factor(self, cutoff):
        r = 2 * np.pi * cutoff * self.t_e
        return r / (r + 1)

    def _exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, x):
        if not self.is_init:
            self.is_init = True
            self.x_prev = x
            self.dx_prev = np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0
            return x.copy() if isinstance(x, np.ndarray) else x

        dx = (x - self.x_prev) / self.t_e
        a_d = self._smoothing_factor(self.d_cutoff)
        dx_hat = self._exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._smoothing_factor(cutoff)
        x_hat = self._exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

    def reset(self):
        """Reset filter state."""
        self.x_prev = None
        self.dx_prev = None
        self.is_init = False

# =================== 2. Threaded Camera Reader ===================
class CameraStream:
    """Threaded stereo camera capture to reduce latency."""
    def __init__(self, src=CAMERA_INDEX, width=FULL_WIDTH, height=HEIGHT):
        self.stream = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.stream.isOpened():
            print(f"❌ Unable to open camera {src}, trying ID 1...")
            self.stream.release()
            self.stream = cv2.VideoCapture(1, cv2.CAP_V4L2)
            self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.stream.isOpened():
            print("❌ FATAL: No camera detected.")
            sys.exit()

        print(f"✅ Camera started: {int(self.stream.get(3))}x{int(self.stream.get(4))}")
        self.grabbed, self.frame = self.stream.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update)
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            try:
                grabbed, frame = self.stream.read()
                if grabbed and frame is not None:
                    with self.read_lock:
                        self.grabbed = grabbed
                        self.frame = frame
                else:
                    time.sleep(0.01)
            except:
                pass

    def read(self):
        with self.read_lock:
            return self.grabbed, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.started = False
        if self.thread.is_alive():
            self.thread.join()
        self.stream.release()

# =================== 3. Visualization Class ===================
class HandVisualizerAllInOne:
    """
    Displays stereo camera frames and 3D hand reconstruction.
    Shows calibration/tracking status in window title.
    """
    def __init__(self, w, h):
        plt.ion()
        self.fig = plt.figure(figsize=(16, 12))
        self.w = w
        self.h = h
        self.conn = list(mp.solutions.hands.HAND_CONNECTIONS)

        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 3], hspace=0.1, wspace=0.1)

        # Left 2D view
        self.ax_l = self.fig.add_subplot(gs[0, 0])
        self.ax_l.axis('off')
        self.im_l_disp = self.ax_l.imshow(np.zeros((h, w, 3), dtype=np.uint8))
        self.lines_2d_l = [self.ax_l.plot([], [], 'g-')[0] for _ in self.conn]
        self.points_2d_l = self.ax_l.plot([], [], 'r.')[0]

        # Right 2D view
        self.ax_r = self.fig.add_subplot(gs[0, 1])
        self.ax_r.axis('off')
        self.im_r_disp = self.ax_r.imshow(np.zeros((h, w, 3), dtype=np.uint8))
        self.lines_2d_r = [self.ax_r.plot([], [], 'g-')[0] for _ in self.conn]
        self.points_2d_r = self.ax_r.plot([], [], 'r.')[0]

        # Front 3D view
        self.ax3d_front = self.fig.add_subplot(gs[1, 0], projection='3d')
        self.ax3d_front.view_init(-90, -90)
        self._init_3d_axis(self.ax3d_front)

        # Side 3D view
        self.ax3d_side = self.fig.add_subplot(gs[1, 1], projection='3d')
        self.ax3d_side.view_init(0, 0)
        self._init_3d_axis(self.ax3d_side)

        self.scats_3d = []
        self.lines_3d_collections = []
        for ax in [self.ax3d_front, self.ax3d_side]:
            scat = ax.scatter([], [], [], c='r', s=40)
            lines = [ax.plot([], [], [], 'b-')[0] for _ in self.conn]
            self.scats_3d.append(scat)
            self.lines_3d_collections.append(lines)

        self.fig.tight_layout()

    def _init_3d_axis(self, ax):
        ax.set_xlim(-150, 150)
        ax.set_ylim(-150, 150)
        ax.set_zlim(100, 400)

    def set_status(self, phase, current=0, total=0):
        """
        Update window title text.
        """
        if phase.upper() == "CALIBRATION":
            self.fig.suptitle(f"Stereo Hand Tracking - Calibration ({current}/{total} images needed)", fontsize=14)
        else:
            self.fig.suptitle("Stereo Hand Tracking - Gesture Tracking", fontsize=14)

    def update(self, img_l, img_r, pts_l, pts_r, pts3d):
        try:
            self.im_l_disp.set_data(cv2.resize(img_l, (640, 360)))
            self.im_r_disp.set_data(cv2.resize(img_r, (640, 360)))
            self._update2d(self.lines_2d_l, self.points_2d_l, pts_l)
            self._update2d(self.lines_2d_r, self.points_2d_r, pts_r)
            if pts3d is not None:
                for i in range(2):
                    self.scats_3d[i]._offsets3d = (pts3d[:, 0], pts3d[:, 1], pts3d[:, 2])
                    for line, (s, e) in zip(self.lines_3d_collections[i], self.conn):
                        line.set_data([pts3d[s, 0], pts3d[e, 0]],
                                      [pts3d[s, 1], pts3d[e, 1]])
                        line.set_3d_properties([pts3d[s, 2], pts3d[e, 2]])
            plt.pause(0.001)
        except:
            pass

    def _update2d(self, lines, points, pts):
        if pts is not None:
            points.set_data(pts[:, 0], pts[:, 1])
            for line, (s, e) in zip(lines, self.conn):
                line.set_data([pts[s, 0], pts[e, 0]],
                              [pts[s, 1], pts[e, 1]])
        else:
            points.set_data([], [])
            for line in lines:
                line.set_data([], [])

# =================== 4. Hand Processor ===================
class HandProcessor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands.Hands(max_num_hands=1,
                                                 min_detection_confidence=0.4,
                                                 min_tracking_confidence=0.4)
        self.filter = OneEuroFilter(freq=30, min_cutoff=0.1, beta=5.0, d_cutoff=1.0)

    def process(self, img):
        res = self.mp_hands.process(img)
        if res.multi_hand_landmarks:
            raw = np.array([[lm.x, lm.y] for lm in res.multi_hand_landmarks[0].landmark])
            return self.filter(raw)
        else:
            self.filter.reset()
            return None

# =================== 5. Calibration & Correction Functions ===================
fingers = {
    'thumb':  [0, 1, 2, 3, 4],
    'index':  [0, 5, 6, 7, 8],
    'middle': [0, 9, 10, 11, 12],
    'ring':   [0, 13, 14, 15, 16],
    'pinky':  [0, 17, 18, 19, 20]
}

def calibrate_lengths(accum, pts3d):
    for chain in fingers.values():
        for i in range(len(chain) - 1):
            s, e = chain[i], chain[i + 1]
            accum[(s, e)].append(np.linalg.norm(pts3d[e] - pts3d[s]))

def apply_chain_correction(pts3d, lengths):
    corrected = pts3d.copy()
    for chain in fingers.values():
        for i in range(len(chain) - 1):
            s = chain[i]
            e = chain[i + 1]
            v = corrected[e] - corrected[s]
            curr_len = np.linalg.norm(v)
            if curr_len > 1e-6:
                dir = v / curr_len
                corrected[e] = corrected[s] + dir * lengths[(s, e)]
    return corrected

# =================== 6. Main Loop ===================
def main():
    cam = CameraStream(src=0, width=FULL_WIDTH, height=HEIGHT).start()
    time.sleep(1.0)
    proc_l, proc_r = HandProcessor(), HandProcessor()

    W_RAW, H_RAW = FULL_WIDTH // 2, HEIGHT
    visualizer = HandVisualizerAllInOne(w=W_RAW, h=H_RAW)
    calib_counts = 0
    bone_accum = {b: [] for f in fingers.values() for b in zip(f[:-1], f[1:])}
    bone_lengths_final = {}
    calibration_done = False

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue
        if frame.shape[1] != FULL_WIDTH:
            continue

        img_l, img_r = frame[:, :W_RAW], frame[:, W_RAW:]
        img_l_rgb = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
        img_r_rgb = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)

        # pts_norm 现在包含 (x, y, z)，其中 z 是 MediaPipe 相对深度
        pts_norm_l = proc_l.process(img_l_rgb)
        pts_norm_r = proc_r.process(img_r_rgb)
        px_l = px_r = pts3d = None

        if pts_norm_l is not None:
            px_l = np.column_stack((pts_norm_l[:, 0] * W_RAW, pts_norm_l[:, 1] * H_RAW))
        if pts_norm_r is not None:
            px_r = np.column_stack((pts_norm_r[:, 0] * W_RAW, pts_norm_r[:, 1] * H_RAW))

        if px_l is not None and px_r is not None:
            # Undistort and triangulate
            ud_l = cv2.undistortPoints(px_l.reshape(-1, 1, 2), K1, D1, P=K1)
            ud_r = cv2.undistortPoints(px_r.reshape(-1, 1, 2), K2, D2, P=K2)
            pts_4d = cv2.triangulatePoints(P1, P2,
                                           ud_l.reshape(-1, 2).T,
                                           ud_r.reshape(-1, 2).T)
            pts3d = (pts_4d[:3] / pts_4d[3]).T

            if not calibration_done:
                visualizer.set_status("CALIBRATION", calib_counts, CALIBRATION_FRAMES)
                calibrate_lengths(bone_accum, pts3d)
                calib_counts += 1
                if calib_counts >= CALIBRATION_FRAMES:
                    bone_lengths_final = {b: np.mean(v) for b, v in bone_accum.items()}
                    calibration_done = True
                    print("✅ Calibration finished.")
            else:
                visualizer.set_status("GESTURE TRACKING")
                # Step 1: Correct bone lengths
                pts3d = apply_chain_correction(pts3d, bone_lengths_final)
                # Step 2: Apply joint angle constraints
                pts3d = apply_joint_constraints(pts3d)

        visualizer.update(img_l_rgb, img_r_rgb, px_l, px_r, pts3d)

        if not plt.fignum_exists(visualizer.fig.number):
            break

    cam.stop()
    plt.close()

if __name__ == "__main__":
    main()
