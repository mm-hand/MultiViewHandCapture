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

# Try/Except imports to support both package execution and direct script execution
from MultiViewHandCapture.joint_constraints import apply_joint_constraints
from MultiViewHandCapture.config import CAMERA_INDEX, FULL_WIDTH, HEIGHT, CALIBRATION_FRAMES

# =================== 1. One Euro Filter Class (Unchanged) ===================
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
        self.x_prev = None
        self.dx_prev = None
        self.is_init = False

# =================== 2. Threaded Camera Reader (Unchanged Logic) ===================
class CameraStream:
    """Threaded stereo camera capture to reduce latency."""
    def __init__(self, src, width, height):
        # Prefer V4L2 on Linux
        backend = cv2.CAP_V4L2 if os.name == 'posix' else cv2.CAP_ANY
        self.stream = cv2.VideoCapture(src, backend)
        
        # Optimize MJPG and Buffer
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Fallback mechanism
        if not self.stream.isOpened():
            print(f"❌ Unable to open camera {src}, trying ID 1...")
            self.stream.release()
            self.stream = cv2.VideoCapture(1, backend)
            self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.stream.isOpened():
            raise RuntimeError("FATAL: No camera detected.")

        actual_w = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        print(f"✅ Camera started: {int(actual_w)}x{int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        
        self.grabbed, self.frame = self.stream.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, daemon=True)
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
                    time.sleep(0.005)
            except Exception:
                pass

    def read(self):
        with self.read_lock:
            if not self.grabbed or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self):
        self.started = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join()
        self.stream.release()

# =================== 3. Hand Processor (Unchanged) ===================
class HandProcessor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands.Hands(max_num_hands=1,
                                                 min_detection_confidence=0.4,
                                                 min_tracking_confidence=0.4)
        self.filter = OneEuroFilter(freq=30, min_cutoff=0.1, beta=5.0, d_cutoff=1.0)

    def process(self, img):
        # MediaPipe expects RGB
        res = self.mp_hands.process(img)
        if res.multi_hand_landmarks:
            raw = np.array([[lm.x, lm.y] for lm in res.multi_hand_landmarks[0].landmark])
            return self.filter(raw)
        else:
            self.filter.reset()
            return None

# =================== 4. Helper Dicts & Functions ===================
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
            dist = np.linalg.norm(pts3d[e] - pts3d[s])
            accum[(s, e)].append(dist)

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
                # Correct the position of child node based on parent node
                corrected[e] = corrected[s] + dir * lengths[(s, e)]
    return corrected

# =================== 5. Main Tracker Logic Class ===================
class StereoHandTracker:
    """
    Encapsulates the entire Stereo Hand Tracking logic.
    Can be used by external applications (like GeoRT) or the local main loop.
    """
    def __init__(self, camera_index=None):
        if camera_index is None:
            camera_index = CAMERA_INDEX

        # 1. Load Calibration
        self._load_calibration()
        
        # 2. Init Camera
        self.full_width = FULL_WIDTH
        self.height = HEIGHT
        self.single_width = FULL_WIDTH // 2
        self.cam = CameraStream(src=camera_index, width=self.full_width, height=self.height)
        self.cam.start()

        # 3. Init Processors
        self.proc_l = HandProcessor()
        self.proc_r = HandProcessor()

        # 4. Calibration State
        self.calib_counts = 0
        self.calib_frames_total = CALIBRATION_FRAMES
        self.bone_accum = {b: [] for f in fingers.values() for b in zip(f[:-1], f[1:])}
        self.bone_lengths_final = {}
        self.calibration_done = False
        
        print("[Tracker] Initialization Complete. Waiting for camera warmup...")
        time.sleep(1.0)

    def _load_calibration(self):
        # Locate json relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, "stereo_params.json")

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"❌ ERROR: Calibration file not found at {json_path}")

        print(f"✅ Loading parameters from {json_path}")
        with open(json_path, 'r') as f:
            params = json.load(f)

        # Matrices
        self.K1 = np.array(params["K1"], dtype=np.float64)
        self.D1 = np.array(params["D1"], dtype=np.float64)
        self.K2 = np.array(params["K2"], dtype=np.float64)
        self.D2 = np.array(params["D2"], dtype=np.float64)
        R  = np.array(params["R"],  dtype=np.float64)
        T  = np.array(params["T"],  dtype=np.float64)

        # Compute Projection (Rectification) Matrices
        # Simple projection: K * [I|0] (Assuming no rectification needed for simple P, 
        # but usually we need stereoRectify for ideal results. 
        # Keeping logic ORIGINAL as requested: P = K * [R|T])
        self.P1 = self.K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
        self.P2 = self.K2 @ np.hstack((R, T))
        print("[Tracker] Calibration loaded.")

    def step(self):
        """
        Runs one iteration of tracking.
        Returns:
            processed_data (dict): contains 'found', 'joints', 'status_phase', 'debug_images'
        """
        ret, frame = self.cam.read()
        
        output = {
            "found": False,
            "joints": None,      # (21, 3) in mm
            "image_left": None,  # RGB
            "image_right": None, # RGB
            "px_left": None,     # (21, 2)
            "px_right": None,    # (21, 2)
            "phase": "WAITING"   # CALIBRATION or TRACKING
        }

        if not ret or frame is None:
            return output
        
        if frame.shape[1] != self.full_width:
             # Skip invalid frame
            return output

        # Split images
        img_l = frame[:, :self.single_width]
        img_r = frame[:, self.single_width:]
        
        # Convert to RGB
        img_l_rgb = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
        img_r_rgb = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
        
        output["image_left"] = img_l_rgb
        output["image_right"] = img_r_rgb

        # Process MediaPipe
        # Returns normalized coordinates (x, y, z_rel)
        pts_norm_l = self.proc_l.process(img_l_rgb)
        pts_norm_r = self.proc_r.process(img_r_rgb)
        
        px_l = None
        px_r = None

        # Convert normalized to pixel coordinates for triangulation
        if pts_norm_l is not None:
            px_l = np.column_stack((pts_norm_l[:, 0] * self.single_width, pts_norm_l[:, 1] * self.height))
        if pts_norm_r is not None:
            px_r = np.column_stack((pts_norm_r[:, 0] * self.single_width, pts_norm_r[:, 1] * self.height))

        output["px_left"] = px_l
        output["px_right"] = px_r

        # Triangulation Logic
        if px_l is not None and px_r is not None:
            # 1. Undistort points
            # using K as P argument ensures points are normalized to camera matrix
            ud_l = cv2.undistortPoints(px_l.reshape(-1, 1, 2), self.K1, self.D1, P=self.K1)
            ud_r = cv2.undistortPoints(px_r.reshape(-1, 1, 2), self.K2, self.D2, P=self.K2)
            
            # 2. Triangulate
            pts_4d = cv2.triangulatePoints(self.P1, self.P2,
                                           ud_l.reshape(-1, 2).T,
                                           ud_r.reshape(-1, 2).T)
            
            # 3. Convert from homogeneous
            pts3d = (pts_4d[:3] / pts_4d[3]).T # Shape (21, 3)

            # 4. Calibration / Correction Phase
            if not self.calibration_done:
                output["phase"] = f"CALIBRATION ({self.calib_counts}/{self.calib_frames_total})"
                
                # Accumulate lengths
                calibrate_lengths(self.bone_accum, pts3d)
                self.calib_counts += 1
                
                if self.calib_counts >= self.calib_frames_total:
                    # Finalize calibration
                    self.bone_lengths_final = {b: np.mean(v) for b, v in self.bone_accum.items()}
                    self.calibration_done = True
                    print("✅ Calibration finished. Switching to Tracking Mode.")
            else:
                output["phase"] = "GESTURE TRACKING"
                
                # Apply length constraints (Bone Correction)
                pts3d = apply_chain_correction(pts3d, self.bone_lengths_final)
                
                # Apply joint angle constraints (Kinematics)
                pts3d = apply_joint_constraints(pts3d)

            output["found"] = True
            output["joints"] = pts3d
            
        return output

    def close(self):
        self.cam.stop()

# =================== 6. Visualizer (Original Logic kept) ===================
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

    def set_status(self, text):
        self.fig.suptitle(f"Stereo Hand Tracking - {text}", fontsize=14)

    def update(self, img_l, img_r, pts_l, pts_r, pts3d):
        try:
            # Resize for display performance if needed, currently 640x360 fixed in original code
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
        except Exception:
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

# =================== 7. Main Loop (Entry Point) ===================
def main():
    # Instantiate the tracker logic
    tracker = StereoHandTracker()
    
    # Initialize UI
    W_RAW = tracker.single_width
    H_RAW = tracker.height
    visualizer = HandVisualizerAllInOne(w=W_RAW, h=H_RAW)

    print("Hit simple loop. Ctrl+C to exit.")

    try:
        while True:
            # Core logic step
            data = tracker.step()
            
            if data["image_left"] is None:
                time.sleep(0.01)
                continue
            
            # Update Title
            visualizer.set_status(data["phase"])

            # Update Plot
            visualizer.update(
                data["image_left"], 
                data["image_right"], 
                data["px_left"], 
                data["px_right"], 
                data["joints"]
            )

            if not plt.fignum_exists(visualizer.fig.number):
                print("Window closed.")
                break

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        tracker.close()
        plt.close()

if __name__ == "__main__":
    main()