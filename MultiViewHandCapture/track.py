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

from MultiViewHandCapture.camera import CameraStream, rotate_image
from MultiViewHandCapture.config import CAMERA_INDEX, FULL_WIDTH, HEIGHT, CALIBRATION_FRAMES, ROTATE_LEFT, ROTATE_RIGHT

# One Euro Filter for smoothing landmarks
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

# Hand Processor with MediaPipe
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

# Finger bone structure definitions
fingers = {
    'thumb':  [0, 1, 2, 3, 4],
    'index':  [0, 5, 6, 7, 8],
    'middle': [0, 9, 10, 11, 12],
    'ring':   [0, 13, 14, 15, 16],
    'pinky':  [0, 17, 18, 19, 20]
}

def calibrate_lengths(accum, pts3d):
    """Accumulate bone lengths during calibration phase"""
    for chain in fingers.values():
        for i in range(len(chain) - 1):
            s, e = chain[i], chain[i + 1]
            dist = np.linalg.norm(pts3d[e] - pts3d[s])
            accum[(s, e)].append(dist)

def apply_chain_correction(pts3d, lengths):
    """Apply bone length constraints to 3D points"""
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

# Main Stereo Hand Tracker
class StereoHandTracker:
    """Real-time stereo hand tracking with calibration and constraints"""
    def __init__(self, camera_index=None):
        if camera_index is None:
            camera_index = CAMERA_INDEX

        # Load calibration parameters
        self._load_calibration()
        
        # Initialize camera with rotated dimensions
        self.full_width = FULL_WIDTH
        self.height = HEIGHT
        self.single_width = FULL_WIDTH // 2
        
        # Calculate rotated image dimensions
        test_img = np.zeros((self.height, self.single_width, 3), dtype=np.uint8)
        _, self.rotated_width, self.rotated_height = rotate_image(test_img, ROTATE_LEFT)
        
        self.cam = CameraStream(src=camera_index, width=self.full_width, height=self.height)
        self.cam.start()

        # Initialize hand processors
        self.proc_l = HandProcessor()
        self.proc_r = HandProcessor()

        # Calibration state
        self.calib_counts = 0
        self.calib_frames_total = CALIBRATION_FRAMES
        self.bone_accum = {b: [] for f in fingers.values() for b in zip(f[:-1], f[1:])}
        self.bone_lengths_final = {}
        self.calibration_done = False
        
        print("[Tracker] Initialization Complete. Waiting for camera warmup...")
        time.sleep(1.0)

    def _load_calibration(self):
        """Load stereo calibration parameters"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, "stereo_params.json")

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Calibration file not found at {json_path}")

        print(f"Loading parameters from {json_path}")
        with open(json_path, 'r') as f:
            params = json.load(f)

        # Load calibration matrices
        self.K1 = np.array(params["K1"], dtype=np.float64)
        self.D1 = np.array(params["D1"], dtype=np.float64)
        self.K2 = np.array(params["K2"], dtype=np.float64)
        self.D2 = np.array(params["D2"], dtype=np.float64)
        R  = np.array(params["R"],  dtype=np.float64)
        T  = np.array(params["T"],  dtype=np.float64)

        # Use rotated image dimensions from calibration
        if "rotated_width" in params and "rotated_height" in params:
            self.rotated_width = params["rotated_width"]
            self.rotated_height = params["rotated_height"]
        else:
            # Fallback: calculate from current rotation settings
            test_img = np.zeros((HEIGHT, FULL_WIDTH // 2, 3), dtype=np.uint8)
            _, self.rotated_width, self.rotated_height = rotate_image(test_img, ROTATE_LEFT)

        # Compute projection matrices
        self.P1 = self.K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
        self.P2 = self.K2 @ np.hstack((R, T))
        
        print(f"[Tracker] Calibration loaded. Rotated dimensions: {self.rotated_width}x{self.rotated_height}")

    def step(self):
        """Process one frame of stereo hand tracking"""
        ret, frame = self.cam.read()
        
        output = {
            "found": False,
            "joints": None,
            "image_left": None,
            "image_right": None,
            "px_left": None,
            "px_right": None,
            "phase": "WAITING"
        }

        if not ret or frame is None:
            return output
        
        if frame.shape[1] != self.full_width:
            return output

        # Split and rotate images
        img_l = frame[:, :self.single_width]
        img_r = frame[:, self.single_width:]
        
        img_l_rotated, _, _ = rotate_image(img_l, ROTATE_LEFT)
        img_r_rotated, _, _ = rotate_image(img_r, ROTATE_RIGHT)
        
        # Convert to RGB for MediaPipe
        img_l_rgb = cv2.cvtColor(img_l_rotated, cv2.COLOR_BGR2RGB)
        img_r_rgb = cv2.cvtColor(img_r_rotated, cv2.COLOR_BGR2RGB)
        
        output["image_left"] = img_l_rgb
        output["image_right"] = img_r_rgb

        # Process with MediaPipe
        pts_norm_l = self.proc_l.process(img_l_rgb)
        pts_norm_r = self.proc_r.process(img_r_rgb)
        
        px_l = None
        px_r = None

        # Convert normalized coordinates to pixels using rotated dimensions
        if pts_norm_l is not None:
            px_l = np.column_stack((pts_norm_l[:, 0] * self.rotated_width, 
                                   pts_norm_l[:, 1] * self.rotated_height))
        if pts_norm_r is not None:
            px_r = np.column_stack((pts_norm_r[:, 0] * self.rotated_width, 
                                   pts_norm_r[:, 1] * self.rotated_height))

        output["px_left"] = px_l
        output["px_right"] = px_r

        # Triangulate 3D points
        if px_l is not None and px_r is not None:
            # Undistort points
            ud_l = cv2.undistortPoints(px_l.reshape(-1, 1, 2), self.K1, self.D1, P=self.K1)
            ud_r = cv2.undistortPoints(px_r.reshape(-1, 1, 2), self.K2, self.D2, P=self.K2)
            
            # Triangulate
            pts_4d = cv2.triangulatePoints(self.P1, self.P2,
                                           ud_l.reshape(-1, 2).T,
                                           ud_r.reshape(-1, 2).T)
            
            # Convert to 3D
            pts3d = (pts_4d[:3] / pts_4d[3]).T

            # Apply calibration or tracking constraints
            if not self.calibration_done:
                output["phase"] = f"CALIBRATION ({self.calib_counts}/{self.calib_frames_total})"
                calibrate_lengths(self.bone_accum, pts3d)
                self.calib_counts += 1
                
                if self.calib_counts >= self.calib_frames_total:
                    self.bone_lengths_final = {b: np.mean(v) for b, v in self.bone_accum.items()}
                    self.calibration_done = True
                    print("Calibration finished. Switching to Tracking Mode.")
            else:
                output["phase"] = "GESTURE TRACKING"
                pts3d = apply_chain_correction(pts3d, self.bone_lengths_final)

            output["found"] = True
            output["joints"] = pts3d
            
        return output

    def close(self):
        """Cleanup resources"""
        self.cam.stop()

# Visualization class for real-time display
class HandVisualizerAllInOne:
    """Displays stereo camera feeds and 3D hand reconstruction"""
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(16, 12))
        self.conn = list(mp.solutions.hands.HAND_CONNECTIONS)
        
        # Fixed display dimensions
        self.display_width = 640
        self.display_height = 360

        # Create subplot layout
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 3], hspace=0.1, wspace=0.1)

        # Camera views
        self.ax_l = self.fig.add_subplot(gs[0, 0])
        self.ax_l.axis('off')
        self.im_l_disp = self.ax_l.imshow(np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8))
        
        self.ax_r = self.fig.add_subplot(gs[0, 1])
        self.ax_r.axis('off')
        self.im_r_disp = self.ax_r.imshow(np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8))

        # 3D views
        self.ax3d_front = self.fig.add_subplot(gs[1, 0], projection='3d')
        self.ax3d_front.view_init(-90, -90)
        self._init_3d_axis(self.ax3d_front)

        self.ax3d_side = self.fig.add_subplot(gs[1, 1], projection='3d')
        self.ax3d_side.view_init(0, 0)
        self._init_3d_axis(self.ax3d_side)

        # Initialize visualization elements
        self.lines_2d_l = [self.ax_l.plot([], [], 'g-', linewidth=2)[0] for _ in self.conn]
        self.points_2d_l = self.ax_l.plot([], [], 'ro', markersize=4)[0]
        self.lines_2d_r = [self.ax_r.plot([], [], 'g-', linewidth=2)[0] for _ in self.conn]
        self.points_2d_r = self.ax_r.plot([], [], 'ro', markersize=4)[0]

        self.scats_3d = []
        self.lines_3d_collections = []
        for ax in [self.ax3d_front, self.ax3d_side]:
            scat = ax.scatter([], [], [], c='r', s=40)
            lines = [ax.plot([], [], [], 'b-', linewidth=2)[0] for _ in self.conn]
            self.scats_3d.append(scat)
            self.lines_3d_collections.append(lines)

        self.fig.tight_layout()

    def _init_3d_axis(self, ax):
        """Initialize 3D axis limits"""
        ax.set_xlim(-150, 150)
        ax.set_ylim(-150, 150)
        ax.set_zlim(100, 400)

    def set_status(self, text):
        """Update window title with tracking status"""
        self.fig.suptitle(f"Stereo Hand Tracking - {text}", fontsize=14)

    def update(self, img_l, img_r, pts_l, pts_r, pts3d, rotated_width, rotated_height):
        """Update all visualization elements"""
        try:
            # Scale images to fit display while preserving aspect ratio
            h_l, w_l = img_l.shape[:2]
            h_r, w_r = img_r.shape[:2]
            
            # Calculate scaling factors
            scale_l = min(self.display_width / w_l, self.display_height / h_l)
            scale_r = min(self.display_width / w_r, self.display_height / h_r)
            
            new_w_l, new_h_l = int(w_l * scale_l), int(h_l * scale_l)
            new_w_r, new_h_r = int(w_r * scale_r), int(h_r * scale_r)
            
            # Resize images
            display_img_l = cv2.resize(img_l, (new_w_l, new_h_l))
            display_img_r = cv2.resize(img_r, (new_w_r, new_h_r))
            
            # Create padded display images
            padded_img_l = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
            padded_img_r = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
            
            # Center images in display area
            y_offset_l = (self.display_height - new_h_l) // 2
            x_offset_l = (self.display_width - new_w_l) // 2
            y_offset_r = (self.display_height - new_h_r) // 2
            x_offset_r = (self.display_width - new_w_r) // 2
            
            padded_img_l[y_offset_l:y_offset_l+new_h_l, x_offset_l:x_offset_l+new_w_l] = display_img_l
            padded_img_r[y_offset_r:y_offset_r+new_h_r, x_offset_r:x_offset_r+new_w_r] = display_img_r
            
            # Update camera displays
            self.im_l_disp.set_data(padded_img_l)
            self.im_r_disp.set_data(padded_img_r)
            
            # Update 2D landmarks with correct scaling
            scale_factors_l = (scale_l, scale_l, x_offset_l, y_offset_l)
            scale_factors_r = (scale_r, scale_r, x_offset_r, y_offset_r)
            
            self._update2d(self.lines_2d_l, self.points_2d_l, pts_l, scale_factors_l, (w_l, h_l))
            self._update2d(self.lines_2d_r, self.points_2d_r, pts_r, scale_factors_r, (w_r, h_r))
            
            # Update 3D visualization
            if pts3d is not None:
                for i in range(2):
                    self.scats_3d[i]._offsets3d = (pts3d[:, 0], pts3d[:, 1], pts3d[:, 2])
                    for line, (s, e) in zip(self.lines_3d_collections[i], self.conn):
                        line.set_data([pts3d[s, 0], pts3d[e, 0]],
                                      [pts3d[s, 1], pts3d[e, 1]])
                        line.set_3d_properties([pts3d[s, 2], pts3d[e, 2]])
            
            plt.pause(0.001)
        except Exception as e:
            print(f"Visualization error: {e}")

    def _update2d(self, lines, points, pts, scale_factors, original_size):
        """Update 2D landmark visualization with correct scaling"""
        if pts is not None:
            scale_x, scale_y, offset_x, offset_y = scale_factors
            orig_w, orig_h = original_size
            
            # Scale keypoints to match display image
            scaled_pts = pts.copy()
            scaled_pts[:, 0] = pts[:, 0] * scale_x + offset_x
            scaled_pts[:, 1] = pts[:, 1] * scale_y + offset_y
            
            points.set_data(scaled_pts[:, 0], scaled_pts[:, 1])
            for line, (s, e) in zip(lines, self.conn):
                line.set_data([scaled_pts[s, 0], scaled_pts[e, 0]],
                              [scaled_pts[s, 1], scaled_pts[e, 1]])
        else:
            points.set_data([], [])
            for line in lines:
                line.set_data([], [])


# Main application loop
def main():
    """Main application entry point"""
    # Initialize tracker and visualizer
    tracker = StereoHandTracker()
    visualizer = HandVisualizerAllInOne()

    print("Stereo Hand Tracking started. Press Ctrl+C to exit.")

    try:
        while True:
            # Process one frame
            data = tracker.step()
            
            if data["image_left"] is None:
                time.sleep(0.01)
                continue
            
            # Update status display
            visualizer.set_status(data["phase"])

            # Update visualization with rotated dimensions
            rotated_width = data.get("rotated_width", data["image_left"].shape[1])
            rotated_height = data.get("rotated_height", data["image_left"].shape[0])
            
            visualizer.update(
                data["image_left"], 
                data["image_right"], 
                data["px_left"], 
                data["px_right"], 
                data["joints"],
                rotated_width,
                rotated_height
            )

            # Check if window was closed
            if not plt.fignum_exists(visualizer.fig.number):
                print("Visualization window closed.")
                break

    except KeyboardInterrupt:
        print("\nStopping tracker...")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Cleanup
        tracker.close()
        plt.close('all')
        print("Tracker stopped.")

if __name__ == "__main__":
    main()