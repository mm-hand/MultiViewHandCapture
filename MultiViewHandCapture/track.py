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

def compute_relative_coordinates(pts3d_absolute):
    """Compute relative hand coordinates normalized to palm coordinate system"""
    if pts3d_absolute is None or len(pts3d_absolute) != 21:
        return None
    
    # 1. Translate to wrist-centered coordinates
    wrist_pos = pts3d_absolute[0].copy()
    pts_centered = pts3d_absolute - wrist_pos
    
    # 2. Calculate palm size for normalization
    # Use average distance from wrist to all four finger bases
    finger_base_indices = [5, 9, 13, 17]  # Index, middle, ring, pinky finger bases
    palm_size = 0.0
    for idx in finger_base_indices:
        palm_size += np.linalg.norm(pts_centered[idx])
    palm_size /= len(finger_base_indices)
    
    if palm_size < 1e-6:
        return None
    
    # 3. Length normalization
    pts_normalized = pts_centered / palm_size
    
    # 4. Improved rotation normalization - align with palm coordinate system
    
    # 4.1 Z-axis: from wrist to average of four finger bases
    finger_bases = pts_normalized[finger_base_indices]
    finger_center = np.mean(finger_bases, axis=0)
    z_axis = finger_center - pts_normalized[0]  # Wrist to finger center
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # 4.2 Y-axis: average of two orthogonal vectors in the hand plane
    # Vector 1: from index finger base to ring finger base
    vec_y1 = pts_normalized[13] - pts_normalized[5]  # Ring to index
    vec_y1 = vec_y1 / np.linalg.norm(vec_y1)
    
    # Vector 2: from middle finger base to pinky finger base
    vec_y2 = pts_normalized[17] - pts_normalized[9]  # Pinky to middle
    vec_y2 = vec_y2 / np.linalg.norm(vec_y2)
    
    # Average of the two vectors
    y_axis_avg = (vec_y1 + vec_y2) / 2.0
    y_axis_avg = y_axis_avg / np.linalg.norm(y_axis_avg)
    
    # 4.3 X-axis: cross product of average Y-axis and Z-axis
    x_axis = np.cross(y_axis_avg, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # 4.4 Recompute Y-axis to ensure orthogonality
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # 5. Construct rotation matrix
    # Rotation matrix from palm coordinate system to world coordinate system
    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
    
    # 6. Apply inverse rotation to align with palm coordinate system
    # This transforms points from world coordinates to palm coordinates
    pts_rotated = pts_normalized @ rotation_matrix
    
    return pts_rotated

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
            handedness = res.multi_handedness[0].classification[0].label
            if handedness == 'Right':
                handedness = 'Left'
            elif handedness == 'Left':
                handedness = 'Right'
            return self.filter(raw), handedness
        else:
            self.filter.reset()
            return None, None

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
            "keypoint_absolute": None,
            "keypoint_relative": None, 
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
        pts_norm_l, handedness_l = self.proc_l.process(img_l_rgb)
        pts_norm_r, handedness_r = self.proc_r.process(img_r_rgb) 
        
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
            pts3d_absolute = (pts_4d[:3] / pts_4d[3]).T
            pts3d_relative = compute_relative_coordinates(pts3d_absolute)

            final_handedness = handedness_l if handedness_l else handedness_r

            # Apply calibration or tracking constraints
            if not self.calibration_done:
                output["phase"] = f"CALIBRATION ({self.calib_counts}/{self.calib_frames_total}) - {final_handedness} Hand"
                calibrate_lengths(self.bone_accum, pts3d_absolute)                
                self.calib_counts += 1
                
                if self.calib_counts >= self.calib_frames_total:
                    self.bone_lengths_final = {b: np.mean(v) for b, v in self.bone_accum.items()}
                    self.calibration_done = True
                    print("Calibration finished. Switching to Tracking Mode.")
            else:
                output["phase"] = f"GESTURE TRACKING - {final_handedness} Hand"
                pts3d_absolute = apply_chain_correction(pts3d_absolute, self.bone_lengths_final)
                pts3d_relative = compute_relative_coordinates(pts3d_absolute)

            output["found"] = True
            output["handedness"] = final_handedness
            output["keypoint_absolute"] = pts3d_absolute
            output["keypoint_relative"] = pts3d_relative
            
        return output

    def close(self):
        """Cleanup resources"""
        self.cam.stop()

# Visualization class for real-time display
class HandVisualizerAllInOne:
    """Displays stereo camera feeds and 3D hand reconstruction for both absolute and relative coordinates"""
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(20, 12))  # Increased width for additional plots
        self.conn = list(mp.solutions.hands.HAND_CONNECTIONS)
        
        # Fixed display dimensions
        self.display_width = 640
        self.display_height = 360

        # Create subplot layout: 2 rows, 3 columns
        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 3], hspace=0.1, wspace=0.15)

        # Camera views - first row, first two columns
        self.ax_l = self.fig.add_subplot(gs[0, 0])
        self.ax_l.axis('off')
        self.ax_l.set_title('Left Camera')
        self.im_l_disp = self.ax_l.imshow(np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8))
        
        self.ax_r = self.fig.add_subplot(gs[0, 1])
        self.ax_r.axis('off')
        self.ax_r.set_title('Right Camera')
        self.im_r_disp = self.ax_r.imshow(np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8))

        # Placeholder for the third column in first row (empty)
        self.ax_placeholder = self.fig.add_subplot(gs[0, 2])
        self.ax_placeholder.axis('off')
        self.ax_placeholder.set_title('Coordinate Systems')
        # Add text explanation
        self.ax_placeholder.text(0.5, 0.7, 'Absolute: World coordinates\nRelative: Palm-centered', 
                                 ha='center', va='center', transform=self.ax_placeholder.transAxes, fontsize=12)
        self.ax_placeholder.text(0.5, 0.3, 'Red: Absolute\nGreen: Relative', 
                                 ha='center', va='center', transform=self.ax_placeholder.transAxes, fontsize=12)

        # 3D views - second row, three columns
        # Absolute coordinates - front view
        self.ax3d_abs_front = self.fig.add_subplot(gs[1, 0], projection='3d')
        self.ax3d_abs_front.view_init(-90, -90)
        self.ax3d_abs_front.set_title('Absolute Coords - Front View')
        self._init_3d_axis(self.ax3d_abs_front)

        # Absolute coordinates - side view  
        self.ax3d_abs_side = self.fig.add_subplot(gs[1, 1], projection='3d')
        self.ax3d_abs_side.view_init(0, 0)
        self.ax3d_abs_side.set_title('Absolute Coords - Side View')
        self._init_3d_axis(self.ax3d_abs_side)

        # Relative coordinates - front view
        self.ax3d_rel_front = self.fig.add_subplot(gs[1, 2], projection='3d')
        self.ax3d_rel_front.view_init(0, 45)
        self.ax3d_rel_front.set_title('Relative Coords - Front View')
        self._init_relative_3d_axis(self.ax3d_rel_front)

        # Initialize visualization elements for both absolute and relative coordinates
        self._init_visualization_elements()

    def _init_3d_axis(self, ax):
        """Initialize 3D axis limits for absolute coordinates (world scale)"""
        ax.set_xlim(-150, 150)
        ax.set_ylim(-150, 150)
        ax.set_zlim(100, 400)
        # Set labels
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')

    def _init_relative_3d_axis(self, ax):
        """Initialize 3D axis limits for relative coordinates (normalized scale)"""
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        # Set labels
        ax.set_xlabel('X (normalized)')
        ax.set_ylabel('Y (normalized)')
        ax.set_zlabel('Z (normalized)')

    def _init_visualization_elements(self):
        """Initialize visualization elements for both coordinate systems"""
        # 2D landmarks for camera views
        self.lines_2d_l = [self.ax_l.plot([], [], 'g-', linewidth=2)[0] for _ in self.conn]
        self.points_2d_l = self.ax_l.plot([], [], 'ro', markersize=4)[0]
        self.lines_2d_r = [self.ax_r.plot([], [], 'g-', linewidth=2)[0] for _ in self.conn]
        self.points_2d_r = self.ax_r.plot([], [], 'ro', markersize=4)[0]

        # 3D visualization for absolute coordinates (two views)
        self.scats_abs = []
        self.lines_abs = []
        for ax in [self.ax3d_abs_front, self.ax3d_abs_side]:
            scat = ax.scatter([], [], [], c='r', s=40, label='keypoint')
            lines = [ax.plot([], [], [], 'b-', linewidth=2)[0] for _ in self.conn]
            self.scats_abs.append(scat)
            self.lines_abs.append(lines)
            # Add legend
            ax.legend()

        # 3D visualization for relative coordinates (one view)
        self.scat_rel = self.ax3d_rel_front.scatter([], [], [], c='g', s=40, label='keypoint')
        self.lines_rel = [self.ax3d_rel_front.plot([], [], [], 'y-', linewidth=2)[0] for _ in self.conn]
        # Add legend
        self.ax3d_rel_front.legend()

    def set_status(self, text):
        """Update window title with tracking status"""
        self.fig.suptitle(f"Stereo Hand Tracking - {text}", fontsize=14)

    def update(self, img_l, img_r, pts_l, pts_r, pts3d_absolute, pts3d_relative, rotated_width, rotated_height):
        """Update all visualization elements including both coordinate systems"""
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
            
            # Update 3D visualization for absolute coordinates
            if pts3d_absolute is not None:
                for i in range(2):  # For both front and side views
                    self.scats_abs[i]._offsets3d = (pts3d_absolute[:, 0], pts3d_absolute[:, 1], pts3d_absolute[:, 2])
                    for line, (s, e) in zip(self.lines_abs[i], self.conn):
                        line.set_data([pts3d_absolute[s, 0], pts3d_absolute[e, 0]],
                                      [pts3d_absolute[s, 1], pts3d_absolute[e, 1]])
                        line.set_3d_properties([pts3d_absolute[s, 2], pts3d_absolute[e, 2]])

            # Update 3D visualization for relative coordinates
            if pts3d_relative is not None:
                self.scat_rel._offsets3d = (pts3d_relative[:, 0], pts3d_relative[:, 1], pts3d_relative[:, 2])
                for line, (s, e) in zip(self.lines_rel, self.conn):
                    line.set_data([pts3d_relative[s, 0], pts3d_relative[e, 0]],
                                  [pts3d_relative[s, 1], pts3d_relative[e, 1]])
                    line.set_3d_properties([pts3d_relative[s, 2], pts3d_relative[e, 2]])
            
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
                data["keypoint_absolute"],
                data["keypoint_relative"],
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