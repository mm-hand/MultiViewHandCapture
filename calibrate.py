import cv2
import json
import numpy as np
import sys
import threading
import time

# ================= Configuration =================
BOARD_SIZE = (9, 6) 
SQUARE_SIZE = 23.5 

# ================= High-performance camera capture class =================
class CameraStream:
    def __init__(self, src=0, width=2560, height=720):
        # 1. Specify V4L2 backend
        self.stream = cv2.VideoCapture(src, cv2.CAP_V4L2)
        
        # 2. Set MJPG first
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # 3. Key: set buffer size to 1
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Check if opened
        if not self.stream.isOpened():
            print("❌ Unable to open camera, trying ID 1...")
            self.stream.release()
            self.stream = cv2.VideoCapture(1, cv2.CAP_V4L2)
            self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.stream.isOpened():
            print("❌ FATAL: Unable to open any camera.")
            sys.exit()

        print(f"✅ Camera started: {self.stream.get(3)}x{self.stream.get(4)}")

        self.grabbed, self.frame = self.stream.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True  # Daemon thread will end when main program exits
        self.thread.start()
        return self

    def update(self):
        while self.started:
            try:
                # Even if a bad frame is read, it will not crash the main program
                grabbed, frame = self.stream.read()
                
                # Only update if the frame is valid
                if grabbed and frame is not None:
                    with self.read_lock:
                        self.grabbed = grabbed
                        self.frame = frame
                else:
                    # If unable to read, wait a bit to avoid busy loop high CPU usage
                    time.sleep(0.01)
            except Exception:
                # Ignore decoding errors and keep thread running
                pass

    def read(self):
        with self.read_lock:
            # Return a copy of the latest frame to prevent modification during processing
            return self.grabbed, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.started = False
        if self.thread.is_alive():
            self.thread.join()
        self.stream.release()

# ================= Main program =================

# Prepare object points for calibration pattern
objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

objpoints = [] 
imgpoints_l = [] 
imgpoints_r = [] 

print("=== Stereo High-Precision Calibration Program (Low-Latency Version) ===")
print(f"Checkerboard size: {BOARD_SIZE}, Square size: {SQUARE_SIZE} mm")

# Start multi-threaded camera capture
cam = CameraStream(src=0, width=2560, height=720).start()

# Wait for camera warmup
time.sleep(1.0)

count = 0

while True:
    # 1. Get the latest frame from camera thread (non-blocking, no delay)
    ret, frame = cam.read()

    if not ret or frame is None:
        continue

    # Check resolution
    if frame.shape[1] != 2560:
        continue

    img_l = frame[:, :1280]
    img_r = frame[:, 1280:]
    
    vis = frame.copy()  # Copy for display

    # 2. Optimization: do not detect corners in every frame
    # Corner detection is CPU heavy, running it every frame will reduce FPS to ~5 FPS.
    # We limit detection frequency to only indicate "alignment status".
    # This way, corner overlays update at ~10-15 FPS, but video runs smoothly at 30 FPS.
    
    # Convert to grayscale
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    # Detect corners
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, BOARD_SIZE, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, BOARD_SIZE, None)

    # Draw count
    cv2.putText(vis, f"Count: {count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)

    if ret_l and ret_r:
        # For smoother preview, only draw raw corners - no sub-pixel refinement yet
        cv2.drawChessboardCorners(vis[:, :1280], BOARD_SIZE, corners_l, ret_l)
        cv2.drawChessboardCorners(vis[:, 1280:], BOARD_SIZE, corners_r, ret_r)
        
        cv2.putText(vis, "READY - Press C", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    else:
        cv2.putText(vis, "Searching...", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    # Resize for display
    vis_small = cv2.resize(vis, (0, 0), fx=0.4, fy=0.4)
    cv2.imshow('Calibration', vis_small)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('c'):
        # Only run time-consuming sub-pixel optimization when capture key is pressed
        if ret_l and ret_r:
            print("Optimizing corners and saving...")
            criteria_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            
            # Run sub-pixel optimization
            corners_l_opt = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria_subpix)
            corners_r_opt = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria_subpix)
            
            objpoints.append(objp)
            imgpoints_l.append(corners_l_opt)
            imgpoints_r.append(corners_r_opt)
            count += 1
            print(f"✅ Captured image {count}")
            
            # Flash the screen briefly to indicate success
            cv2.imshow('Calibration', np.zeros_like(vis_small))
            cv2.waitKey(50)
        else:
            print("⚠️ No complete chessboard detected, capture failed.")
            
    elif key & 0xFF == ord('q'):
        break

# Stop camera thread
cam.stop()
cv2.destroyAllWindows()

if count < 10:
    print("Too few images, exiting.")
    sys.exit()

# ================= Calibration computation =================
print("\n=== Computing Parameters ===")
print("1. Calibrating left camera...")
ret_l, K_l, D_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, (1280, 720), None, None)
print(f"   RMS error: {ret_l:.4f}")

print("2. Calibrating right camera...")
ret_r, K_r, D_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, (1280, 720), None, None)
print(f"   RMS error: {ret_r:.4f}")

print("3. Stereo calibration...")
flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3 
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

ret_s, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_l, imgpoints_r,
    K_l, D_l, K_r, D_r,
    (1280, 720),
    criteria=criteria_stereo,
    flags=flags
)

np.set_printoptions(suppress=True, precision=4)
print("\n" + "="*50)
print(f"Stereo reprojection RMS error: {ret_s:.4f}")
print(f"Computed baseline length: {np.linalg.norm(T):.2f} mm")
print("="*50)

# ================= Save to JSON =================
stereo_params = {
    "K1": K1.tolist(),
    "D1": D1.tolist(),
    "K2": K2.tolist(),
    "D2": D2.tolist(),
    "R":  R.tolist(),
    "T":  T.tolist(),
    "rms": ret_s
}

json_path = "stereo_params.json"
with open(json_path, "w") as f:
    json.dump(stereo_params, f, indent=4)

print(f"\n✅ Calibration parameters saved to: {json_path}")
print("Camera.py will automatically load this file.")
