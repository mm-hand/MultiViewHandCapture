import cv2
import json
import numpy as np
import sys
import time

from MultiViewHandCapture.camera import CameraStream, rotate_image
from MultiViewHandCapture.config import CAMERA_INDEX, FULL_WIDTH, HEIGHT, BOARD_SIZE, SQUARE_SIZE, ROTATE_LEFT, ROTATE_RIGHT

# Prepare object points for calibration pattern
objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

objpoints = [] 
imgpoints_l = [] 
imgpoints_r = [] 

print("=== Stereo High-Precision Calibration Program ===")
print(f"Checkerboard size: {BOARD_SIZE}, Square size: {SQUARE_SIZE} mm")

# Start multi-threaded camera capture
cam = CameraStream(src=CAMERA_INDEX, width=FULL_WIDTH, height=HEIGHT).start()

# Wait for camera warmup
time.sleep(1.0)

count = 0

# Pre-calculate display dimensions based on rotation
single_width = FULL_WIDTH // 2
test_img = np.zeros((HEIGHT, single_width, 3), dtype=np.uint8)

# Calculate dimensions after rotation
_, lw, lh = rotate_image(test_img, ROTATE_LEFT)
_, rw, rh = rotate_image(test_img, ROTATE_RIGHT)

# Calculate total dimensions after merging
total_width = lw + rw
total_height = max(lh, rh)

# Calculate scale factor for display
max_display_width = 800
scale = min(1.0, max_display_width / total_width)
display_width = int(total_width * scale)
display_height = int(total_height * scale)

# Create resizable window
cv2.namedWindow('Stereo Calibration', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Stereo Calibration', display_width, display_height)

while True:
    # Get the latest frame from camera thread
    ret, frame = cam.read()

    if not ret or frame is None:
        continue

    # Check resolution
    if frame.shape[1] != FULL_WIDTH:
        continue

    # Split and rotate images
    img_l = frame[:, :single_width]
    img_r = frame[:, single_width:]
    
    img_l_rotated, _, _ = rotate_image(img_l, ROTATE_LEFT)
    img_r_rotated, _, _ = rotate_image(img_r, ROTATE_RIGHT)
    
    # Ensure consistent height for display
    h1, w1 = img_l_rotated.shape[:2]
    h2, w2 = img_r_rotated.shape[:2]
    max_height = max(h1, h2)
    
    if h1 != max_height:
        img_l_rotated = cv2.resize(img_l_rotated, (int(w1 * max_height / h1), max_height))
    if h2 != max_height:
        img_r_rotated = cv2.resize(img_r_rotated, (int(w2 * max_height / h2), max_height))
    
    # Create display image
    vis = np.hstack((img_l_rotated, img_r_rotated))

    # Convert to grayscale for corner detection
    gray_l = cv2.cvtColor(img_l_rotated, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r_rotated, cv2.COLOR_BGR2GRAY)

    # Detect corners on rotated images
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, BOARD_SIZE, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, BOARD_SIZE, None)

    # Draw count and status
    cv2.putText(vis, f"Count: {count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    if ret_l and ret_r:
        # Draw corners on rotated images
        vis_left = vis[:, :img_l_rotated.shape[1]]
        vis_right = vis[:, img_l_rotated.shape[1]:]
        
        cv2.drawChessboardCorners(vis_left, BOARD_SIZE, corners_l, ret_l)
        cv2.drawChessboardCorners(vis_right, BOARD_SIZE, corners_r, ret_r)
        
        cv2.putText(vis, "READY - Press C", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(vis, "Searching...", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Resize for display
    display_frame = cv2.resize(vis, (display_width, display_height))
    cv2.imshow('Stereo Calibration', display_frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('c'):
        # Capture calibration image when both checkerboards are detected
        if ret_l and ret_r:
            print("Optimizing corners and saving...")
            criteria_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            
            # Run sub-pixel optimization on rotated images
            corners_l_opt = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria_subpix)
            corners_r_opt = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria_subpix)
            
            objpoints.append(objp)
            imgpoints_l.append(corners_l_opt)
            imgpoints_r.append(corners_r_opt)
            count += 1
            print(f"Captured image {count}")
            
            # Flash screen briefly to indicate success
            cv2.imshow('Stereo Calibration', np.zeros_like(display_frame))
            cv2.waitKey(50)
        else:
            print("No complete chessboard detected, capture failed.")
            
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

# Use rotated image dimensions for calibration
rotated_width = lw
rotated_height = lh

print("1. Calibrating left camera...")
ret_l, K_l, D_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, (rotated_width, rotated_height), None, None)
print(f"   RMS error: {ret_l:.4f}")

print("2. Calibrating right camera...")
ret_r, K_r, D_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, (rotated_width, rotated_height), None, None)
print(f"   RMS error: {ret_r:.4f}")

print("3. Stereo calibration...")
flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3 
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

ret_s, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_l, imgpoints_r,
    K_l, D_l, K_r, D_r,
    (rotated_width, rotated_height),
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
    "rms": ret_s,
    "rotated_width": rotated_width,
    "rotated_height": rotated_height
}

json_path = "stereo_params.json"
with open(json_path, "w") as f:
    json.dump(stereo_params, f, indent=4)

print(f"\nCalibration parameters saved to: {json_path}")
print("Camera will automatically load this file.")