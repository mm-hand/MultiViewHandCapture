import cv2
import threading
import time
import os
import numpy as np

def rotate_image(image, angle_degrees):
    """Rotate image and return rotated image with new dimensions"""
    if angle_degrees == 0:
        return image, image.shape[1], image.shape[0]
    
    height, width = image.shape[:2]
    
    if angle_degrees == 90 or angle_degrees == -270:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return rotated, height, width
    elif angle_degrees == -90 or angle_degrees == 270:
        rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return rotated, height, width
    elif abs(angle_degrees) == 180:
        rotated = cv2.rotate(image, cv2.ROTATE_180)
        return rotated, width, height
    else:
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, -angle_degrees, 1.0)
        cos_val = abs(rotation_matrix[0, 0])
        sin_val = abs(rotation_matrix[0, 1])
        new_width = int((height * sin_val) + (width * cos_val))
        new_height = int((height * cos_val) + (width * sin_val))
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
        return rotated, new_width, new_height

class CameraStream:
    def __init__(self, src, width, height):
        backend = cv2.CAP_V4L2 if os.name == 'posix' else cv2.CAP_ANY
        self.stream = cv2.VideoCapture(src, backend)
        
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.stream.isOpened():
            print(f"Unable to open camera {src}, trying ID 1...")
            self.stream.release()
            self.stream = cv2.VideoCapture(1, backend)
            self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.stream.isOpened():
            raise RuntimeError("No camera detected.")

        actual_w = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Camera started: {int(actual_w)}x{int(actual_h)}")
        
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

def test_camera_robust(camera_index, full_width, height):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print(f"Unable to open camera {camera_index}")
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, full_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera started: {int(actual_w)}x{int(actual_h)}")

    # Create a resizable window
    cv2.namedWindow('Stereo Camera Test', cv2.WINDOW_NORMAL)
    
    # Pre-calculate target display size once to avoid recalculating every frame
    from config import ROTATE_LEFT, ROTATE_RIGHT
    
    # Pre-calculate display size using default image dimensions
    single_width = full_width // 2
    test_img = np.zeros((height, single_width, 3), dtype=np.uint8)
    
    # Calculate dimensions after rotation for left and right images
    _, lw, lh = rotate_image(test_img, ROTATE_LEFT)
    _, rw, rh = rotate_image(test_img, ROTATE_RIGHT)
    
    # Calculate total dimensions after merging
    total_width = lw + rw
    total_height = max(lh, rh)
    
    # Calculate scale factor to fit display
    max_display_width = 1200
    scale = min(1.0, max_display_width / total_width)
    display_width = int(total_width * scale)
    display_height = int(total_height * scale)
    
    # Set initial window size (only once)
    cv2.resizeWindow('Stereo Camera Test', display_width, display_height)

    prev_time = 0
    error_count = 0

    while True:
        try:
            ret, frame = cap.read()
            
            if not ret or frame is None:
                error_count += 1
                if error_count > 50:
                    break
                continue
            
            error_count = 0

            # Split left and right images
            single_width = full_width // 2
            img_l = frame[:, :single_width]
            img_r = frame[:, single_width:]
            
            # Apply rotation
            img_l_rotated, _, _ = rotate_image(img_l, ROTATE_LEFT)
            img_r_rotated, _, _ = rotate_image(img_r, ROTATE_RIGHT)
            
            # Ensure consistent height
            h1, w1 = img_l_rotated.shape[:2]
            h2, w2 = img_r_rotated.shape[:2]
            max_height = max(h1, h2)
            
            if h1 != max_height:
                img_l_rotated = cv2.resize(img_l_rotated, (int(w1 * max_height / h1), max_height))
            if h2 != max_height:
                img_r_rotated = cv2.resize(img_r_rotated, (int(w2 * max_height / h2), max_height))
            
            # Merge images
            frame_rotated = np.hstack((img_l_rotated, img_r_rotated))
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time

            # Add info text
            rotation_info = f"L:{ROTATE_LEFT} degree R:{ROTATE_RIGHT} degree FPS: {int(fps)}"
            cv2.putText(frame_rotated, rotation_info, (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Scale using pre-calculated dimensions (no window resizing)
            display_frame = cv2.resize(frame_rotated, (display_width, display_height))
            cv2.imshow('Stereo Camera Test', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    from config import CAMERA_INDEX, FULL_WIDTH, HEIGHT
    test_camera_robust(CAMERA_INDEX, FULL_WIDTH, HEIGHT)