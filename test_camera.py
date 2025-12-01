import cv2
import time
from config import CAMERA_INDEX, FULL_WIDTH, HEIGHT 

def show_stereo_camera_robust():
    # 1. Specify backend as V4L2 (required for Linux)
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print(f"❌ Unable to open camera {CAMERA_INDEX}")
        return

    # 2. Set MJPG (must be set before resolution)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    # 3. Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FULL_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    
    # 4. Set buffer size to 1 to reduce frame backlog and screen tearing/delay
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Print parameter confirmation
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"✅ Camera started: {int(actual_w)}x{int(actual_h)} (MJPG)")

    prev_time = 0
    error_count = 0

    while True:
        try:
            # 5. Wrap read() in try-except to prevent imdecode crashes
            ret, frame = cap.read()
            
            # If no frame is read or frame is None
            if not ret or frame is None:
                print("⚠️ Dropped Frame (Empty Frame)")
                error_count += 1
                if error_count > 50:  # Exit if more than 50 consecutive failures
                    print("❌ Too many consecutive failures, exiting...")
                    break
                continue
            
            # Reset error counter upon successful read
            error_count = 0

            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time

            # Display FPS on frame
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Scale down display to 0.5 size
            display_frame = cv2.resize(frame, (int(actual_w * 0.5), int(actual_h * 0.5)))
            cv2.imshow('Stereo Camera', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except cv2.error as e:
            # 6. Catch OpenCV internal C++ errors
            print(f"⚠️ OpenCV decoding error (ignored): {e}")
            time.sleep(0.01)  # Pause briefly to allow buffer to recover
            continue

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_stereo_camera_robust()
