import cv2
import time

def show_stereo_camera_robust():
    CAMERA_INDEX = 0
    TARGET_WIDTH = 2560
    TARGET_HEIGHT = 720
    
    # 1. 指定后端为 V4L2 (Linux 必备)
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头 {CAMERA_INDEX}")
        return

    # 2. 设置 MJPG (必须在前)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    # 3. 设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
    
    # 4. 【新加】设置缓冲区大小为1，减少积压和花屏/延迟
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 打印参数确认
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"✅ 相机已启动: {int(actual_w)}x{int(actual_h)} (MJPG)")

    prev_time = 0
    error_count = 0

    while True:
        try:
            # 5. 【关键】使用 try-except 包裹 read()，防止 imdecode 崩溃
            ret, frame = cap.read()
            
            # 如果没有读到帧，或者帧是空的
            if not ret or frame is None:
                print("⚠️ 丢帧 (Empty Frame)")
                error_count += 1
                if error_count > 50: # 连续50次失败则退出
                    print("❌ 连续失败次数过多，退出...")
                    break
                continue
            
            # 成功读取，重置错误计数
            error_count = 0

            # 计算FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time

            # 显示
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 缩放显示 (0.5倍)
            display_frame = cv2.resize(frame, (int(actual_w*0.5), int(actual_h*0.5)))
            cv2.imshow('Stereo Camera', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except cv2.error as e:
            # 6. 捕获 OpenCV 的内部 C++ 错误
            print(f"⚠️ OpenCV 解码错误 (忽略): {e}")
            time.sleep(0.01) # 稍微暂停一下，让缓冲区恢复
            continue

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_stereo_camera_robust()
