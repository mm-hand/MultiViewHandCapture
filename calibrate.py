import cv2
import json
import numpy as np
import sys
import threading
import time

# ================= 配置区域 =================
BOARD_SIZE = (9, 6) 
SQUARE_SIZE = 23.5 

# ================= 高性能摄像头读取类 =================
class CameraStream:
    def __init__(self, src=0, width=2560, height=720):
        # 1. 指定 V4L2 后端
        self.stream = cv2.VideoCapture(src, cv2.CAP_V4L2)
        
        # 2. 必须先设置 MJPG
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # 3. 关键：设置缓冲区为 1
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # 检查是否打开
        if not self.stream.isOpened():
            print("❌ 无法打开摄像头，尝试切换 ID...")
            self.stream.release()
            self.stream = cv2.VideoCapture(1, cv2.CAP_V4L2)
            self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.stream.isOpened():
            print("❌ 严重错误：无法打开任何摄像头。")
            sys.exit()

        print(f"✅ 相机已启动: {self.stream.get(3)}x{self.stream.get(4)}")

        self.grabbed, self.frame = self.stream.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True # 设为守护线程，主程序退出时自动结束
        self.thread.start()
        return self

    def update(self):
        while self.started:
            try:
                # 这里的 read 即使遇到坏帧抛出异常，也不会崩溃主程序
                grabbed, frame = self.stream.read()
                
                # 只有读到有效帧才更新
                if grabbed and frame is not None:
                    with self.read_lock:
                        self.grabbed = grabbed
                        self.frame = frame
                else:
                    # 如果读不到，稍微等待一下，避免死循环占满CPU
                    time.sleep(0.01)
            except Exception:
                # 忽略解码错误，保持线程运行
                pass

    def read(self):
        with self.read_lock:
            # 返回最新的帧的副本，防止在主线程处理时被子线程修改
            return self.grabbed, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.started = False
        if self.thread.is_alive():
            self.thread.join()
        self.stream.release()

# ================= 主程序 =================

# 初始化数据容器
objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

objpoints = [] 
imgpoints_l = [] 
imgpoints_r = [] 

print("=== 双目高精度标定程序 (极低延迟版) ===")
print(f"棋盘规格: {BOARD_SIZE}, 格子大小: {SQUARE_SIZE}mm")

# 启动多线程相机读取
cam = CameraStream(src=0, width=2560, height=720).start()

# 等待摄像头预热
time.sleep(1.0)

count = 0
last_detection_time = 0

while True:
    # 1. 从子线程获取最新帧 (非阻塞，无延迟)
    ret, frame = cam.read()

    if not ret or frame is None:
        continue

    # 检查分辨率
    if frame.shape[1] != 2560:
        continue

    img_l = frame[:, :1280]
    img_r = frame[:, 1280:]
    
    vis = frame.copy() # 用于显示的画布

    # 2. 优化：不要每一帧都去检测角点
    # 角点检测非常耗时，如果跑满每一帧会把FPS拖慢到 5帧/秒
    # 我们限制检测频率，只用来提示用户“当前是否对准了”
    # 这样虽然显示的角点可能只有 10-15 FPS，但视频画面是 30 FPS 流畅的
    
    # 将图像转为灰度
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    # 检测角点
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, BOARD_SIZE, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, BOARD_SIZE, None)

    # 绘制
    cv2.putText(vis, f"Count: {count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)

    if ret_l and ret_r:
        # 为了预览流畅，这里只画粗略角点，不做亚像素优化（亚像素优化在按 'c' 后做）
        cv2.drawChessboardCorners(vis[:, :1280], BOARD_SIZE, corners_l, ret_l)
        cv2.drawChessboardCorners(vis[:, 1280:], BOARD_SIZE, corners_r, ret_r)
        
        cv2.putText(vis, "READY - Press C", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    else:
        cv2.putText(vis, "Searching...", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    # 缩放显示
    vis_small = cv2.resize(vis, (0, 0), fx=0.4, fy=0.4)
    cv2.imshow('Calibration', vis_small)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('c'):
        # 按下采集键时，才进行高耗时的亚像素优化
        if ret_l and ret_r:
            print("正在优化角点并保存...")
            criteria_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            
            # 使用刚才的灰度图进行亚像素优化
            corners_l_opt = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria_subpix)
            corners_r_opt = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria_subpix)
            
            objpoints.append(objp)
            imgpoints_l.append(corners_l_opt)
            imgpoints_r.append(corners_r_opt)
            count += 1
            print(f"✅ 已采集第 {count} 张")
            
            # 闪烁一下屏幕提示成功
            cv2.imshow('Calibration', np.zeros_like(vis_small))
            cv2.waitKey(50)
        else:
            print("⚠️ 未检测到完整的棋盘格，无法采集")
            
    elif key & 0xFF == ord('q'):
        break

# 停止摄像头线程
cam.stop()
cv2.destroyAllWindows()

if count < 10:
    print("图片过少，退出。")
    sys.exit()

# ================= 标定计算部分 (保持不变) =================
print("\n=== 开始计算参数 ===")
print("1. 标定左相机...")
ret_l, K_l, D_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, (1280, 720), None, None)
print(f"   RMS: {ret_l:.4f}")

print("2. 标定右相机...")
ret_r, K_r, D_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, (1280, 720), None, None)
print(f"   RMS: {ret_r:.4f}")

print("3. 双目联合标定...")
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
print(f"双目重投影误差 RMS: {ret_s:.4f}")
print(f"计算出的基线长度: {np.linalg.norm(T):.2f} mm")
print("="*50)

# ================= 保存为 JSON =================
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

print(f"\n✅ 标定参数已保存至: {json_path}")
print("Camera.py 将自动加载此文件。")