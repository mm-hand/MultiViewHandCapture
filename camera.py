import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
import sys
import time

# ================= 0. 标定参数 =================
# 左相机内参
K1 = np.array([[714.1656,   0.    , 595.7304],
               [  0.    , 715.3248, 427.8916],
               [  0.    ,   0.    ,   1.    ]], dtype=np.float64)
D1 = np.array([[ 0.0442,  0.157 ,  0.0113, -0.0031, -0.4461]], dtype=np.float64)
# 右相机内参
K2 = np.array([[718.2567,   0.    , 629.9524],
               [  0.    , 719.0196, 342.6635],
               [  0.    ,   0.    ,   1.    ]], dtype=np.float64)
D2 = np.array([[ 0.0586, -0.0063,  0.0077, -0.0042, -0.1363]], dtype=np.float64)
# 旋转矩阵
R  = np.array([[ 0.9985,  0.0008, -0.0547],
               [ 0.0013,  0.9993,  0.0385],
               [ 0.0547, -0.0385,  0.9978]], dtype=np.float64)
# 平移向量
T  = np.array([[51.3741],
               [ 1.2832],
               [-6.6857]], dtype=np.float64)

P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = K2 @ np.hstack((R, T))

# ================= 1. 多线程相机读取类 =================
class CameraStream:
    def __init__(self, src=0, width=2560, height=720):
        self.stream = cv2.VideoCapture(src, cv2.CAP_V4L2)
        # 强制 MJPG 和 缓存=1
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.stream.isOpened():
            print("❌ 无法打开摄像头 ID 0，尝试 ID 1...")
            self.stream.release()
            self.stream = cv2.VideoCapture(1, cv2.CAP_V4L2)
            self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.stream.isOpened():
            print("❌ 严重错误：无法打开摄像头。")
            sys.exit()
            
        print(f"✅ 相机已启动: {self.stream.get(3)}x{self.stream.get(4)}")
        self.grabbed, self.frame = self.stream.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started: return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
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
        if self.thread.is_alive(): self.thread.join()
        self.stream.release()

# ================= 2. 3D 可视化类 =================
class HandVisualizer3D:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.view_init(elev=-90, azim=-90) 
        
        self.conn = list(mp.solutions.hands.HAND_CONNECTIONS)
        self.scat = self.ax.scatter([], [], [], c='r', s=25, depthshade=True)
        self.lines = [self.ax.plot([], [], [], 'b-', linewidth=2)[0] for _ in range(len(self.conn))]
        
        self.ax.set_xlim(-150, 150)
        self.ax.set_ylim(-150, 150)
        self.ax.set_zlim(300, 800)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

    def update(self, p3d):
        try:
            self.scat._offsets3d = (p3d[:, 0], p3d[:, 1], p3d[:, 2])
            for line, (i, j) in zip(self.lines, self.conn):
                line.set_data([p3d[i, 0], p3d[j, 0]], [p3d[i, 1], p3d[j, 1]])
                line.set_3d_properties([p3d[i, 2], p3d[j, 2]])
            plt.pause(0.001) 
        except Exception:
            pass

# ================= 3. 手势处理器 =================
class HandProcessor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands.Hands(
            max_num_hands=1, 
            min_detection_confidence=0.4, 
            min_tracking_confidence=0.4
        )
        self.x_prev = None

    def process(self, img):
        res = self.mp_hands.process(img)
        if res.multi_hand_landmarks:
            raw = np.array([[lm.x, lm.y] for lm in res.multi_hand_landmarks[0].landmark])
            if self.x_prev is None: self.x_prev = raw
            filtered = 0.6 * raw + 0.4 * self.x_prev
            self.x_prev = filtered
            return filtered
        return None

# ================= 4. 主程序 =================
def main():
    cam = CameraStream(src=0, width=2560, height=720).start()
    time.sleep(1.0)
    
    proc_l, proc_r = HandProcessor(), HandProcessor()
    visualizer = HandVisualizer3D()
    W_RAW, H_RAW = 1280, 720
    # 旋转后的图像尺寸
    W_ROT, H_ROT = 720, 1280  # 逆时针90°: (H_RAW, W_RAW)

    print("=== 开始手势跟踪 ===")
    print("按 'q' 退出")

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue
        if frame.shape[1] != 2560: continue

        # 分割图像
        img_l_raw = frame[:, :1280]
        img_r_raw = frame[:, 1280:]

        # 旋转
        img_l_rot = cv2.rotate(img_l_raw, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_r_rot = cv2.rotate(img_r_raw, cv2.ROTATE_90_CLOCKWISE)
        
        # 转RGB供检测
        img_l_rgb = cv2.cvtColor(img_l_rot, cv2.COLOR_BGR2RGB)
        img_r_rgb = cv2.cvtColor(img_r_rot, cv2.COLOR_BGR2RGB)
        
        pts_l = proc_l.process(img_l_rgb)
        pts_r = proc_r.process(img_r_rgb)

        if pts_l is not None and pts_r is not None:
            # === 1. 坐标映射 - 从旋转后的图像坐标映射回原始相机坐标 ===
            # 左相机：逆时针旋转90°
            # 旋转后的x(0~720)对应原始的v(0~720)，但方向反向: v = (1-x)*720
            # 旋转后的y(0~1280)对应原始的u(0~1280)，方向反向: u = (1-y)*1280
            u_l = (1.0 - pts_l[:, 1]) * W_RAW      # 映射到u轴 [0, 1280]
            v_l = (1.0 - pts_l[:, 0]) * H_RAW      # 映射到v轴 [0, 720] ← 修复！
            px_l = np.column_stack((u_l, v_l))

            # 右相机：顺时针旋转90°
            # 旋转后的x(0~720)对应原始的v，方向相同: v = x*720
            # 旋转后的y(0~1280)对应原始的u，方向相同: u = y*1280
            u_r = pts_r[:, 1] * W_RAW              # 映射到u轴 [0, 1280]
            v_r = pts_r[:, 0] * H_RAW              # 映射到v轴 [0, 720]
            px_r = np.column_stack((u_r, v_r))

            # === 2. 3D 重建 ===
            ud_l = cv2.undistortPoints(px_l.reshape(-1, 1, 2), K1, D1, P=K1)
            ud_r = cv2.undistortPoints(px_r.reshape(-1, 1, 2), K2, D2, P=K2)
            pts_4d = cv2.triangulatePoints(P1, P2, ud_l.reshape(-1, 2).T, ud_r.reshape(-1, 2).T)
            pts_3d = (pts_4d[:3] / pts_4d[3]).T 
            
            # === 坐标系调整 ===
            # X轴反向（因为旋转变换导致的方向问题）
            pts_3d[:, 0] = -pts_3d[:, 0]
            # Y轴反向（使Y正方向向上）
            pts_3d[:, 1] = -pts_3d[:, 1]

            visualizer.update(pts_3d)

            # === 3. 2D 显示 ===
            h_rot, w_rot = img_l_rot.shape[:2]
            for p in pts_l:
                cx, cy = int(p[0] * w_rot), int(p[1] * h_rot)
                cv2.circle(img_l_rot, (cx, cy), 4, (0, 255, 0), -1)
            
            for conn in mp.solutions.hands.HAND_CONNECTIONS:
                start_idx = conn[0]
                end_idx = conn[1]
                p1 = pts_l[start_idx]
                p2 = pts_l[end_idx]
                pt1 = (int(p1[0] * w_rot), int(p1[1] * h_rot))
                pt2 = (int(p2[0] * w_rot), int(p2[1] * h_rot))
                cv2.line(img_l_rot, pt1, pt2, (0, 255, 0), 1)

        cv2.imshow('Tracking (Left)', cv2.resize(img_l_rot, (0,0), fx=0.5, fy=0.5))

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cam.stop()
    cv2.destroyAllWindows()
    plt.close()

if __name__ == "__main__":
    main()
