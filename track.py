import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
import os
import sys
import time
import json

# ================= 0. 标定参数  =================
JSON_PATH = "stereo_params.json"

if not os.path.exists(JSON_PATH):
    print(f"❌ 错误: 找不到标定文件 {JSON_PATH}")
    print("请先运行 calibrate.py 进行标定！")
    sys.exit()

print(f"✅ 正在加载标定文件: {JSON_PATH}")
with open(JSON_PATH, 'r') as f:
    params = json.load(f)

# 将 list 转换为 numpy array
K1 = np.array(params["K1"], dtype=np.float64)
D1 = np.array(params["D1"], dtype=np.float64)
K2 = np.array(params["K2"], dtype=np.float64)
D2 = np.array(params["D2"], dtype=np.float64)
R  = np.array(params["R"],  dtype=np.float64)
T  = np.array(params["T"],  dtype=np.float64)

# 计算投影矩阵
P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = K2 @ np.hstack((R, T))

print("标定参数加载完成。")
# ================= 1. 一欧元滤波器 (One Euro Filter) =================
class OneEuroFilter:
    def __init__(self, freq, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        if freq <= 0: raise ValueError("freq must be positive")
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = None
        self.is_init = False
        self.t_e = 1.0 / self.freq

    def _smoothing_factor(self, t_e, cutoff):
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)

    def _exponential_smoothing(self, a, x, x_prev):
        return a * x + (1.0 - a) * x_prev

    def __call__(self, x):
        if not self.is_init:
            self.is_init = True
            self.x_prev = x
            if isinstance(x, np.ndarray):
                self.dx_prev = np.zeros_like(x)
            else:
                self.dx_prev = 0.0
            return x.copy() if isinstance(x, np.ndarray) else x

        dx = (x - self.x_prev) / self.t_e
        a_d = self._smoothing_factor(self.t_e, self.d_cutoff)
        dx_hat = self._exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._smoothing_factor(self.t_e, cutoff)
        x_hat = self._exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.is_init = False

# ================= 2. 多线程相机读取类 =================
class CameraStream:
    def __init__(self, src=0, width=2560, height=720):
        self.stream = cv2.VideoCapture(src, cv2.CAP_V4L2)
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

# ================= 3. 3D 可视化类 (修正比例和深度) =================
class HandVisualizer3D:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(14, 7))
        
        # 视图 1: 正面 (XY 平面)
        self.ax1 = self.fig.add_subplot(121, projection='3d')
        self.ax1.set_title("Front View (XY)")
        self.ax1.view_init(elev=-90, azim=-90) 
        
        # 视图 2: 侧面 (YZ 平面)
        self.ax2 = self.fig.add_subplot(122, projection='3d')
        self.ax2.set_title("Side View (Depth YZ)")
        self.ax2.view_init(elev=0, azim=0)

        self.axes = [self.ax1, self.ax2]
        self.conn = list(mp.solutions.hands.HAND_CONNECTIONS)
        
        self.scats = []
        self.lines_collections = []

        for ax in self.axes:
            # === 设置坐标轴范围 ===
            # X: -150 ~ 150 (跨度300)
            # Y: -150 ~ 150 (跨度300)
            # Z:  100 ~ 400 (跨度300)
            ax.set_xlim(-150, 150)
            ax.set_ylim(-150, 150)
            ax.set_zlim(100, 400)
            
            # === 强制等比例 ===
            # 因为三个轴的跨度都是300，设置aspect为(1,1,1)可保证视觉上不变形
            ax.set_box_aspect((1, 1, 1))
            
            ax.set_xlabel('X (Right)')
            ax.set_ylabel('Y (Down)')
            ax.set_zlabel('Z (Forward)')

            scat = ax.scatter([], [], [], c='r', s=25, depthshade=True)
            self.scats.append(scat)

            lines = [ax.plot([], [], [], 'b-', linewidth=2)[0] for _ in range(len(self.conn))]
            self.lines_collections.append(lines)

    def update(self, p3d):
        try:
            for i in range(2): 
                self.scats[i]._offsets3d = (p3d[:, 0], p3d[:, 1], p3d[:, 2])
                lines = self.lines_collections[i]
                for line, (start, end) in zip(lines, self.conn):
                    line.set_data([p3d[start, 0], p3d[end, 0]], [p3d[start, 1], p3d[end, 1]])
                    line.set_3d_properties([p3d[start, 2], p3d[end, 2]])
            plt.pause(0.001) 
        except Exception:
            pass

# ================= 4. 手势处理器 (含 OneEuroFilter) =================
class HandProcessor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands.Hands(
            max_num_hands=1, 
            min_detection_confidence=0.4, 
            min_tracking_confidence=0.4
        )
        # freq=30, min_cutoff=0.1 (静止时稳), beta=5.0 (移动时快)
        self.filter = OneEuroFilter(freq=30, min_cutoff=0.1, beta=5.0, d_cutoff=1.0)

    def process(self, img):
        res = self.mp_hands.process(img)
        if res.multi_hand_landmarks:
            raw = np.array([[lm.x, lm.y] for lm in res.multi_hand_landmarks[0].landmark])
            filtered = self.filter(raw)
            return filtered
        else:
            self.filter.reset()
            return None

# ================= 5. 主程序 =================
def main():
    cam = CameraStream(src=0, width=2560, height=720).start()
    time.sleep(1.0)

    proc_l, proc_r = HandProcessor(), HandProcessor()
    visualizer = HandVisualizer3D()
    
    W_RAW, H_RAW = 1280, 720
    
    print("=== 双目手势跟踪 ===")
    print("按 'q' 退出")

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue
        if frame.shape[1] != 2560: continue

        # 1. 图像分割
        img_l = frame[:, :1280]
        img_r = frame[:, 1280:]

        # 2. 转 RGB
        img_l_rgb = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
        img_r_rgb = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)

        # 3. 处理
        pts_l = proc_l.process(img_l_rgb)
        pts_r = proc_r.process(img_r_rgb)

        if pts_l is not None and pts_r is not None:
            # === 坐标映射 ===
            u_l, v_l = pts_l[:, 0] * W_RAW, pts_l[:, 1] * H_RAW
            px_l = np.column_stack((u_l, v_l))

            u_r, v_r = pts_r[:, 0] * W_RAW, pts_r[:, 1] * H_RAW
            px_r = np.column_stack((u_r, v_r))

            # === 3D 重建 ===
            ud_l = cv2.undistortPoints(px_l.reshape(-1, 1, 2), K1, D1, P=K1)
            ud_r = cv2.undistortPoints(px_r.reshape(-1, 1, 2), K2, D2, P=K2)
            pts_4d = cv2.triangulatePoints(P1, P2, ud_l.reshape(-1, 2).T, ud_r.reshape(-1, 2).T)
            pts_3d = (pts_4d[:3] / pts_4d[3]).T 

            visualizer.update(pts_3d)

            # === 2D 绘制 (左右眼都画) ===
            # 画左图
            for p in px_l:
                cv2.circle(img_l, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)
            for conn in mp.solutions.hands.HAND_CONNECTIONS:
                p1, p2 = px_l[conn[0]], px_l[conn[1]]
                cv2.line(img_l, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 2)
            
            # 画右图
            for p in px_r:
                cv2.circle(img_r, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)
            for conn in mp.solutions.hands.HAND_CONNECTIONS:
                p1, p2 = px_r[conn[0]], px_r[conn[1]]
                cv2.line(img_r, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 2)

        # === 显示逻辑优化 ===
        # 1. 横向拼接
        combined = np.hstack((img_l, img_r))
        
        # 2. 缩放到 1/5 尺寸
        # 原尺寸 2560 x 720 -> 目标尺寸 512 x 144
        vis_frame = cv2.resize(combined, (0, 0), fx=0.2, fy=0.2)

        cv2.imshow('Stereo Tracker', vis_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cam.stop()
    cv2.destroyAllWindows()
    plt.close()

if __name__ == "__main__":
    main()
