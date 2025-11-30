import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
import sys
import time

# ================= 0. 标定参数 (保持不变) =================
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

# ================= 1. 多线程相机读取类 (保持不变) =================
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

# ================= 2. 3D 可视化类 (已更新：双视图) =================
class HandVisualizer3D:
    def __init__(self):
        plt.ion()
        # 创建宽一点的窗口，容纳两个子图
        self.fig = plt.figure(figsize=(14, 7))
        
        # === 视图 1: 正面 (XY 平面) ===
        self.ax1 = self.fig.add_subplot(121, projection='3d')
        self.ax1.set_title("Front View (XY Plane)")
        # elev=-90, azim=-90: 模拟从Z轴正看向原点，X轴向右，Y轴向下(因为Y数据被翻转了)
        self.ax1.view_init(elev=-90, azim=-90) 
        
        # === 视图 2: 侧面 (YZ 平面) ===
        self.ax2 = self.fig.add_subplot(122, projection='3d')
        self.ax2.set_title("Side View (YZ Plane - Depth)")
        # elev=0, azim=0: 从侧面看，主要观察 Y(高低) 和 Z(深度) 的关系
        self.ax2.view_init(elev=0, azim=0)

        self.axes = [self.ax1, self.ax2]
        self.conn = list(mp.solutions.hands.HAND_CONNECTIONS)
        
        # 为每个视图分别存储散点和线段对象
        self.scats = []
        self.lines_collections = []

        for ax in self.axes:
            # 1. 设置坐标轴范围 (根据实际情况调整)
            ax.set_xlim(-150, 150) # X (左右)
            ax.set_ylim(-150, 150) # Y (上下)
            ax.set_zlim(200, 500)  # Z (深度，通常在300-500mm之间)
            
            ax.set_xlabel('X (Right)')
            ax.set_ylabel('Y (Down)')
            ax.set_zlabel('Z (Forward)')

            # 2. 初始化散点
            scat = ax.scatter([], [], [], c='r', s=25, depthshade=True)
            self.scats.append(scat)

            # 3. 初始化线段
            lines = [ax.plot([], [], [], 'b-', linewidth=2)[0] for _ in range(len(self.conn))]
            self.lines_collections.append(lines)

    def update(self, p3d):
        try:
            # 同时更新两个视图
            for i in range(2): 
                # 更新散点
                self.scats[i]._offsets3d = (p3d[:, 0], p3d[:, 1], p3d[:, 2])
                
                # 更新线段
                lines = self.lines_collections[i]
                for line, (start, end) in zip(lines, self.conn):
                    line.set_data([p3d[start, 0], p3d[end, 0]], [p3d[start, 1], p3d[end, 1]])
                    line.set_3d_properties([p3d[start, 2], p3d[end, 2]])
            
            plt.pause(0.001) 
        except Exception:
            pass

# ================= 3. 手势处理器 (保持不变) =================
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

# ================= 4. 主程序 (保持修正后的逻辑) =================
def main():
    # 启动相机
    cam = CameraStream(src=0, width=2560, height=720).start()
    time.sleep(1.0)

    proc_l, proc_r = HandProcessor(), HandProcessor()
    
    # 初始化双视图可视化器
    visualizer = HandVisualizer3D()
    
    # 原始单目尺寸
    W_RAW, H_RAW = 1280, 720
    
    print("=== 开始手势跟踪 (双视图版) ===")
    print("左图: 正面 (XY) | 右图: 侧面深度 (YZ)")
    print("按 'q' 退出")

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue
        if frame.shape[1] != 2560: continue

        # 1. 直接分割图像，不做旋转
        img_l = frame[:, :1280]
        img_r = frame[:, 1280:]

        # 2. 转 RGB 供 MediaPipe 检测
        img_l_rgb = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
        img_r_rgb = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)

        # 3. 推理
        pts_l = proc_l.process(img_l_rgb)
        pts_r = proc_r.process(img_r_rgb)

        if pts_l is not None and pts_r is not None:
            # === 1. 坐标映射 ===
            # 左相机
            u_l = pts_l[:, 0] * W_RAW
            v_l = pts_l[:, 1] * H_RAW
            px_l = np.column_stack((u_l, v_l))

            # 右相机
            u_r = pts_r[:, 0] * W_RAW
            v_r = pts_r[:, 1] * H_RAW
            px_r = np.column_stack((u_r, v_r))

            # === 2. 3D 重建 ===
            ud_l = cv2.undistortPoints(px_l.reshape(-1, 1, 2), K1, D1, P=K1)
            ud_r = cv2.undistortPoints(px_r.reshape(-1, 1, 2), K2, D2, P=K2)
            
            pts_4d = cv2.triangulatePoints(P1, P2, ud_l.reshape(-1, 2).T, ud_r.reshape(-1, 2).T)
            pts_3d = (pts_4d[:3] / pts_4d[3]).T 

            # === 坐标系调整 ===
            # OpenCV 原始坐标系: X右, Y下, Z前
            # 为了可视化直观，我们将Y轴取反，变成 Y上
            pts_3d[:, 1] = -pts_3d[:, 1]  

            # 更新可视化 (Visualizer内部会处理两个视图)
            visualizer.update(pts_3d)

            # === 3. 2D 显示 ===
            for p in px_l:
                cv2.circle(img_l, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)

            for conn in mp.solutions.hands.HAND_CONNECTIONS:
                p1 = px_l[conn[0]]
                p2 = px_l[conn[1]]
                cv2.line(img_l, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 1)

        # 显示
        cv2.imshow('Tracking (Left)', cv2.resize(img_l, (0,0), fx=0.6, fy=0.6))

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cam.stop()
    cv2.destroyAllWindows()
    plt.close()

if __name__ == "__main__":
    main()
