import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  # 引入 GridSpec 用于布局调整
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

# ================= 3. 全能可视化类 (布局优化版) =================
class HandVisualizerAllInOne:
    def __init__(self, w=1280, h=720):
        plt.ion()
        # 调整画布大小，适应垂直布局
        self.fig = plt.figure(figsize=(16, 12))
        self.w = w
        self.h = h
        self.conn = list(mp.solutions.hands.HAND_CONNECTIONS)

        # === 使用 GridSpec 定义不均匀的布局 ===
        # 2行2列
        # height_ratios=[1, 3] 表示第二行的高度是第一行的 3 倍
        # hspace=0.1, wspace=0.1 减少子图间距
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 3], hspace=0.1, wspace=0.1)

        # --- 1. 左上: 左眼 2D (较小) ---
        self.ax_l = self.fig.add_subplot(gs[0, 0])
        self.ax_l.set_title("Left Camera (2D)", fontsize=10)
        self.ax_l.axis('off') # 关闭坐标轴，看起来更像监控画面
        self.im_l_disp = self.ax_l.imshow(np.zeros((h, w, 3), dtype=np.uint8))
        self.lines_2d_l = [self.ax_l.plot([], [], 'g-', linewidth=1)[0] for _ in self.conn]
        self.points_2d_l = self.ax_l.plot([], [], 'r.', markersize=3)[0]

        # --- 2. 右上: 右眼 2D (较小) ---
        self.ax_r = self.fig.add_subplot(gs[0, 1])
        self.ax_r.set_title("Right Camera (2D)", fontsize=10)
        self.ax_r.axis('off')
        self.im_r_disp = self.ax_r.imshow(np.zeros((h, w, 3), dtype=np.uint8))
        self.lines_2d_r = [self.ax_r.plot([], [], 'g-', linewidth=1)[0] for _ in self.conn]
        self.points_2d_r = self.ax_r.plot([], [], 'r.', markersize=3)[0]

        # --- 3. 左下: 3D 正视图 (较大) ---
        self.ax3d_front = self.fig.add_subplot(gs[1, 0], projection='3d')
        self.ax3d_front.set_title("3D Reconstruction - Front (XY)", fontsize=12)
        self.ax3d_front.view_init(elev=-90, azim=-90)
        self._init_3d_axis(self.ax3d_front)

        # --- 4. 右下: 3D 侧视图 (较大) ---
        self.ax3d_side = self.fig.add_subplot(gs[1, 1], projection='3d')
        self.ax3d_side.set_title("3D Reconstruction - Side (Depth YZ)", fontsize=12)
        self.ax3d_side.view_init(elev=0, azim=0)
        self._init_3d_axis(self.ax3d_side)

        # 存储3D对象引用
        self.scats_3d = []
        self.lines_3d_collections = []
        for ax in [self.ax3d_front, self.ax3d_side]:
            # 加大3D点的尺寸 s=40
            scat = ax.scatter([], [], [], c='r', s=40, depthshade=True)
            self.scats_3d.append(scat)
            # 加粗3D连线 linewidth=3
            lines = [ax.plot([], [], [], 'b-', linewidth=3)[0] for _ in range(len(self.conn))]
            self.lines_3d_collections.append(lines)
        
        # 紧凑布局
        self.fig.tight_layout()

    def _init_3d_axis(self, ax):
        # 坐标轴范围设置
        ax.set_xlim(-150, 150)
        ax.set_ylim(-150, 150)
        ax.set_zlim(100, 400)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    def update(self, img_l, img_r, pts_2d_l, pts_2d_r, pts_3d):
        try:
            # 1. 更新图像背景 (缩小显示分辨率以提速)
            disp_h, disp_w = 360, 640
            img_l_small = cv2.resize(img_l, (disp_w, disp_h))
            img_r_small = cv2.resize(img_r, (disp_w, disp_h))
            
            self.im_l_disp.set_data(img_l_small)
            self.im_r_disp.set_data(img_r_small)
            
            # 确保imshow的extent正确，否则点会画偏
            self.im_l_disp.set_extent([0, self.w, self.h, 0])
            self.im_r_disp.set_extent([0, self.w, self.h, 0])

            # 2. 更新 2D 骨架
            self._update_2d_skeleton(self.lines_2d_l, self.points_2d_l, pts_2d_l)
            self._update_2d_skeleton(self.lines_2d_r, self.points_2d_r, pts_2d_r)

            # 3. 更新 3D 骨架
            if pts_3d is not None:
                for i in range(2):
                    self.scats_3d[i]._offsets3d = (pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2])
                    lines = self.lines_3d_collections[i]
                    for line, (start, end) in zip(lines, self.conn):
                        line.set_data([pts_3d[start, 0], pts_3d[end, 0]], 
                                      [pts_3d[start, 1], pts_3d[end, 1]])
                        line.set_3d_properties([pts_3d[start, 2], pts_3d[end, 2]])
            else:
                for i in range(2):
                    self.scats_3d[i]._offsets3d = ([], [], [])
                    for line in self.lines_3d_collections[i]:
                        line.set_data([], [])
                        line.set_3d_properties([])

            plt.pause(0.001)
        except Exception as e:
            # 当窗口关闭时可能会报错，忽略
            pass

    def _update_2d_skeleton(self, lines_objs, points_obj, pts):
        if pts is not None:
            points_obj.set_data(pts[:, 0], pts[:, 1])
            for line, (start, end) in zip(lines_objs, self.conn):
                line.set_data([pts[start, 0], pts[end, 0]], 
                              [pts[start, 1], pts[end, 1]])
        else:
            points_obj.set_data([], [])
            for line in lines_objs:
                line.set_data([], [])

# ================= 4. 手势处理器 (含 OneEuroFilter) =================
class HandProcessor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands.Hands(
            max_num_hands=1, 
            min_detection_confidence=0.4, 
            min_tracking_confidence=0.4
        )
        self.filter = OneEuroFilter(freq=30, min_cutoff=0.1, beta=5.0, d_cutoff=1.0)

    def process(self, img):
        res = self.mp_hands.process(img)
        if res.multi_hand_landmarks:
            # <--- 修改: 这里增加提取 lm.z，结果为 (21, 3) --->
            raw = np.array([[lm.x, lm.y, lm.z] for lm in res.multi_hand_landmarks[0].landmark])
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
    visualizer = HandVisualizerAllInOne(w=1280, h=720)

    W_RAW, H_RAW = 1280, 720

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue
        if frame.shape[1] != 2560: continue

        img_l = frame[:, :1280]
        img_r = frame[:, 1280:]

        img_l_rgb = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
        img_r_rgb = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)

        # pts_norm 现在包含 (x, y, z)，其中 z 是 MediaPipe 相对深度
        pts_norm_l = proc_l.process(img_l_rgb)
        pts_norm_r = proc_r.process(img_r_rgb)

        px_l, px_r = None, None
        pts_3d = None

        # 提取像素坐标 (只用 x, y 计算)
        if pts_norm_l is not None:
            u_l, v_l = pts_norm_l[:, 0] * W_RAW, pts_norm_l[:, 1] * H_RAW
            px_l = np.column_stack((u_l, v_l))

        if pts_norm_r is not None:
            u_r, v_r = pts_norm_r[:, 0] * W_RAW, pts_norm_r[:, 1] * H_RAW
            px_r = np.column_stack((u_r, v_r))

        if px_l is not None and px_r is not None:
            # 1. 原始双目三角测量
            ud_l = cv2.undistortPoints(px_l.reshape(-1, 1, 2), K1, D1, P=K1)
            ud_r = cv2.undistortPoints(px_r.reshape(-1, 1, 2), K2, D2, P=K2)
            pts_4d = cv2.triangulatePoints(P1, P2, ud_l.reshape(-1, 2).T, ud_r.reshape(-1, 2).T)
            pts_3d_stereo = (pts_4d[:3] / pts_4d[3]).T # 原始绝对坐标
            
            # 提取 MediaPipe 左手相对深度 Z_mp
            z_mp = pts_norm_l[:, 2]
            # 提取双目计算的绝对深度 Z_stereo
            z_stereo = pts_3d_stereo[:, 2]

            # 计算映射关系 y = kx + b
            # 使用 np.polyfit 进行线性回归 (1次多项式)
            # 目的是找到 Z_stereo = k * Z_mp + b
            try:
                k, b = np.polyfit(z_mp, z_stereo, 1)
                
                # 使用映射关系重新计算绝对深度
                z_new = k * z_mp + b
                
                # 更新 pts_3d (保留双目的X,Y，替换Z)
                pts_3d = pts_3d_stereo.copy()
                pts_3d[:, 2] = z_new
            except Exception as e:
                # 如果回归失败，回退到原始双目结果
                pts_3d = pts_3d_stereo

        visualizer.update(img_l_rgb, img_r_rgb, px_l, px_r, pts_3d)

        if not plt.fignum_exists(visualizer.fig.number):
            break

    cam.stop()
    plt.close()

if __name__ == "__main__":
    main()