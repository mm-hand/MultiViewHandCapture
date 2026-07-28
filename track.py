import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time

import cv2
import numpy as np

from config import (
    CAMERA_TYPE,
    CAMERA_INDEX,
    KEYPOINT_LAYOUT,
    KEYPOINT_TOPIC,
    ROBOT_LAYOUT,
    ROBOT_TOPIC,
    SKELETON_EDGES,
    URDF_PATH,
    WEB_FPS,
    WEB_PORT,
    VIEW_PORTS,
)
from hand_core import Camera, RealSenseCamera, StereoProcessor
from retarget import Retargeter


def _overlay(image, points):
    image = image.copy()
    if points is not None:
        for start, end in SKELETON_EDGES:
            cv2.line(image, tuple(points[start].astype(int)), tuple(points[end].astype(int)), (0, 255, 0), 2)
        for point in points.astype(int):
            cv2.circle(image, tuple(point), 3, (255, 60, 60), -1)
    return cv2.resize(image, (640, 360))


class Viewer:
    def __init__(self, model=None):
        import viser
        from viser.extras import ViserUrdf

        self.servers = [viser.ViserServer(port=port) for port in VIEW_PORTS]
        normalized, robot_server = self.servers
        self._camera(normalized, (0.24, 0, 0.05), (0, 0, 0.05), (0, 0, 1), 42)
        self._camera(robot_server, (0.28, -0.32, 0.22), (0, 0, 0.06), (0, 0, 1), 42)
        self.handles = {"normalized": self._skeleton(normalized, (60, 170, 255), 0.004)}
        self.urdf = self.urdf_names = None
        if model is not None:
            robot_server.scene.add_frame("/robot", show_axes=False)
            self.urdf = ViserUrdf(robot_server, URDF_PATH, root_node_name="/robot")
            self.urdf_names = self.urdf.get_actuated_joint_names()
            self.robot_index = model.index
            self.urdf.update_cfg(np.zeros(len(self.urdf_names)))
        self.last_update, self.status = 0.0, "WAITING"
        self.stereo = cv2.imencode(".jpg", np.zeros((360, 1280, 3), np.uint8))[1].tobytes()
        self._start_dashboard()
        print(f"Dashboard: http://localhost:{WEB_PORT}")

    @staticmethod
    def _camera(server, position, look_at, up, fov):
        server.scene.set_up_direction("+z")
        server.gui.configure_theme(show_logo=False, show_share_button=False)
        server.initial_camera.position = position
        server.initial_camera.look_at = look_at
        server.initial_camera.up = up
        server.initial_camera.fov = np.radians(fov)

    @staticmethod
    def _skeleton(server, color, point_size):
        points = np.zeros((21, 3), np.float32)
        cloud = server.scene.add_point_cloud(
            "/points", points, color, point_size=point_size, point_shape="circle", visible=False
        )
        lines = server.scene.add_line_segments(
            "/bones", points[np.asarray(SKELETON_EDGES)], color, line_width=4, visible=False
        )
        return cloud, lines

    def _start_dashboard(self):
        owner = self
        normalized_port, robot_port = [server.get_port() for server in self.servers]
        page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>MultiView Hand Capture</title><style>
*{{box-sizing:border-box}} body{{margin:0;height:100vh;background:#101318;color:#eef;
font:15px sans-serif;display:grid;grid-template:40vh 60vh/repeat(2,1fr);gap:6px}}
.panel{{position:relative;overflow:hidden;background:#181d25;border:1px solid #343b48}}
.stereo{{grid-column:1/3}} h2{{position:absolute;z-index:2;margin:0;padding:8px 12px;
font-size:15px;background:#101318cc;border-radius:0 0 6px 0}}
img,iframe{{width:100%;height:100%;display:block;border:0;object-fit:contain}}
#status{{color:#9bd;margin-left:12px;font-weight:normal}}</style></head><body>
<section class="panel stereo"><h2>Stereo + MediaPipe <span id="status">WAITING</span></h2>
<img id="stereo"></section>
<section class="panel"><h2>Normalized hand · wrist frame · 0.086 m</h2><iframe id="normalized"></iframe></section>
<section class="panel"><h2>Retargeted MMHand</h2><iframe id="robot"></iframe></section>
<script>
const host=location.hostname, statusText=document.getElementById("status"),
stereoImage=document.getElementById("stereo");
for(const [id,port] of [["normalized",{normalized_port}],["robot",{robot_port}]])
  document.getElementById(id).src=`http://${{host}}:${{port}}`;
setInterval(()=>{{stereoImage.src="/stereo.jpg?t="+Date.now();
fetch("/status").then(r=>r.text()).then(x=>statusText.textContent=x)}},{round(1000 / WEB_FPS)});
</script></body></html>""".encode()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path == "/stereo.jpg":
                    body, content_type = owner.stereo, "image/jpeg"
                elif path == "/status":
                    body, content_type = owner.status.encode(), "text/plain; charset=utf-8"
                elif path == "/":
                    body, content_type = page, "text/html; charset=utf-8"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_):
                pass

        self.http = ThreadingHTTPServer(("0.0.0.0", WEB_PORT), Handler)
        self.http_thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.http_thread.start()

    def _update_skeleton(self, name, points):
        cloud, lines = self.handles[name]
        visible = points is not None
        cloud.visible = lines.visible = visible
        if visible:
            points = np.asarray(points, np.float32)
            cloud.points = points
            lines.points = points[np.asarray(SKELETON_EDGES)]

    def update(self, result, robot=None):
        now = time.monotonic()
        if now - self.last_update < 1 / WEB_FPS:
            return
        self.last_update = now
        quality = result["quality"]
        detail = f" · rejected: {quality['rejected_reason']}" if quality["rejected_reason"] else ""
        if quality["reprojection_error"] is not None:
            detail += f" · reprojection: {quality['reprojection_error']:.1f} px"
        self.status = f"{result['phase']}{detail}"
        views = [
            _overlay(image, points) if image is not None else np.zeros((360, 640, 3), np.uint8)
            for image, points in zip(
                (result["image_left"], result["image_right"]),
                (result["px_left"], result["px_right"]),
            )
        ]
        ok, encoded = cv2.imencode(".jpg", np.hstack(views), (cv2.IMWRITE_JPEG_QUALITY, 82))
        if ok:
            self.stereo = encoded.tobytes()
        self._update_skeleton("normalized", result["keypoint_relative"])
        if robot is not None and self.urdf is not None:
            self.urdf.update_cfg(np.asarray([robot[self.robot_index[name]] for name in self.urdf_names]))

    def close(self):
        self.http.shutdown()
        self.http.server_close()
        for server in self.servers:
            server.stop()


class RosOutput:
    def __init__(self, mode):
        import rclpy
        from std_msgs.msg import Float32MultiArray

        rclpy.init(args=None)
        self.rclpy, self.message = rclpy, Float32MultiArray
        self.node = rclpy.create_node("multiview_hand_capture")
        topic = KEYPOINT_TOPIC if mode == "points" else ROBOT_TOPIC
        self.publisher = self.node.create_publisher(Float32MultiArray, topic, 1)
        print(f"ROS 2: publishing {topic}")

    def publish(self, values, label, shape):
        from std_msgs.msg import MultiArrayDimension, MultiArrayLayout

        dimensions, stride = [], int(np.prod(shape))
        for name, size in zip(("keypoint", "xyz") if len(shape) == 2 else ("joint",), shape):
            dimensions.append(MultiArrayDimension(label=label if not dimensions else name, size=size, stride=stride))
            stride //= size
        message = self.message()
        message.layout = MultiArrayLayout(dim=dimensions, data_offset=0)
        message.data = np.asarray(values, np.float32).ravel().tolist()
        self.publisher.publish(message)

    def points(self, points, handedness):
        self.publish(points, f"{KEYPOINT_LAYOUT}:hand={handedness}", (21, 3))

    def joints(self, degrees):
        self.publish(degrees, ROBOT_LAYOUT, (21,))

    def close(self):
        self.node.destroy_node()
        self.rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("points", "retarget"), default="points")
    parser.add_argument("--ros", action="store_true")
    args = parser.parse_args()

    retargeter = Retargeter() if args.mode == "retarget" else None
    if CAMERA_TYPE == "d435":
        camera = RealSenseCamera(CAMERA_INDEX)
        processor = StereoProcessor(camera.params)
    elif CAMERA_TYPE == "stereo":
        camera, processor = Camera(CAMERA_INDEX), StereoProcessor()
    else:
        raise ValueError("CAMERA_TYPE must be 'd435' or 'stereo'")
    viewer = Viewer(None if retargeter is None else retargeter.model)
    ros = RosOutput(args.mode) if args.ros else None
    last_timestamp = None
    try:
        while True:
            ok, left, right, timestamp = camera.read()
            if not ok or timestamp == last_timestamp:
                time.sleep(0.002)
                continue
            last_timestamp = timestamp
            result = processor.process(left, right, timestamp)
            valid = result["found"] and not result["stale"] and result["keypoint_relative"] is not None
            robot = None
            if retargeter is not None:
                if valid and result["handedness"] == "Left" and result["phase"].startswith("GESTURE"):
                    robot = retargeter.solve(result["keypoint_relative"])
                    if ros is not None and robot is not None:
                        ros.joints(np.degrees(robot))
                else:
                    retargeter.pause()
            elif ros is not None and valid:
                ros.points(result["keypoint_relative"], result["handedness"])
            viewer.update(result, robot)
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        processor.close()
        viewer.close()
        if ros is not None:
            ros.close()


if __name__ == "__main__":
    main()
