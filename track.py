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
    PARAMS_PATH,
    ROBOT_LAYOUT,
    ROBOT_JOINT_NAMES,
    ROBOT_TOPIC,
    SKELETON_EDGES,
    URDF_PATH,
    WEB_FPS,
    WEB_PORT,
    VIEW_PORTS,
)
from hand_core import Camera, RealSenseCamera, StereoProcessor
from retarget import Retargeter
from teleop_maniskill.teleop_protocol import UdpRetargetSender


def _udp_quality(quality):
    reprojection = quality.get("reprojection_error")
    if reprojection is not None:
        reprojection = float(reprojection)
        if not np.isfinite(reprojection):
            reprojection = None
    reason = quality.get("rejected_reason")
    return {
        "reprojection_error": reprojection,
        "rejected_reason": None if reason is None else str(reason),
    }


def _camera_configuration_error(camera_type=CAMERA_TYPE, params_path=PARAMS_PATH):
    if camera_type not in {"d435", "stereo"}:
        return "CAMERA_TYPE must be 'd435' or 'stereo'"
    if camera_type == "stereo" and not params_path.is_file():
        return (
            f"stereo calibration not found: {params_path}; run calibrate.py "
            "or set CAMERA_TYPE='d435' in config.py"
        )
    return None

ROBOT_PAD_ARROW_LENGTH = 0.025
ROBOT_CAMERA_DISTANCE = 0.42


def _human_view_wxyz(handedness, points):
    """Face the palm toward -X and level the Index-to-Little MCP line."""
    right = handedness == "Right"
    points = np.asarray(points, float)
    side = points[5] - points[17]
    if right:
        side = side * (-1, -1, 1)
    roll = -np.arctan2(side[2], side[1])
    roll = (roll + np.pi / 2) % np.pi - np.pi / 2
    cosine, sine = np.cos(roll / 2), np.sin(roll / 2)
    return (0.0, 0.0, -sine, cosine) if right else (cosine, sine, 0.0, 0.0)


def _robot_camera_pose(model):
    zero = np.zeros(21)
    tips, _ = model.fingertip_pads(zero)
    look_at = np.vstack((model.palm_position, tips)).mean(0)
    palm_normal = model.palm_frame[:, 0]
    transforms = model.fk(zero)
    base = (
        transforms["finger_1_proximal_phalanx_1"][:3, 3]
        - transforms["finger_4_proximal_phalanx_1"][:3, 3]
    )
    normal = palm_normal - base * np.dot(palm_normal, base) / np.dot(base, base)
    normal /= np.linalg.norm(normal)
    return look_at - ROBOT_CAMERA_DISTANCE * normal, look_at, (
        0.9993984744, -0.0346479515, -0.0014862239
    )


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
        self.model = model
        self._camera(
            normalized,
            (-0.26, 0, 0.06),
            (0, 0, 0.06),
            (0, 0.2588190451, 0.9659258263),
            42,
        )
        robot_camera = (
            _robot_camera_pose(model)
            if model is not None
            else ((0.28, -0.32, 0.22), (0, 0, 0.06), (0, 0, 1))
        )
        self._camera(robot_server, *robot_camera, 42)
        self.human_frame = normalized.scene.add_frame("/hand", show_axes=False)
        empty_hand = np.zeros((21, 3), np.float32)
        self.human_cloud = normalized.scene.add_point_cloud(
            "/hand/points", empty_hand, (60, 170, 255),
            point_size=0.004, point_shape="circle", visible=False,
        )
        self.human_bones = normalized.scene.add_line_segments(
            "/hand/bones", empty_hand[np.asarray(SKELETON_EDGES)],
            (60, 170, 255), line_width=4, visible=False,
        )
        self.urdf = self.urdf_names = self.robot_pad_arrows = None
        if model is not None:
            robot_server.scene.add_frame("/robot", show_axes=False)
            self.urdf = ViserUrdf(robot_server, URDF_PATH, root_node_name="/robot")
            self.robot_pad_arrows = robot_server.scene.add_arrows(
                "/robot/pad_directions", np.zeros((5, 2, 3), np.float32),
                (255, 145, 45), shaft_radius=0.0012, head_radius=0.003,
                head_length=0.006, visible=False,
            )
            self.urdf_names = self.urdf.get_actuated_joint_names()
            self.robot_index = model.index
            self.urdf.update_cfg(np.zeros(len(self.urdf_names)))
        self.last_update, self.status = 0.0, "WAITING"
        self.stereo = cv2.imencode(".jpg", np.zeros((360, 1280, 3), np.uint8))[1].tobytes()
        self._start_dashboard()
        print(f"Dashboard: http://localhost:{WEB_PORT}")

    @staticmethod
    def _camera(server, position, look_at, up, fov):
        server.scene.set_up_direction(up)
        server.gui.configure_theme(show_logo=False, show_share_button=False)
        server.initial_camera.position = position
        server.initial_camera.look_at = look_at
        server.initial_camera.up = up
        server.initial_camera.fov = np.radians(fov)

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
        points = result["keypoint_relative"]
        if points is not None and result["handedness"] in ("Left", "Right"):
            self.human_frame.wxyz = _human_view_wxyz(result["handedness"], points)
        visible = points is not None
        self.human_cloud.visible = self.human_bones.visible = visible
        if visible:
            points = np.asarray(points, np.float32)
            self.human_cloud.points = points
            self.human_bones.points = points[np.asarray(SKELETON_EDGES)]
        if robot is not None and self.urdf is not None:
            self.urdf.update_cfg(np.asarray([robot[self.robot_index[name]] for name in self.urdf_names]))
            tips, directions = self.model.fingertip_pads(robot)
            tips, directions = np.asarray(tips, np.float32), np.asarray(directions, np.float32)
            self.robot_pad_arrows.points = np.stack(
                (tips, tips + ROBOT_PAD_ARROW_LENGTH * directions), axis=1
            )
            self.robot_pad_arrows.visible = True

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
            dimensions.append(
                MultiArrayDimension(
                    label=label if not dimensions else name,
                    size=size,
                    stride=stride,
                )
            )
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
    parser.add_argument("--udp", metavar="HOST:PORT")
    args = parser.parse_args()
    if args.udp is not None and args.mode != "retarget":
        parser.error("--udp is only available with --mode retarget")

    camera_error = _camera_configuration_error()
    if camera_error is not None:
        parser.error(camera_error)

    retargeter = Retargeter() if args.mode == "retarget" else None
    udp = camera = processor = viewer = ros = None
    try:
        try:
            udp = UdpRetargetSender(args.udp) if args.udp is not None else None
        except (TypeError, ValueError, OSError) as exc:
            parser.error(f"invalid --udp endpoint: {exc}")
        if udp is not None:
            print(f"UDP retarget: publishing to {args.udp}")

        if CAMERA_TYPE == "d435":
            camera = RealSenseCamera(CAMERA_INDEX)
            processor = StereoProcessor(camera.params)
        else:
            # Validate calibration and MediaPipe before starting Camera's
            # background VideoCapture thread, so partial startup cannot abort
            # the interpreter during exception unwinding.
            processor = StereoProcessor()
            camera = Camera(CAMERA_INDEX)

        viewer = Viewer(None if retargeter is None else retargeter.model)
        ros = RosOutput(args.mode) if args.ros else None
        last_timestamp = None
        # Keep sequence numbers increasing when only the capture process is
        # restarted while the simulator/receiver remains alive.
        sequence = time.monotonic_ns()
        while True:
            ok, left, right, timestamp = camera.read()
            if not ok or timestamp == last_timestamp:
                time.sleep(0.002)
                continue
            last_timestamp = timestamp
            sequence += 1
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
            if udp is not None:
                udp_valid = (
                    robot is not None
                    and valid
                    and result["handedness"] == "Left"
                    and result["phase"].startswith("GESTURE")
                )
                udp.send(
                    sequence,
                    time.monotonic(),
                    udp_valid,
                    result["phase"],
                    result["handedness"],
                    ROBOT_JOINT_NAMES,
                    robot.copy() if udp_valid else None,
                    _udp_quality(result["quality"]),
                )
            viewer.update(result, robot)
    except KeyboardInterrupt:
        pass
    finally:
        if camera is not None:
            camera.close()
        if processor is not None:
            processor.close()
        if viewer is not None:
            viewer.close()
        if ros is not None:
            ros.close()
        if udp is not None:
            udp.close()


if __name__ == "__main__":
    main()
