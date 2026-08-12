from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from config import URDF_PATH, WEB_FPS, WEB_PORT, VIEW_PORTS
from input.frame import SKELETON_EDGES
from retarget import compute_cmc_frame

ROBOT_CAMERA_DISTANCE = 0.42
PALM_FRAME_AXIS_LENGTH = 0.04
PALM_FRAME_AXIS_RADIUS = 0.0015
PALM_FRAME_ORIGIN_RADIUS = 0.004
PAD_DIRECTION_LENGTH = 0.025
ARROW_SHAFT_RADIUS = 0.0008
ARROW_HEAD_RADIUS = 0.0025
ARROW_HEAD_LENGTH = 0.006
TIP_INDICES = np.array((4, 8, 12, 16, 20))
LOSS_LABELS = (
    ("thumb_tip", "thumb tip"),
    ("thumb_mcp_ip", "thumb MCP-IP"),
    ("thumb_ip_tip", "thumb IP-TIP"),
    ("total", "total"),
)


def _loss_text(losses=None):
    lines = ["Weighted retarget loss"]
    if losses:
        total = losses["total"]
        lines += [
            f"{label:<18}{losses[name]:.3e}  "
            f"{(100 * losses[name] / total if total else 0):5.1f}%"
            for name, label in LOSS_LABELS
        ]
    else:
        lines.append("waiting")
    return "\n".join(lines)


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


def _frame_wxyz(frame):
    quaternion = Rotation.from_matrix(np.asarray(frame, float)).as_quat()
    return quaternion[[3, 0, 1, 2]]


def _arrow_points(starts, directions):
    return np.stack(
        (starts, starts + PAD_DIRECTION_LENGTH * directions), axis=1
    ).astype(np.float32)


def _robot_camera_pose(model):
    tips = model.fingertips(model.seed)
    look_at = np.vstack((model.palm_position, tips)).mean(0)
    palm_normal = model.palm_frame[:, 2]
    transforms = model.fk(model.seed)
    base = (
        transforms["finger_1_proximal_phalanx_1"][:3, 3]
        - transforms["finger_4_proximal_phalanx_1"][:3, 3]
    )
    normal = palm_normal - base * np.dot(palm_normal, base) / np.dot(base, base)
    normal /= np.linalg.norm(normal)
    return look_at - ROBOT_CAMERA_DISTANCE * normal, look_at, (
        0.9993984744,
        -0.0346479515,
        -0.0014862239,
    )


class Viewer:
    def __init__(self, model):
        import viser
        from viser.extras import ViserUrdf

        self.servers = [viser.ViserServer(port=port) for port in VIEW_PORTS]
        self.model = model
        normalized, robot_server = self.servers
        self._camera(
            normalized,
            (-0.26, 0, 0.06),
            (0, 0, 0.06),
            (0, 0.2588190451, 0.9659258263),
            42,
        )
        self._camera(robot_server, *_robot_camera_pose(model), 42)
        self.human_frame = normalized.scene.add_frame("/hand", show_axes=False)
        self.human_retarget_frame = normalized.scene.add_frame(
            "/hand/retarget_frame",
            show_axes=True,
            axes_length=PALM_FRAME_AXIS_LENGTH,
            axes_radius=PALM_FRAME_AXIS_RADIUS,
            origin_radius=PALM_FRAME_ORIGIN_RADIUS,
            visible=False,
        )
        empty_hand = np.zeros((21, 3), np.float32)
        self.human_cloud = normalized.scene.add_point_cloud(
            "/hand/points",
            empty_hand,
            (60, 170, 255),
            point_size=0.004,
            point_shape="circle",
            visible=False,
        )
        self.human_bones = normalized.scene.add_line_segments(
            "/hand/bones",
            empty_hand[np.asarray(SKELETON_EDGES)],
            (60, 170, 255),
            line_width=4,
            visible=False,
        )
        self.pad_directions = normalized.scene.add_arrows(
            "/hand/pad_directions",
            np.zeros((5, 2, 3), np.float32),
            (255, 150, 70),
            shaft_radius=ARROW_SHAFT_RADIUS,
            head_radius=ARROW_HEAD_RADIUS,
            head_length=ARROW_HEAD_LENGTH,
            visible=False,
        )
        robot_server.scene.add_frame("/robot", show_axes=False)
        self.urdf = ViserUrdf(robot_server, URDF_PATH, root_node_name="/robot")
        self.robot_palm_frame = robot_server.scene.add_frame(
            "/robot/palm_frame",
            show_axes=True,
            position=model.palm_position,
            wxyz=_frame_wxyz(model.palm_frame),
            axes_length=PALM_FRAME_AXIS_LENGTH,
            axes_radius=PALM_FRAME_AXIS_RADIUS,
            origin_radius=PALM_FRAME_ORIGIN_RADIUS,
        )
        self.urdf_names = self.urdf.get_actuated_joint_names()
        self.robot_index = model.index
        self.urdf.update_cfg(
            np.asarray([model.seed[self.robot_index[name]] for name in self.urdf_names])
        )
        starts, directions = model.fingertip_pads(model.seed)
        self.robot_pad_directions = robot_server.scene.add_arrows(
            "/robot/pad_directions",
            _arrow_points(starts, directions),
            (255, 150, 70),
            shaft_radius=ARROW_SHAFT_RADIUS,
            head_radius=ARROW_HEAD_RADIUS,
            head_length=ARROW_HEAD_LENGTH,
        )
        self.last_update, self.status = 0.0, "WAITING"
        self.loss_text = _loss_text()
        self.preview = cv2.imencode(".jpg", np.zeros((360, 1280, 3), np.uint8))[1].tobytes()
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
<title>mmhand_teleop</title><style>
*{{box-sizing:border-box}} body{{margin:0;height:100vh;background:#101318;color:#eef;
font:15px sans-serif;display:grid;grid-template:40vh 60vh/repeat(2,1fr);gap:6px}}
.panel{{position:relative;overflow:hidden;background:#181d25;border:1px solid #343b48}}
.input{{grid-column:1/3}} h2{{position:absolute;z-index:2;margin:0;padding:8px 12px;
font-size:15px;background:#101318cc;border-radius:0 0 6px 0}}
img,iframe{{width:100%;height:100%;display:block;border:0;object-fit:contain}}
#status{{color:#9bd;margin-left:12px;font-weight:normal}}
#losses{{position:absolute;z-index:2;right:0;top:0;margin:0;padding:9px 12px;
background:#101318dd;color:#bde2ff;font:13px/1.35 monospace;pointer-events:none}}</style></head><body>
<section class="panel input"><h2>Input <span id="status">WAITING</span></h2>
<pre id="losses">Weighted retarget loss\nwaiting</pre><img id="preview"></section>
<section class="panel"><h2>Normalized hand</h2><iframe id="normalized"></iframe></section>
<section class="panel"><h2>Retargeted MMHand</h2><iframe id="robot"></iframe></section>
<script>
const host=location.hostname, statusText=document.getElementById("status"),
previewImage=document.getElementById("preview"),lossText=document.getElementById("losses");
for(const [id,port] of [["normalized",{normalized_port}],["robot",{robot_port}]])
  document.getElementById(id).src=`http://${{host}}:${{port}}`;
setInterval(()=>{{previewImage.src="/preview.jpg?t="+Date.now();
fetch("/status").then(r=>r.text()).then(x=>statusText.textContent=x);
fetch("/losses").then(r=>r.text()).then(x=>lossText.textContent=x)}},{round(1000 / WEB_FPS)});
</script></body></html>""".encode()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path == "/preview.jpg":
                    body, content_type = owner.preview, "image/jpeg"
                elif path == "/status":
                    body, content_type = owner.status.encode(), "text/plain; charset=utf-8"
                elif path == "/losses":
                    body, content_type = owner.loss_text.encode(), "text/plain; charset=utf-8"
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

    def update(self, frame, robot=None, losses=None):
        now = time.monotonic()
        if now - self.last_update < 1 / WEB_FPS:
            return
        self.last_update = now
        self.status = frame.status
        if losses is not None:
            self.loss_text = _loss_text(losses)
        elif frame.points is None or frame.handedness != "Left":
            self.loss_text = _loss_text()
        preview = frame.preview
        if preview is None:
            preview = np.zeros((360, 1280, 3), np.uint8)
        ok, encoded = cv2.imencode(".jpg", preview, (cv2.IMWRITE_JPEG_QUALITY, 82))
        if ok:
            self.preview = encoded.tobytes()
        points = frame.points
        if points is not None and frame.handedness in ("Left", "Right"):
            self.human_frame.wxyz = _human_view_wxyz(frame.handedness, points)
        visible = points is not None
        self.human_cloud.visible = self.human_bones.visible = visible
        self.pad_directions.visible = False
        if not visible:
            self.human_retarget_frame.visible = False
        else:
            points = np.asarray(points, float)
            self.human_cloud.points = points.astype(np.float32)
            self.human_bones.points = points[np.asarray(SKELETON_EDGES)].astype(np.float32)
            directions = frame.finger_pad_directions
            if directions is not None:
                directions = np.asarray(directions, float)
                if directions.shape == (5, 3) and np.isfinite(directions).all():
                    starts = points[TIP_INDICES]
                    self.pad_directions.points = _arrow_points(starts, directions)
                    self.pad_directions.visible = True
            try:
                origin, palm_frame = compute_cmc_frame(points)
            except ValueError:
                self.human_retarget_frame.visible = False
            else:
                self.human_retarget_frame.position = origin
                self.human_retarget_frame.wxyz = _frame_wxyz(palm_frame)
                self.human_retarget_frame.visible = True
        if robot is not None:
            self.urdf.update_cfg(
                np.asarray([robot[self.robot_index[name]] for name in self.urdf_names])
            )
            starts, directions = self.model.fingertip_pads(robot)
            self.robot_pad_directions.points = _arrow_points(starts, directions)

    def close(self):
        self.http.shutdown()
        self.http.server_close()
        for server in self.servers:
            server.stop()
