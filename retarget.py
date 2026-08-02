import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares

import config as C
from hand_core import extract_angles

EPS = 1e-9
THUMB = np.arange(16, 21)
AA = np.array((0, 4, 8, 12))
FINGER_TIPS = tuple(f"{finger}-tip_Link" for finger in (1, 2, 3, 4))
THUMB_TIP = "5-tip_Link"
THUMB_PAD_LINK = "mmhand_thumb_1_finger_7_fingertip_1"
PAD_SPECS = ((THUMB_TIP, THUMB_PAD_LINK, C.THUMB_PAD_AXIS),) + tuple(
    (f"{finger}-tip_Link", f"finger_{finger}_fingertip_1", axis)
    for finger, axis in enumerate(C.FINGER_PAD_AXES, 1)
)


def _values(text, default):
    return np.fromstring(text if text is not None else default, sep=" ")


def _unit(vector):
    return vector / max(np.linalg.norm(vector), EPS)


def _rotation(axis, angle):
    axis = _unit(np.asarray(axis, float))
    cross = np.array(((0, -axis[2], axis[1]), (axis[2], 0, -axis[0]), (-axis[1], axis[0], 0)))
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def _origin(node):
    xyz = _values(node.get("xyz"), "0 0 0")
    roll, pitch, yaw = _values(node.get("rpy"), "0 0 0")
    transform = np.eye(4)
    transform[:3, :3] = _rotation((0, 0, 1), yaw) @ _rotation((0, 1, 0), pitch) @ _rotation((1, 0, 0), roll)
    transform[:3, 3] = xyz
    return transform


class RobotModel:
    def __init__(self, urdf_path=C.URDF_PATH):
        self.joints, self.children = {}, defaultdict(list)
        limits = {}
        for node in ET.parse(urdf_path).getroot().findall("joint"):
            origin, axis, limit = node.find("origin"), node.find("axis"), node.find("limit")
            joint = {
                "parent": node.find("parent").get("link"),
                "child": node.find("child").get("link"),
                "origin": _origin(origin) if origin is not None else np.eye(4),
                "axis": _values(axis.get("xyz"), "1 0 0") if axis is not None else np.array((1.0, 0, 0)),
            }
            name = node.get("name")
            self.joints[name] = joint
            self.children[joint["parent"]].append(name)
            if node.get("type") == "revolute":
                limits[name] = (float(limit.get("lower")), float(limit.get("upper")))
        self.names = tuple(name for name in C.ROBOT_JOINT_NAMES if name in limits)
        if self.names != C.ROBOT_JOINT_NAMES:
            raise ValueError(f"Unexpected MMHand joints: {self.names}")
        self.index = {name: index for index, name in enumerate(self.names)}
        self.lower, self.upper = np.asarray([limits[name] for name in self.names]).T
        q = np.zeros(21)
        q[AA] = np.radians(C.MCP_AA_NEUTRAL_DEG)
        transforms = self.fk(q)
        mcp = np.asarray([transforms[f"finger_{i}_proximal_phalanx_1"][:3, 3] for i in range(1, 5)])
        pip = np.asarray([transforms[f"finger_{i}_distal_phalanx_1"][:3, 3] for i in range(1, 5)])
        side = _unit(mcp[0] - mcp[3])
        forward = _unit((pip - mcp).mean(0) - side * np.dot((pip - mcp).mean(0), side))
        normal = _unit(np.cross(side, forward))
        self.palm_position = transforms["palm_1"][:3, 3]
        self.palm_frame = np.column_stack((normal, np.cross(forward, normal), forward))

    def fk(self, q):
        transforms, stack = {"base_link": np.eye(4)}, ["base_link"]
        while stack:
            parent = stack.pop()
            for name in self.children[parent]:
                joint = self.joints[name]
                child = transforms[parent] @ joint["origin"]
                if name in self.index:
                    motion = np.eye(4)
                    motion[:3, :3] = _rotation(joint["axis"], q[self.index[name]])
                    child = child @ motion
                transforms[joint["child"]] = child
                stack.append(joint["child"])
        return transforms

    def thumb(self, q):
        transforms = self.fk(q)
        tip = transforms[THUMB_TIP][:3, 3]
        fingers = np.asarray([transforms[name][:3, 3] for name in FINGER_TIPS])
        tip = (tip - self.palm_position) @ self.palm_frame
        fingers = (fingers - self.palm_position) @ self.palm_frame
        normal = (transforms[THUMB_PAD_LINK][:3, :3] @ np.asarray(C.THUMB_PAD_AXIS)) @ self.palm_frame
        return tip, _unit(normal), fingers

    def fingertip_pads(self, q):
        transforms = self.fk(q)
        tips = np.asarray([transforms[tip][:3, 3] for tip, _, _ in PAD_SPECS])
        directions = np.asarray(
            [
                _unit(transforms[link][:3, :3] @ np.asarray(axis))
                for _, link, axis in PAD_SPECS
            ]
        )
        return tips, directions


class Retargeter:
    def __init__(self, model=None):
        self.model = RobotModel() if model is None else model
        self.palm_origin = np.asarray(C.MMHAND_PALM_ORIGIN)
        self.tip_scale = C.THUMB_TIP_SCALE
        self.tip_weight = np.sqrt(C.THUMB_TIP_WEIGHT)
        self.neutral = np.zeros(21)
        self.neutral[AA] = np.radians(C.MCP_AA_NEUTRAL_DEG)
        self.neutral[16] = np.radians(C.THUMB_MCP_AA_NEUTRAL_DEG)
        self.model.lower[17:20] = np.maximum(self.model.lower[17:20], 0)
        self.q = np.clip(self.neutral, self.model.lower, self.model.upper)

    @staticmethod
    def _finger_q(points, handedness):
        points = np.asarray(points, float)
        forward = _unit(points[9] - points[0])
        side = _unit(points[5] - points[17])
        normal = _unit(np.cross(side, forward))
        side = _unit(np.cross(forward, normal))
        aa = [
            np.arctan2(np.dot(_unit(points[mcp + 1] - points[mcp]), side),
                       np.dot(_unit(points[mcp + 1] - points[mcp]), forward))
            for mcp in (5, 9, 13, 17)
        ]
        flex = np.maximum(np.radians(extract_angles(points, handedness)), 0)
        q = np.zeros(21)
        q[AA] = np.radians(C.MCP_AA_NEUTRAL_DEG) + np.asarray(aa)[::-1]
        for source, target in zip((11, 8, 5, 2), (0, 4, 8, 12)):
            q[target + 1:target + 4] = flex[source:source + 3]
        return q

    def _task(self, points):
        vectors = points[[8, 12, 16, 20]] - points[4]
        return points[4] * self.tip_scale, vectors, np.argsort(np.linalg.norm(vectors, axis=1))[:2]

    def _residual(self, thumb, q, target, vectors, nearest):
        q[THUMB] = thumb
        tip, normal, fingers = self.model.thumb(q)
        tip_error = self.tip_weight * (tip - self.palm_origin - target) / C.STANDARD_PALM_SIZE
        vector_error = (fingers - tip - vectors) / C.STANDARD_PALM_SIZE
        toward = _unit(fingers[nearest].mean(0) - tip)
        pad_error = np.sqrt(C.THUMB_PAD_WEIGHT) * (normal - toward)
        return np.r_[tip_error, vector_error.ravel(), pad_error]

    def solve(self, points, handedness="Left"):
        points = np.asarray(points, float)
        if points.shape != (21, 3) or not np.isfinite(points).all():
            return None
        try:
            q = self._finger_q(points, handedness)
            q[:16] = np.clip(q[:16], self.model.lower[:16], self.model.upper[:16])
            target, vectors, nearest = self._task(points)
            result = least_squares(
                self._residual, self.q[THUMB], bounds=(self.model.lower[THUMB], self.model.upper[THUMB]),
                args=(q, target, vectors, nearest),
                ftol=C.THUMB_FTOL, xtol=C.THUMB_FTOL, gtol=C.THUMB_FTOL,
                max_nfev=C.THUMB_MAX_EVAL,
            )
            q[THUMB] = result.x
        except (np.linalg.LinAlgError, ValueError):
            return None
        if not np.isfinite(q).all():
            return None
        self.q = q
        return q.copy()

    def pause(self):
        self.q = self.neutral.copy()
