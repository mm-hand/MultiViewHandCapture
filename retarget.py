import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares

import config as C
from hand_core import extract_angles, relative_points

EPS = 1e-9
THUMB = np.arange(16, 21)
AA = np.array((0, 4, 8, 12))
TIP_LINKS = tuple(f"{finger}-tip_Link" for finger in (5, 1, 2, 3, 4))
THUMB_LINKS = (
    "mmhand_thumb_1_thumb_abduction_adduction_link_1",
    "mmhand_thumb_1_finger_7_fingertip_1",
    "5-tip_Link",
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
        origin, dip, tip = (transforms[name][:3, 3] for name in THUMB_LINKS)
        normal = transforms[THUMB_LINKS[1]][:3, :3] @ np.asarray(C.THUMB_PAD_AXIS)
        fingers = np.asarray([transforms[name][:3, 3] for name in TIP_LINKS[1:]])
        return np.asarray((dip - origin, tip - origin)), tip, _unit(normal), fingers

    def points(self, q):
        transforms = self.fk(q)
        return {name: transforms[name][:3, 3].copy() for name in TIP_LINKS}


class Retargeter:
    def __init__(self, model=None):
        self.model = RobotModel() if model is None else model
        self.rotations = np.asarray([
            _rotation(vector, np.linalg.norm(vector))
            for vector in np.asarray(C.THUMB_VECTOR_ROTATION_VECS)
        ])
        self.scaling = np.asarray(C.THUMB_VECTOR_SCALING)[:, None]
        self.weights = np.sqrt(np.asarray(C.THUMB_VECTOR_WEIGHTS))[:, None]
        self.neutral = np.zeros(21)
        self.neutral[AA] = np.radians(C.MCP_AA_NEUTRAL_DEG)
        self.neutral[16] = np.radians(28)
        self.model.lower[17:20] = np.maximum(self.model.lower[17:20], 0)
        self.raw_q = np.clip(self.neutral, self.model.lower, self.model.upper)

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

    @staticmethod
    def _human_points(points, handedness):
        human = relative_points(points)
        if human is None:
            raise ValueError("degenerate hand")
        return human

    def _task(self, points, handedness):
        human = self._human_points(points, handedness)
        vectors = np.asarray((human[3] - human[1], human[4] - human[1]))
        targets = np.einsum("vij,vj->vi", self.rotations, vectors) * self.scaling
        nearest = np.argsort(np.linalg.norm(human[[8, 12, 16, 20]] - human[4], axis=1))[:2]
        return targets, nearest

    def _residual(self, thumb, q, targets, nearest):
        q[THUMB] = thumb
        vectors, tip, normal, fingers = self.model.thumb(q)
        lengths = np.maximum(np.linalg.norm(targets, axis=1)[:, None], EPS)
        vector_error = self.weights * (vectors - targets) / lengths
        toward = _unit(fingers[nearest].mean(0) - tip)
        pad_error = np.sqrt(C.THUMB_PAD_WEIGHT) * (normal - toward)
        return np.r_[vector_error.ravel(), pad_error]

    def solve(self, points, timestamp=None, handedness="Left"):
        points = np.asarray(points, float)
        if points.shape != (21, 3) or not np.isfinite(points).all():
            return self._failure(float("inf"))
        try:
            q = self._finger_q(points, handedness)
            q[:16] = np.clip(q[:16], self.model.lower[:16], self.model.upper[:16])
            targets, nearest = self._task(points, handedness)
            last = self.raw_q[THUMB].copy()
            result = least_squares(
                self._residual, last, bounds=(self.model.lower[THUMB], self.model.upper[THUMB]),
                args=(q, targets, nearest),
                ftol=C.THUMB_FTOL, xtol=C.THUMB_FTOL, gtol=C.THUMB_FTOL,
                max_nfev=C.THUMB_MAX_EVAL,
            )
            q[THUMB] = result.x
            vectors = self.model.thumb(q)[0]
            rms = float(np.sqrt(np.mean((vectors - targets) ** 2)))
        except (np.linalg.LinAlgError, ValueError):
            return self._failure(float("inf"))
        if not np.isfinite(q).all():
            return self._failure(rms)
        self.raw_q = q
        return {
            "success": True, "q": q.copy(), "q_raw": q.copy(),
            "q_deg": np.degrees(q), "rms": rms, "points": self.model.points(q),
        }

    def _failure(self, rms):
        return {
            "success": False, "q": None, "q_raw": self.raw_q.copy(), "q_deg": None,
            "rms": rms, "points": self.model.points(self.raw_q),
        }

    def pause(self):
        self.raw_q = self.neutral.copy()
