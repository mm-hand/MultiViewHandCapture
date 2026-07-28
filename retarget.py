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
        fingers = np.asarray([transforms[name][:3, 3] for name in FINGER_TIPS])
        return np.asarray((dip - origin, tip - origin)), tip, _unit(normal), fingers

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
        vectors = np.asarray((points[3] - points[1], points[4] - points[1]))
        targets = np.einsum("vij,vj->vi", self.rotations, vectors) * self.scaling
        nearest = np.argsort(np.linalg.norm(points[[8, 12, 16, 20]] - points[4], axis=1))[:2]
        return targets, nearest

    def _residual(self, thumb, q, targets, nearest):
        q[THUMB] = thumb
        vectors, tip, normal, fingers = self.model.thumb(q)
        lengths = np.maximum(np.linalg.norm(targets, axis=1)[:, None], EPS)
        vector_error = self.weights * (vectors - targets) / lengths
        toward = _unit(fingers[nearest].mean(0) - tip)
        pad_error = np.sqrt(C.THUMB_PAD_WEIGHT) * (normal - toward)
        return np.r_[vector_error.ravel(), pad_error]

    def solve(self, points, handedness="Left"):
        points = np.asarray(points, float)
        if points.shape != (21, 3) or not np.isfinite(points).all():
            return None
        try:
            q = self._finger_q(points, handedness)
            q[:16] = np.clip(q[:16], self.model.lower[:16], self.model.upper[:16])
            targets, nearest = self._task(points)
            result = least_squares(
                self._residual, self.q[THUMB], bounds=(self.model.lower[THUMB], self.model.upper[THUMB]),
                args=(q, targets, nearest),
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
