import json
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize

import config as C

EPS = 1e-9


def _values(text, default):
    return np.fromstring(text if text is not None else default, sep=" ")


def _rotation(axis, angle):
    axis = np.asarray(axis, float)
    axis /= max(np.linalg.norm(axis), EPS)
    cross = np.array(((0, -axis[2], axis[1]), (axis[2], 0, -axis[0]), (-axis[1], axis[0], 0)))
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def _origin(node):
    xyz = _values(node.get("xyz"), "0 0 0")
    roll, pitch, yaw = _values(node.get("rpy"), "0 0 0")
    transform = np.eye(4)
    transform[:3, :3] = _rotation((0, 0, 1), yaw) @ _rotation((0, 1, 0), pitch) @ _rotation((1, 0, 0), roll)
    transform[:3, 3] = xyz
    return transform


class LowPass:
    def __init__(self, alpha):
        self.alpha, self.value = alpha, None

    def __call__(self, value):
        self.value = value.copy() if self.value is None else self.alpha * value + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None


class RobotModel:
    def __init__(self, urdf_path=C.URDF_PATH, contract_path=C.URDF_CONTRACT_PATH):
        self.joints, self.children, parent_joint = {}, defaultdict(list), {}
        urdf_limits = {}
        for node in ET.parse(urdf_path).getroot().findall("joint"):
            origin, axis, limit = node.find("origin"), node.find("axis"), node.find("limit")
            joint = {
                "type": node.get("type"),
                "parent": node.find("parent").get("link"),
                "child": node.find("child").get("link"),
                "origin": _origin(origin) if origin is not None else np.eye(4),
                "axis": _values(axis.get("xyz"), "1 0 0") if axis is not None else np.array((1.0, 0, 0)),
            }
            name = node.get("name")
            self.joints[name] = joint
            self.children[joint["parent"]].append(name)
            parent_joint[joint["child"]] = name
            if joint["type"] == "revolute":
                urdf_limits[name] = (float(limit.get("lower")), float(limit.get("upper")))
        self.names = tuple(name for name in C.ROBOT_JOINT_NAMES if name in urdf_limits)
        if self.names != C.ROBOT_JOINT_NAMES:
            raise ValueError(f"Unexpected HKU Hand joints: {self.names}")
        self.index = {name: index for index, name in enumerate(self.names)}

        contract = json.loads(contract_path.read_text())
        contract_limits = np.radians([joint["limit_deg"] for joint in contract["joints"]])
        mapping = np.asarray(C.ROBOT_TO_CONTRACT)
        limits = np.asarray([urdf_limits[name] for name in self.names])
        self.lower = np.maximum(limits[:, 0], contract_limits[mapping, 0])
        self.upper = np.minimum(limits[:, 1], contract_limits[mapping, 1])
        if np.any(self.lower > self.upper):
            raise ValueError("URDF and ROS contract joint limits do not overlap")
        self.topic = contract["topic"]
        self.layout = (
            f"mmhand:J00-J20:{contract['input_space']}:{contract['mapping_id']}:"
            f"v{contract['mapping_version']}:{contract['urdf_config_crc']}"
        )
        self.links = tuple(dict.fromkeys(C.VECTOR_ORIGIN_LINKS + C.VECTOR_TASK_LINKS))
        self.origin = np.asarray([self.links.index(name) for name in C.VECTOR_ORIGIN_LINKS])
        self.task = np.asarray([self.links.index(name) for name in C.VECTOR_TASK_LINKS])
        self.tip_links = tuple(f"{finger}-tip_Link" for finger in (5, 1, 2, 3, 4))
        self.ancestors = {}
        for link in self.links:
            chain, cursor = [], link
            while cursor in parent_joint:
                name = parent_joint[cursor]
                if name in self.index:
                    chain.append(self.index[name])
                cursor = self.joints[name]["parent"]
            self.ancestors[link] = chain

    def fk(self, q):
        transforms, origins, axes = {"base_link": np.eye(4)}, np.zeros((21, 3)), np.zeros((21, 3))
        stack = ["base_link"]
        while stack:
            parent = stack.pop()
            for name in self.children[parent]:
                joint = self.joints[name]
                frame = transforms[parent] @ joint["origin"]
                child = frame
                if name in self.index:
                    index = self.index[name]
                    origins[index], axes[index] = frame[:3, 3], frame[:3, :3] @ joint["axis"]
                    motion = np.eye(4)
                    motion[:3, :3] = _rotation(joint["axis"], q[index])
                    child = frame @ motion
                transforms[joint["child"]] = child
                stack.append(joint["child"])
        return transforms, origins, axes

    def vectors(self, q, jacobian=False):
        transforms, origins, axes = self.fk(q)
        positions = np.asarray([transforms[name][:3, 3] for name in self.links])
        vectors = positions[self.task] - positions[self.origin]
        if not jacobian:
            return vectors
        link_jacobian = np.zeros((len(self.links), 3, 21))
        for row, name in enumerate(self.links):
            for joint in self.ancestors[name]:
                link_jacobian[row, :, joint] = np.cross(axes[joint], positions[row] - origins[joint])
        return vectors, link_jacobian[self.task] - link_jacobian[self.origin]

    def points(self, q):
        transforms, _, _ = self.fk(q)
        return {name: transforms[name][:3, 3].copy() for name in self.tip_links}

    def contract_degrees(self, q):
        output = np.empty(21)
        output[np.asarray(C.ROBOT_TO_CONTRACT)] = q
        return np.degrees(output)


class Retargeter:
    def __init__(self, model=None, scaling=C.VECTOR_SCALING, alpha=C.VECTOR_LOW_PASS_ALPHA):
        self.model = RobotModel() if model is None else model
        scaling = np.asarray(scaling, float)
        if scaling.ndim == 0:
            scaling = np.full(16, scaling)
        if scaling.shape != (16,) or not np.isfinite(scaling).all() or np.any(scaling <= 0):
            raise ValueError("scaling must be a positive scalar or 16-vector")
        self.scaling, self.filter = scaling[:, None], LowPass(alpha)
        self.raw_q = self.model.lower.copy()

    @staticmethod
    def _hand_frame(points):
        palm = points[[0, 5, 9]]
        x = palm[0] - palm[2]
        _, _, vh = np.linalg.svd(palm - palm.mean(0))
        normal = vh[2]
        x -= np.dot(x, normal) * normal
        x /= max(np.linalg.norm(x), EPS)
        z = np.cross(x, normal)
        if np.dot(z, palm[1] - palm[2]) < 0:
            normal, z = -normal, -z
        return np.column_stack((x, normal, z))

    def targets(self, points):
        points = (np.asarray(points, float) - points[0]) * 0.001
        mano = points @ self._hand_frame(points) @ np.asarray(C.OPERATOR2MANO_LEFT)
        hku = mano[:, (2, 1, 0)] * (1, 1, -1)
        return hku[np.asarray(C.VECTOR_HUMAN_TASKS)] - hku[np.asarray(C.VECTOR_HUMAN_ORIGINS)]

    def objective(self, q, target, last=None):
        vectors, jacobian = self.model.vectors(q, True)
        error = vectors - target * self.scaling
        distance = np.linalg.norm(error, axis=1)
        quadratic = distance < C.VECTOR_HUBER_DELTA
        loss = np.where(quadratic, 0.5 * distance**2 / C.VECTOR_HUBER_DELTA, distance - 0.5 * C.VECTOR_HUBER_DELTA)
        direction = error / np.maximum(np.where(quadratic, C.VECTOR_HUBER_DELTA, distance)[:, None], EPS)
        gradient = np.einsum("vi,vij->j", direction, jacobian) / len(error)
        value = loss.mean()
        if last is not None:
            delta = q - last
            value += C.VECTOR_NORM_DELTA * np.dot(delta, delta)
            gradient += 2 * C.VECTOR_NORM_DELTA * delta
        return float(value), gradient, float(np.sqrt(np.mean(distance**2)))

    def solve(self, points, timestamp=None):
        points = np.asarray(points, float)
        if points.shape != (21, 3) or not np.isfinite(points).all():
            return self._failure(float("inf"))
        try:
            target, last = self.targets(points), self.raw_q.copy()
            result = minimize(
                lambda q: self.objective(q, target, last)[:2],
                last,
                jac=True,
                bounds=tuple(zip(self.model.lower, self.model.upper)),
                method="SLSQP",
                options={"ftol": 1e-6, "maxiter": C.VECTOR_MAX_EVAL},
            )
            q = np.asarray(result.x)
            _, _, rms = self.objective(q, target)
        except (np.linalg.LinAlgError, ValueError):
            return self._failure(float("inf"))
        if not result.success or not np.isfinite(q).all() or not np.isfinite(rms):
            return self._failure(rms)
        self.raw_q = np.clip(q, self.model.lower, self.model.upper)
        filtered = np.clip(self.filter(self.raw_q), self.model.lower, self.model.upper)
        pose_rms = float(np.sqrt(np.mean((self.model.vectors(filtered) - target) ** 2)))
        return {
            "success": True,
            "q": filtered,
            "q_raw": self.raw_q.copy(),
            "q_deg": self.model.contract_degrees(filtered),
            "rms": rms,
            "pose_rms": pose_rms,
            "points": self.model.points(filtered),
        }

    def _failure(self, rms):
        return {
            "success": False, "q": None, "q_raw": self.raw_q.copy(), "q_deg": None,
            "rms": rms, "pose_rms": float("inf"), "points": self.model.points(self.raw_q),
        }

    def pause(self):
        self.filter.reset()
