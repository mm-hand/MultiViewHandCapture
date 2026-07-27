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
    def __init__(self, urdf_path=C.URDF_PATH):
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
            raise ValueError(f"Unexpected MMHand joints: {self.names}")
        self.index = {name: index for index, name in enumerate(self.names)}

        limits = np.asarray([urdf_limits[name] for name in self.names])
        self.lower, self.upper = limits.T
        robot_origins = tuple(item[2] for item in C.VECTOR_MAP)
        robot_tasks = tuple(item[3] for item in C.VECTOR_MAP)
        self.links = tuple(dict.fromkeys(robot_origins + robot_tasks))
        self.origin = np.asarray([self.links.index(name) for name in robot_origins])
        self.task = np.asarray([self.links.index(name) for name in robot_tasks])
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


class Retargeter:
    def __init__(
        self,
        model=None,
        scaling=C.VECTOR_SCALING,
        weights=C.VECTOR_WEIGHTS,
        thumb_weights=C.THUMB_ANGLE_WEIGHTS,
        alpha=C.VECTOR_LOW_PASS_ALPHA,
    ):
        self.model = RobotModel() if model is None else model
        count = len(C.VECTOR_MAP)
        scaling, weights, thumb_weights = map(
            lambda value: np.asarray(value, float), (scaling, weights, thumb_weights)
        )
        if scaling.ndim == 0:
            scaling = np.full(count, scaling)
        if weights.ndim == 0:
            weights = np.full(count, weights)
        if scaling.shape != (count,) or not np.isfinite(scaling).all() or np.any(scaling <= 0):
            raise ValueError(f"scaling must be a positive scalar or {count}-vector")
        if weights.shape != (count,) or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError(f"weights must be a positive scalar or {count}-vector")
        if thumb_weights.shape != (2,) or not np.isfinite(thumb_weights).all() or np.any(thumb_weights < 0):
            raise ValueError("thumb_weights must be a non-negative 2-vector")
        rotvecs = np.asarray(C.VECTOR_ROTATION_VECS, float)
        if rotvecs.shape != (count, 3):
            raise ValueError(f"VECTOR_ROTATION_VECS must have shape ({count}, 3)")
        self.rotations = np.asarray([
            np.eye(3) if np.linalg.norm(vector) < EPS else _rotation(vector, np.linalg.norm(vector))
            for vector in rotvecs
        ])
        self.scaling, self.weights = scaling[:, None], weights
        self.thumb_weights = thumb_weights
        self.thumb_offset = np.radians(C.THUMB_BEND_OFFSET_DEG)
        self.filter = LowPass(alpha)
        self.neutral = np.zeros(21)
        self.neutral[[0, 4, 8, 12, 16]] = np.radians(C.MCP_AA_NEUTRAL_DEG)
        self.neutral = np.clip(self.neutral, self.model.lower, self.model.upper)
        self.raw_q = self.neutral.copy()

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

    def human_vectors(self, points, handedness="Left"):
        points = (np.asarray(points, float) - points[0]) * 0.001
        mano = points @ self._hand_frame(points) @ np.asarray(C.OPERATOR2MANO_LEFT)
        robot = mano[:, (2, 1, 0)] * (1, 1, -1)
        if handedness == "Right":
            robot[:, 2] *= -1
        origins = np.asarray([item[0] for item in C.VECTOR_MAP])
        tasks = np.asarray([item[1] for item in C.VECTOR_MAP])
        return robot[tasks] - robot[origins]

    def targets(self, points, handedness="Left"):
        return np.einsum("vij,vj->vi", self.rotations, self.human_vectors(points, handedness))

    @staticmethod
    def _bend(vectors, jacobian=None):
        output, gradients = [], []
        for first, second in ((17, 18), (18, 19)):
            u, v = vectors[first], vectors[second]
            nu, nv = max(np.linalg.norm(u), EPS), max(np.linalg.norm(v), EPS)
            uh, vh = u / nu, v / nv
            cosine = np.clip(np.dot(uh, vh), -1 + EPS, 1 - EPS)
            output.append(np.arccos(cosine))
            if jacobian is not None:
                du = (np.eye(3) - np.outer(uh, uh)) @ jacobian[first] / nu
                dv = (np.eye(3) - np.outer(vh, vh)) @ jacobian[second] / nv
                derivative = vh @ du + uh @ dv
                gradients.append(-derivative / max(np.sqrt(1 - cosine * cosine), 1e-6))
        return np.asarray(output) if jacobian is None else (np.asarray(output), np.asarray(gradients))

    def objective(self, q, target, last=None, human_bends=None):
        vectors, jacobian = self.model.vectors(q, True)
        error = vectors - target * self.scaling
        distance = np.linalg.norm(error, axis=1)
        quadratic = distance < C.VECTOR_HUBER_DELTA
        loss = np.where(quadratic, 0.5 * distance**2 / C.VECTOR_HUBER_DELTA, distance - 0.5 * C.VECTOR_HUBER_DELTA)
        direction = error / np.maximum(np.where(quadratic, C.VECTOR_HUBER_DELTA, distance)[:, None], EPS)
        total = self.weights.sum()
        gradient = np.einsum("v,vi,vij->j", self.weights, direction, jacobian) / total
        value = np.dot(self.weights, loss) / total
        if human_bends is not None:
            bends, bend_jacobian = self._bend(vectors, jacobian)
            angle_error = bends - human_bends - self.thumb_offset
            delta = np.radians(C.THUMB_ANGLE_HUBER_DEG)
            absolute = np.abs(angle_error)
            quadratic = absolute < delta
            angle_loss = np.where(
                quadratic, 0.5 * angle_error**2 / delta, absolute - 0.5 * delta
            )
            angle_direction = np.where(quadratic, angle_error / delta, np.sign(angle_error))
            value += np.dot(self.thumb_weights, angle_loss)
            gradient += np.einsum(
                "a,a,aj->j", self.thumb_weights, angle_direction, bend_jacobian
            )
        if last is not None:
            delta = q - last
            value += C.VECTOR_NORM_DELTA * np.dot(delta, delta)
            gradient += 2 * C.VECTOR_NORM_DELTA * delta
        return float(value), gradient, float(np.sqrt(np.mean(distance**2)))

    def solve(self, points, timestamp=None, handedness="Left"):
        points = np.asarray(points, float)
        if points.shape != (21, 3) or not np.isfinite(points).all():
            return self._failure(float("inf"))
        try:
            human = self.human_vectors(points, handedness)
            target = np.einsum("vij,vj->vi", self.rotations, human)
            human_bends, last = self._bend(human), self.raw_q.copy()
            result = minimize(
                lambda q: self.objective(q, target, last, human_bends)[:2],
                last,
                jac=True,
                bounds=tuple(zip(self.model.lower, self.model.upper)),
                method="SLSQP",
                options={"ftol": 1e-6, "maxiter": C.VECTOR_MAX_EVAL},
            )
            q = np.asarray(result.x)
            _, _, rms = self.objective(q, target, human_bends=human_bends)
        except (np.linalg.LinAlgError, ValueError):
            return self._failure(float("inf"))
        if not result.success or not np.isfinite(q).all() or not np.isfinite(rms):
            return self._failure(rms)
        self.raw_q = np.clip(q, self.model.lower, self.model.upper)
        filtered = np.clip(self.filter(self.raw_q), self.model.lower, self.model.upper)
        return {
            "success": True,
            "q": filtered,
            "q_raw": self.raw_q.copy(),
            "q_deg": np.degrees(filtered),
            "rms": rms,
            "points": self.model.points(filtered),
        }

    def _failure(self, rms):
        return {
            "success": False, "q": None, "q_raw": self.raw_q.copy(), "q_deg": None,
            "rms": rms, "points": self.model.points(self.raw_q),
        }

    def pause(self):
        self.filter.reset()
        self.raw_q = self.neutral.copy()
