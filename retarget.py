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
        self.tip = np.asarray([self.links.index(name) for name in self.tip_links])
        self.pad_link = "mmhand_thumb_1_finger_7_fingertip_1"
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
        state = self.features(q)
        return state[0:2] if jacobian else state[0]

    def features(self, q):
        transforms, origins, axes = self.fk(q)
        positions = np.asarray([transforms[name][:3, 3] for name in self.links])
        vectors = positions[self.task] - positions[self.origin]
        link_jacobian = np.zeros((len(self.links), 3, 21))
        for row, name in enumerate(self.links):
            for joint in self.ancestors[name]:
                link_jacobian[row, :, joint] = np.cross(axes[joint], positions[row] - origins[joint])
        vector_jacobian = link_jacobian[self.task] - link_jacobian[self.origin]
        normal = transforms[self.pad_link][:3, :3] @ np.asarray(C.THUMB_PAD_AXIS)
        normal /= max(np.linalg.norm(normal), EPS)
        normal_jacobian = np.zeros((3, 21))
        for joint in self.ancestors[self.pad_link]:
            normal_jacobian[:, joint] = np.cross(axes[joint], normal)
        return vectors, vector_jacobian, positions[self.tip], link_jacobian[self.tip], normal, normal_jacobian

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
        opposition_weights=C.OPPOSITION_WEIGHTS,
        pad_weight=C.THUMB_PAD_WEIGHT,
    ):
        self.model = RobotModel() if model is None else model
        count = len(C.VECTOR_MAP)
        scaling, weights, thumb_weights, opposition_weights = map(
            lambda value: np.asarray(value, float),
            (scaling, weights, thumb_weights, opposition_weights),
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
        if opposition_weights.shape != (4,) or not np.isfinite(opposition_weights).all() or np.any(opposition_weights <= 0):
            raise ValueError("opposition_weights must be a positive 4-vector")
        if not np.isfinite(pad_weight) or pad_weight < 0:
            raise ValueError("pad_weight must be non-negative")
        rotvecs = np.asarray(C.VECTOR_ROTATION_VECS, float)
        if rotvecs.shape != (count, 3):
            raise ValueError(f"VECTOR_ROTATION_VECS must have shape ({count}, 3)")
        self.rotations = np.asarray([
            np.eye(3) if np.linalg.norm(vector) < EPS else _rotation(vector, np.linalg.norm(vector))
            for vector in rotvecs
        ])
        self.scaling, self.weights = scaling[:, None], weights
        self.thumb_weights = thumb_weights
        self.opposition_scaling = np.asarray(C.OPPOSITION_SCALING)
        self.opposition_weights = opposition_weights
        radii = np.asarray(C.TIP_RADII)
        self.radii = radii
        self.contact = radii[0] + radii[1:] + C.TIP_CLEARANCE
        self.pad_weight = float(pad_weight)
        self.neutral = np.zeros(21)
        aa = np.radians(np.asarray(C.MCP_AA_NEUTRAL_DEG) + C.MCP_AA_SAFE_OFFSET_DEG)
        self.neutral[[0, 4, 8, 12, 16]] = aa
        self.lower, self.upper = self.model.lower.copy(), self.model.upper.copy()
        for index, offset, value in zip((0, 4, 8, 12, 16), C.MCP_AA_SAFE_OFFSET_DEG, aa):
            if offset < 0:
                self.upper[index] = value
            elif offset > 0:
                self.lower[index] = value
        self.lower[[18, 19]] = np.maximum(self.lower[[18, 19]], 0)
        self.neutral = np.clip(self.neutral, self.lower, self.upper)
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

    def human_points(self, points, handedness="Left"):
        points = (np.asarray(points, float) - points[0]) * 0.001
        mano = points @ self._hand_frame(points) @ np.asarray(C.OPERATOR2MANO_LEFT)
        robot = mano[:, (2, 1, 0)] * (1, 1, -1)
        if handedness == "Right":
            robot[:, 2] *= -1
        return robot

    def human_vectors(self, points, handedness="Left"):
        robot = self.human_points(points, handedness)
        origins = np.asarray([item[0] for item in C.VECTOR_MAP])
        tasks = np.asarray([item[1] for item in C.VECTOR_MAP])
        return robot[tasks] - robot[origins]

    def targets(self, points, handedness="Left"):
        return np.einsum("vij,vj->vi", self.rotations, self.human_vectors(points, handedness))

    def task(self, points, handedness="Left"):
        human_points = self.human_points(points, handedness)
        origins = np.asarray([item[0] for item in C.VECTOR_MAP])
        targets = np.asarray([item[1] for item in C.VECTOR_MAP])
        human = human_points[targets] - human_points[origins]
        target = np.einsum("vij,vj->vi", self.rotations, human)
        thumb_target = np.maximum(
            self._bend(human) - np.radians(C.THUMB_NEUTRAL_BEND_DEG), 0
        )
        human_distance = np.linalg.norm(
            human_points[[8, 12, 16, 20]] - human_points[4], axis=1
        )
        opposition_target = np.maximum(
            self.contact, self.opposition_scaling * human_distance
        )
        pair = np.argsort(human_distance)[:2]
        ratio = human_distance[pair[0]] / max(np.linalg.norm(human_points[12]), EPS)
        near, far = C.THUMB_PAD_GATE
        gate = np.clip((far - ratio) / (far - near), 0, 1)
        gate = gate * gate * (3 - 2 * gate)
        return target, thumb_target, opposition_target, pair, gate

    @staticmethod
    def _bend(vectors):
        output = []
        for first, second in ((17, 18), (18, 19)):
            u, v = vectors[first], vectors[second]
            nu, nv = max(np.linalg.norm(u), EPS), max(np.linalg.norm(v), EPS)
            cosine = np.clip(np.dot(u, v) / (nu * nv), -1, 1)
            output.append(np.arccos(cosine))
        return np.asarray(output)

    @staticmethod
    def _huber(error, delta):
        absolute = np.abs(error)
        quadratic = absolute < delta
        loss = np.where(quadratic, 0.5 * error**2 / delta, absolute - 0.5 * delta)
        direction = np.where(quadratic, error / delta, np.sign(error))
        return loss, direction

    def objective(self, q, task, last=None):
        target, thumb_target, opposition_target, pair, gate = task
        vectors, jacobian, tips, tip_jacobian, normal, normal_jacobian = self.model.features(q)
        error = vectors - target * self.scaling
        distance = np.linalg.norm(error, axis=1)
        quadratic = distance < C.VECTOR_HUBER_DELTA
        loss = np.where(quadratic, 0.5 * distance**2 / C.VECTOR_HUBER_DELTA, distance - 0.5 * C.VECTOR_HUBER_DELTA)
        direction = error / np.maximum(np.where(quadratic, C.VECTOR_HUBER_DELTA, distance)[:, None], EPS)
        total = self.weights.sum()
        gradient = np.einsum("v,vi,vij->j", self.weights, direction, jacobian) / total
        value = np.dot(self.weights, loss) / total
        angle_error = q[[18, 19]] - thumb_target
        angle_loss, angle_direction = self._huber(angle_error, np.radians(C.THUMB_ANGLE_HUBER_DEG))
        value += np.dot(self.thumb_weights, angle_loss)
        gradient[[18, 19]] += self.thumb_weights * angle_direction
        difference = tips[1:] - tips[0]
        robot_distance = np.linalg.norm(difference, axis=1)
        distance_jacobian = np.einsum(
            "vi,vij->vj",
            difference / np.maximum(robot_distance[:, None], EPS),
            tip_jacobian[1:] - tip_jacobian[0],
        )
        opposition_error = robot_distance - opposition_target
        opposition_loss, opposition_direction = self._huber(opposition_error, C.OPPOSITION_HUBER_DELTA)
        value += np.dot(self.opposition_weights, opposition_loss)
        gradient += np.einsum(
            "v,v,vj->j", self.opposition_weights, opposition_direction, distance_jacobian
        )
        finger_difference = tips[2:] - tips[1:-1]
        finger_distance = np.linalg.norm(finger_difference, axis=1)
        finger_jacobian = np.einsum(
            "vi,vij->vj",
            finger_difference / np.maximum(finger_distance[:, None], EPS),
            tip_jacobian[2:] - tip_jacobian[1:-1],
        )
        finger_error = np.minimum(
            finger_distance
            - self.radii[1:-1]
            - self.radii[2:]
            - C.TIP_CLEARANCE,
            0,
        )
        finger_loss, finger_direction = self._huber(
            finger_error, C.FINGER_SAFETY_HUBER
        )
        value += C.FINGER_SAFETY_WEIGHT * finger_loss.sum()
        gradient += C.FINGER_SAFETY_WEIGHT * np.einsum(
            "v,vj->j", finger_direction, finger_jacobian
        )
        midpoint = 0.5 * (tips[pair[0] + 1] + tips[pair[1] + 1])
        midpoint_jacobian = 0.5 * (
            tip_jacobian[pair[0] + 1] + tip_jacobian[pair[1] + 1]
        )
        toward = midpoint - tips[0]
        toward_norm = max(np.linalg.norm(toward), EPS)
        toward /= toward_norm
        toward_jacobian = (
            (np.eye(3) - np.outer(toward, toward))
            @ (midpoint_jacobian - tip_jacobian[0])
            / toward_norm
        )
        cosine = np.clip(np.dot(normal, toward), -1 + EPS, 1 - EPS)
        pad_angle = np.arccos(cosine)
        pad_jacobian = -(
            toward @ normal_jacobian + normal @ toward_jacobian
        ) / max(np.sqrt(1 - cosine * cosine), 1e-6)
        pad_loss, pad_direction = self._huber(pad_angle, np.radians(C.THUMB_PAD_HUBER_DEG))
        value += gate * self.pad_weight * pad_loss
        gradient += gate * self.pad_weight * pad_direction * pad_jacobian
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
            task, last = self.task(points, handedness), self.raw_q.copy()
            result = minimize(
                lambda q: self.objective(q, task, last)[:2],
                last,
                jac=True,
                bounds=tuple(zip(self.lower, self.upper)),
                method="SLSQP",
                options={"ftol": C.VECTOR_FTOL, "maxiter": C.VECTOR_MAX_EVAL},
            )
            q = np.asarray(result.x)
            _, _, rms = self.objective(q, task)
        except (np.linalg.LinAlgError, ValueError):
            return self._failure(float("inf"))
        if not result.success or not np.isfinite(q).all() or not np.isfinite(rms):
            return self._failure(rms)
        candidate = np.clip(q, self.lower, self.upper)
        if not self._safe(candidate):
            return self._failure(rms)
        self.raw_q = candidate
        return {
            "success": True,
            "q": self.raw_q.copy(),
            "q_raw": self.raw_q.copy(),
            "q_deg": np.degrees(self.raw_q),
            "rms": rms,
            "points": self.model.points(self.raw_q),
        }

    def _failure(self, rms):
        return {
            "success": False, "q": None, "q_raw": self.raw_q.copy(), "q_deg": None,
            "rms": rms, "points": self.model.points(self.raw_q),
        }

    def _safe(self, q):
        tips = self.model.features(q)[2]
        thumb = np.linalg.norm(tips[1:] - tips[0], axis=1)
        fingers = np.linalg.norm(np.diff(tips[1:], axis=0), axis=1)
        return bool(
            np.all(thumb >= self.radii[0] + self.radii[1:] - 1e-6)
            and np.all(fingers >= self.radii[1:-1] + self.radii[2:] - 1e-6)
        )

    def pause(self):
        self.raw_q = self.neutral.copy()
