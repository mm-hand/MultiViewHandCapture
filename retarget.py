import xml.etree.ElementTree as ET
from collections import defaultdict
import threading
import time
import warnings

import numpy as np
from scipy.optimize import Bounds, minimize

import config as C
from one_euro import OneEuro

EPS = 1e-9
HUMAN_THUMB = np.array((2, 3, 4))
HUMAN_FINGERS = np.array(
    ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
)
ROBOT_TIPS = ("5-tip_Link", "1-tip_Link", "2-tip_Link", "3-tip_Link", "4-tip_Link")
ROBOT_THUMB = (
    "mmhand_thumb_1_finger_7_distal_phalanx_1",
    "mmhand_thumb_1_finger_7_fingertip_1",
    "5-tip_Link",
)
ROBOT_FINGERS = tuple(
    (
        f"finger_{finger}_proximal_phalanx_1",
        f"finger_{finger}_distal_phalanx_1",
        f"finger_{finger}_fingertip_1",
        f"{finger}-tip_Link",
    )
    for finger in range(1, 5)
)
ROBOT_FINGER_JOINTS = tuple(
    (
        f"{name}_MCP_AA",
        f"{name}_MCP_FE",
        f"finger_{finger}_distal_phalanx_1_PIP_Joint",
        f"finger_{finger}_fingertip_1_DIP_Joint",
    )
    for name, finger in zip(("Index", "Middle", "Ring", "Little"), range(1, 5))
)
THUMB_DIRECTION_NAMES = ("thumb_mcp_ip", "thumb_ip_tip")


class _EvaluationLimit(Exception):
    pass


def _values(text, default):
    return np.fromstring(text if text is not None else default, sep=" ")


def _unit(vector):
    return vector / max(np.linalg.norm(vector), EPS)


def _checked_unit(vector):
    norm = np.linalg.norm(vector)
    if norm <= EPS:
        raise ValueError("Degenerate human palm frame")
    return vector / norm


def compute_cmc_frame(points_world):
    """Return ``origin_world, R_world_from_cmc`` for the existing CMC frame."""
    points = np.asarray(points_world, float)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("Human landmarks must be finite with shape (21, 3)")
    x_axis = _checked_unit(points[9] - points[0])
    side = points[17] - points[5]
    y_axis = _checked_unit(side - x_axis * np.dot(side, x_axis))
    z_axis = _checked_unit(np.cross(x_axis, y_axis))
    y_axis = np.cross(z_axis, x_axis)
    return points[1].copy(), np.column_stack((x_axis, y_axis, z_axis))


def human_palm_frame(points):
    """Backward-compatible name for :func:`compute_cmc_frame`."""
    return compute_cmc_frame(points)


def human_retarget_points(points):
    """Express tracking landmarks in the independent CMC retarget frame."""
    points = np.asarray(points, float)
    origin_world, R_world_from_cmc = compute_cmc_frame(points)
    return (points - origin_world) @ R_world_from_cmc


def _rotation(axis, angle):
    axis = np.asarray(axis, float)
    cross = np.array(((0, -axis[2], axis[1]), (axis[2], 0, -axis[0]), (-axis[1], axis[0], 0)))
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def _origin(node):
    xyz = _values(node.get("xyz"), "0 0 0")
    roll, pitch, yaw = _values(node.get("rpy"), "0 0 0")
    transform = np.eye(4)
    transform[:3, :3] = _rotation((0, 0, 1), yaw) @ _rotation((0, 1, 0), pitch) @ _rotation((1, 0, 0), roll)
    transform[:3, 3] = xyz
    return transform


def _directions(points, jacobians=None):
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths <= EPS):
        raise ValueError("Zero-length retarget segment")
    directions = vectors / lengths[:, None]
    if jacobians is None:
        return directions
    vector_jacobians = np.diff(jacobians, axis=0)
    projectors = np.eye(3) - directions[:, :, None] * directions[:, None, :]
    direction_jacobians = (
        np.einsum("kij,kjn->kin", projectors, vector_jacobians)
        / lengths[:, None, None]
    )
    return directions, direction_jacobians


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
                "axis": _unit(_values(axis.get("xyz"), "1 0 0")) if axis is not None else np.array((1.0, 0, 0)),
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
        for name, joint in self.joints.items():
            joint["index"] = self.index.get(name)
        self.order, stack = [], ["base_link"]
        while stack:
            parent = stack.pop()
            for name in self.children[parent]:
                self.order.append(name)
                stack.append(self.joints[name]["child"])
        ancestors = {"base_link": ()}
        for name in self.order:
            joint = self.joints[name]
            indices = ancestors[joint["parent"]]
            if joint["index"] is not None:
                indices += (joint["index"],)
            ancestors[joint["child"]] = indices
        self.influence = np.zeros((len(ROBOT_THUMB), len(self.names), 1))
        for row, name in enumerate(ROBOT_THUMB):
            self.influence[row, list(ancestors[name])] = 1
        self.lower, self.upper = np.asarray([limits[name] for name in self.names]).T
        self.thumb = np.arange(16, 21)
        self.finger_joints = np.asarray(
            [[self.index[name] for name in names] for names in ROBOT_FINGER_JOINTS]
        )
        if np.any(self.upper - self.lower <= EPS):
            raise ValueError("MMHand revolute joints must have nonzero ranges")
        self.seed = np.clip(np.zeros(len(self.names)), self.lower, self.upper)
        transforms = self.fk(self.seed)
        self.palm_frame = transforms["base_link"][:3, :3].copy()
        self.finger_neutral, self.finger_axis = [], []
        for chain, indices in zip(ROBOT_FINGERS, self.finger_joints):
            direction = (
                transforms[chain[1]][:3, 3] - transforms[chain[0]][:3, 3]
            ) @ self.palm_frame
            joint = self.joints[self.names[indices[0]]]
            frame = transforms[joint["parent"]] @ joint["origin"]
            axis = (frame[:3, :3] @ joint["axis"]) @ self.palm_frame
            if abs(axis[2]) <= EPS:
                raise ValueError("MMHand MCP A-A axis must follow the palm z axis")
            self.finger_axis.append(axis[2])
            self.finger_neutral.append(-np.arctan2(direction[1], direction[0]) / axis[2])
        self.finger_neutral = np.asarray(self.finger_neutral)
        self.finger_axis = np.asarray(self.finger_axis)
        cmc = self.joints["Thumb_CMC"]
        cmc_frame = transforms[cmc["parent"]] @ cmc["origin"]
        cmc_origin = cmc_frame[:3, 3]
        self.cmc_axis = _unit(cmc_frame[:3, :3] @ cmc["axis"])
        mcp_aa_origin = transforms[self.joints["Thumb_MCP_AA"]["child"]][:3, 3]
        self.palm_position = cmc_origin + self.cmc_axis * np.dot(
            mcp_aa_origin - cmc_origin, self.cmc_axis
        )

    def _forward(self, q, jacobian=False):
        transforms = {"base_link": np.eye(4)}
        origins = np.zeros((len(self.names), 3))
        axes = np.zeros_like(origins)
        for name in self.order:
            joint = self.joints[name]
            child = transforms[joint["parent"]] @ joint["origin"]
            index = joint["index"]
            if index is not None:
                origins[index] = child[:3, 3]
                axes[index] = child[:3, :3] @ joint["axis"]
                motion = np.eye(4)
                motion[:3, :3] = _rotation(joint["axis"], q[index])
                child = child @ motion
            transforms[joint["child"]] = child
        return (transforms, origins, axes) if jacobian else transforms

    def fk(self, q):
        return self._forward(np.asarray(q, float))

    def fingertips(self, q):
        transforms = self.fk(q)
        return np.asarray([transforms[name][:3, 3] for name in ROBOT_TIPS])

    def fingertip_pads(self, q):
        """Return fingertip positions and outward pad normals, Thumb to Little."""
        transforms = self.fk(q)
        positions = np.asarray([transforms[name][:3, 3] for name in ROBOT_TIPS])
        directions = np.asarray([-transforms[name][:3, 2] for name in ROBOT_TIPS])
        return positions, directions / np.linalg.norm(directions, axis=1, keepdims=True)

    def finger_angles(self, points, previous=None):
        q = self.seed.copy() if previous is None else np.asarray(previous, float).copy()
        for row, (chain, indices) in enumerate(zip(HUMAN_FINGERS, self.finger_joints)):
            directions = _directions(points[chain])
            proximal, planar = directions[0], np.linalg.norm(directions[0, :2])
            aa = (
                self.finger_neutral[row]
                if previous is None or planar <= EPS
                else q[indices[0]]
            )
            if planar > EPS:
                aa = self.finger_neutral[row] + np.arctan2(proximal[1], proximal[0]) / self.finger_axis[row]
            angles = (
                aa,
                np.arctan2(-proximal[2], planar),
                np.arccos(np.clip(proximal @ directions[1], -1, 1)),
                np.arccos(np.clip(directions[1] @ directions[2], -1, 1)),
            )
            q[indices] = np.clip(angles, self.lower[indices], self.upper[indices])
        return q

    def features(self, q):
        transforms, origins, axes = self._forward(np.asarray(q, float), True)
        origins, axes = origins[self.thumb], axes[self.thumb]
        influence = self.influence[:, self.thumb]
        world = np.asarray([transforms[name][:3, 3] for name in ROBOT_THUMB])
        points = (world - self.palm_position) @ self.palm_frame
        delta = world[:, None] - origins
        world_jacobians = np.empty_like(delta)
        world_jacobians[:, :, 0] = axes[:, 1] * delta[:, :, 2] - axes[:, 2] * delta[:, :, 1]
        world_jacobians[:, :, 1] = axes[:, 2] * delta[:, :, 0] - axes[:, 0] * delta[:, :, 2]
        world_jacobians[:, :, 2] = axes[:, 0] * delta[:, :, 1] - axes[:, 1] * delta[:, :, 0]
        world_jacobians *= influence
        point_jacobians = np.einsum(
            "ij,knj->kin", self.palm_frame.T, world_jacobians
        )

        directions, direction_jacobians = _directions(points, point_jacobians)
        return (points[-1:], directions), (point_jacobians[-1:], direction_jacobians)


class Retargeter:
    def __init__(self, model=None):
        self.model = RobotModel() if model is None else model
        self.bounds = Bounds(self.model.lower[self.model.thumb], self.model.upper[self.model.thumb])
        self.q = self.model.seed.copy()
        self.output_filter = OneEuro(*C.RETARGET_ANGLE_FILTER)
        self.has_previous = False
        self.loss_terms = None
        self.options = {"ftol": C.RETARGET_FTOL, "disp": False}

    def _targets(self, points, previous=None):
        try:
            local = human_retarget_points(points)
            thumb_shape = _directions(local[HUMAN_THUMB])
            fingers = self.model.finger_angles(local, previous)
        except ValueError:
            return None
        return local[4:5], thumb_shape, fingers

    @staticmethod
    def _term(value, jacobian, target, scale, weight, normalizer=1.0):
        error = (value - scale * target) / normalizer
        loss = weight * np.mean(np.sum(error * error, axis=1))
        gradient = (
            2 * weight / (len(error) * normalizer)
            * np.einsum("ki,kin->n", error, jacobian)
        )
        return loss, gradient

    def _losses(self, q, targets):
        values, jacobians = self.model.features(q)
        tip_loss, tip_gradient = self._term(
            values[0], jacobians[0], targets[0],
            C.RETARGET_THUMB_TIP_SCALE, C.RETARGET_THUMB_TIP_WEIGHT,
            C.STANDARD_PALM_SIZE,
        )
        weights = (
            C.RETARGET_THUMB_MCP_IP_WEIGHT,
            C.RETARGET_THUMB_IP_TIP_WEIGHT,
        )
        thumb_terms = tuple(
            self._term(
                values[1][i:i + 1], jacobians[1][i:i + 1], targets[1][i:i + 1],
                1.0, weight / 3,
            )
            for i, weight in enumerate(weights)
        )
        total = tip_loss + sum(loss for loss, _ in thumb_terms)
        gradient = tip_gradient + sum(
            (term[1] for term in thumb_terms), np.zeros(len(self.model.thumb))
        )
        return (
            total,
            gradient,
            {
                "thumb_tip": tip_loss,
                **dict(zip(THUMB_DIRECTION_NAMES, (term[0] for term in thumb_terms))),
                "total": total,
            },
        )

    def solve(self, points, timestamp=None):
        points = np.asarray(points, float)
        if points.shape != (21, 3) or not np.isfinite(points).all():
            return None
        previous = self.q.copy() if self.has_previous else None
        targets = self._targets(points, previous)
        if targets is None:
            return None
        fixed, thumb = targets[2], self.model.thumb
        cached_q, cached_value, evaluations, best = None, None, 0, None

        def evaluate(x):
            nonlocal cached_q, cached_value, evaluations, best
            if cached_q is None or not np.array_equal(x, cached_q):
                if evaluations >= C.RETARGET_MAX_EVALUATIONS:
                    raise _EvaluationLimit
                cached_q = np.asarray(x).copy()
                q = fixed.copy()
                q[thumb] = x
                cached_value = self._losses(q, targets)
                evaluations += 1
                if np.isfinite(cached_value[0]) and np.isfinite(cached_value[1]).all():
                    candidate = cached_value[0], q
                    if best is None or candidate[0] < best[0]:
                        best = candidate
            return cached_value[:2]

        try:
            evaluate(fixed[thumb])
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Values in x were outside bounds during a minimize step, clipping to bounds",
                    category=RuntimeWarning,
                    module=r"scipy\.optimize\._slsqp_py",
                )
                minimize(
                    evaluate, fixed[thumb], method="SLSQP", jac=True,
                    bounds=self.bounds, options=self.options,
                )
        except _EvaluationLimit:
            pass
        if best is None:
            return None
        _, candidate = best
        self.q = candidate.copy()
        self.has_previous = True
        timestamp = time.monotonic() if timestamp is None else timestamp
        output = np.radians(self.output_filter(np.degrees(candidate), timestamp))
        output = np.clip(output, self.model.lower, self.model.upper)
        losses = self._losses(output, targets)[2]
        self.loss_terms = {name: float(value) for name, value in losses.items()}
        return output

    def pause(self):
        self.q = self.model.seed.copy()
        self.output_filter.reset()
        self.has_previous = False
        self.loss_terms = None


class RetargetWorker:
    def __init__(self, retargeter=None):
        self.retargeter = Retargeter() if retargeter is None else retargeter
        self.model = self.retargeter.model
        self.condition = threading.Condition()
        self.pending = self.result = self.error = None
        self.generation, self.running = 0, True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, points, timestamp=None):
        with self.condition:
            self.pending = self.generation, np.asarray(points, float).copy(), timestamp
            self.condition.notify()

    def poll(self):
        with self.condition:
            if self.error is not None:
                raise RuntimeError("Retarget worker failed") from self.error
            result, self.result = self.result, None
            return result

    def pause(self):
        with self.condition:
            self.generation += 1
            self.pending = self.result = None
            self.condition.notify()

    def _run(self):
        generation = 0
        while True:
            with self.condition:
                self.condition.wait_for(lambda: not self.running or self.pending is not None)
                if not self.running:
                    return
                current, points, timestamp = self.pending
                self.pending = None
            try:
                if current != generation:
                    self.retargeter.pause()
                    generation = current
                result = self.retargeter.solve(points, timestamp)
            except Exception as error:
                with self.condition:
                    self.error, self.running = error, False
                    self.condition.notify_all()
                return
            with self.condition:
                if self.running and current == self.generation and result is not None:
                    losses = dict(getattr(self.retargeter, "loss_terms", None) or {})
                    self.result = result, losses

    def close(self):
        with self.condition:
            self.running, self.pending = False, None
            self.condition.notify()
        self.thread.join()
