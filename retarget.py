import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
from scipy.optimize import Bounds, minimize

import config as C

EPS = 1e-9
HUMAN_TIPS = np.array((4, 8, 12, 16, 20))
HUMAN_THUMB = np.array((1, 2, 3, 4))
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
ROBOT_LINKS = tuple(dict.fromkeys(ROBOT_TIPS + ROBOT_THUMB + sum(ROBOT_FINGERS, ())))
ROBOT_LINK_INDEX = {name: index for index, name in enumerate(ROBOT_LINKS)}


def _values(text, default):
    return np.fromstring(text if text is not None else default, sep=" ")


def _unit(vector):
    return vector / max(np.linalg.norm(vector), EPS)


def _checked_unit(vector):
    norm = np.linalg.norm(vector)
    if norm <= EPS:
        raise ValueError("Degenerate human palm frame")
    return vector / norm


def human_palm_frame(points):
    """Return the CMC origin and base_link-style human palm axes."""
    points = np.asarray(points, float)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("Human landmarks must be finite with shape (21, 3)")
    x_axis = _checked_unit(points[9] - points[0])
    side = points[17] - points[5]
    y_axis = _checked_unit(side - x_axis * np.dot(side, x_axis))
    z_axis = _checked_unit(np.cross(x_axis, y_axis))
    y_axis = np.cross(z_axis, x_axis)
    return points[1].copy(), np.column_stack((x_axis, y_axis, z_axis))


def human_retarget_points(points):
    """Express tracking landmarks in the independent CMC retarget frame."""
    points = np.asarray(points, float)
    origin, frame = human_palm_frame(points)
    return (points - origin) @ frame


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
        self.influence = np.zeros((len(ROBOT_LINKS), len(self.names), 1))
        for row, name in enumerate(ROBOT_LINKS):
            self.influence[row, list(ancestors[name])] = 1
        self.lower, self.upper = np.asarray([limits[name] for name in self.names]).T
        self.seed = np.clip(np.zeros(len(self.names)), self.lower, self.upper)
        transforms = self.fk(self.seed)
        self.palm_frame = transforms["base_link"][:3, :3].copy()
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

    def features(self, q, jacobian=False):
        transforms, origins, axes = self._forward(np.asarray(q, float), True)
        world = np.asarray([transforms[name][:3, 3] for name in ROBOT_LINKS])
        points = (world - self.palm_position) @ self.palm_frame
        delta = world[:, None] - origins
        world_jacobians = np.empty_like(delta)
        world_jacobians[:, :, 0] = axes[:, 1] * delta[:, :, 2] - axes[:, 2] * delta[:, :, 1]
        world_jacobians[:, :, 1] = axes[:, 2] * delta[:, :, 0] - axes[:, 0] * delta[:, :, 2]
        world_jacobians[:, :, 2] = axes[:, 0] * delta[:, :, 1] - axes[:, 1] * delta[:, :, 0]
        world_jacobians *= self.influence
        point_jacobians = np.einsum(
            "ij,knj->kin", self.palm_frame.T, world_jacobians
        )

        def select(names):
            indices = [ROBOT_LINK_INDEX[name] for name in names]
            return points[indices], point_jacobians[indices]

        tips, tip_jacobians = select(ROBOT_TIPS)
        thumb, thumb_jacobians = select(ROBOT_THUMB)
        thumb = np.vstack((np.zeros(3), thumb))
        thumb_jacobians = np.concatenate((np.zeros((1, 3, len(self.names))), thumb_jacobians))
        thumb_shape, thumb_shape_jacobian = _directions(thumb, thumb_jacobians)
        finger_shapes, finger_shape_jacobians = [], []
        for chain in ROBOT_FINGERS:
            chain_points, chain_jacobians = select(chain)
            directions, derivatives = _directions(chain_points, chain_jacobians)
            finger_shapes.append(directions)
            finger_shape_jacobians.append(derivatives)
        values = (
            tips,
            tips[1:] - tips[0],
            thumb_shape,
            np.concatenate(finger_shapes),
        )
        if not jacobian:
            return values
        derivatives = (
            tip_jacobians,
            tip_jacobians[1:] - tip_jacobians[0],
            thumb_shape_jacobian,
            np.concatenate(finger_shape_jacobians),
        )
        return values, derivatives


class Retargeter:
    def __init__(self, model=None):
        self.model = RobotModel() if model is None else model
        self.bounds = Bounds(self.model.lower, self.model.upper)
        self.q = self.model.seed.copy()
        self.losses = None
        self.options = {
            "maxiter": C.RETARGET_MAX_ITERATIONS,
            "ftol": C.RETARGET_FTOL,
            "disp": False,
        }

    @staticmethod
    def _targets(points):
        try:
            local = human_retarget_points(points)
            tips = local[HUMAN_TIPS]
            thumb_shape = _directions(local[HUMAN_THUMB])
            finger_shape = np.concatenate([_directions(local[chain]) for chain in HUMAN_FINGERS])
        except ValueError:
            return None
        return tips, local[HUMAN_TIPS[1:]] - local[4], thumb_shape, finger_shape

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
        values, jacobians = self.model.features(q, True)
        palm_loss, palm_gradient = self._term(
            values[0], jacobians[0], targets[0],
            C.RETARGET_PALM_TIPS_SCALE, C.RETARGET_PALM_TIPS_WEIGHT,
            C.STANDARD_PALM_SIZE,
        )
        fingertip_loss, fingertip_gradient = self._term(
            values[1], jacobians[1], targets[1],
            C.RETARGET_THUMB_FINGERTIPS_SCALE,
            C.RETARGET_THUMB_FINGERTIPS_WEIGHT,
            C.STANDARD_PALM_SIZE,
        )
        stage1_loss = palm_loss + fingertip_loss
        stage1_gradient = palm_gradient + fingertip_gradient
        thumb_loss, thumb_gradient = self._term(
            values[2], jacobians[2], targets[2], 1.0,
            C.RETARGET_THUMB_SHAPE_WEIGHT,
        )
        finger_loss, finger_gradient = self._term(
            values[3], jacobians[3], targets[3], 1.0,
            C.RETARGET_FINGER_SHAPE_WEIGHT,
        )
        return (
            stage1_loss,
            stage1_gradient,
            stage1_loss + thumb_loss + finger_loss,
            stage1_gradient + thumb_gradient + finger_gradient,
        )

    def solve(self, points):
        points = np.asarray(points, float)
        if points.shape != (21, 3) or not np.isfinite(points).all():
            return None
        targets = self._targets(points)
        if targets is None:
            return None
        cached_q, cached_value = None, None

        def evaluate(q):
            nonlocal cached_q, cached_value
            if cached_q is None or not np.array_equal(q, cached_q):
                cached_q, cached_value = np.asarray(q).copy(), self._losses(q, targets)
            return cached_value

        first = minimize(
            lambda q: evaluate(q)[:2], self.q, method="SLSQP", jac=True,
            bounds=self.bounds, options=self.options,
        )
        if not first.success or not np.isfinite(first.x).all():
            return None
        q1 = np.clip(first.x, self.model.lower, self.model.upper)
        stage1_loss, _, stage2_at_q1, _ = evaluate(q1)
        limit = stage1_loss * (1 + C.RETARGET_STAGE1_MAX_RELATIVE_INCREASE)
        solver_limit = limit - max(EPS, abs(limit) * 1e-6)
        constraint = {
            "type": "ineq",
            "fun": lambda q: solver_limit - evaluate(q)[0],
            "jac": lambda q: -evaluate(q)[1],
        }
        second = minimize(
            lambda q: evaluate(q)[2:], q1, method="SLSQP", jac=True,
            bounds=self.bounds, constraints=constraint, options=self.options,
        )
        candidate = q1
        if second.success and np.isfinite(second.x).all():
            q2 = np.clip(second.x, self.model.lower, self.model.upper)
            if evaluate(q2)[0] > limit:
                low, high = q1, q2
                for _ in range(16):
                    middle = (low + high) / 2
                    if evaluate(middle)[0] <= limit:
                        low = middle
                    else:
                        high = middle
                q2 = low
            if evaluate(q2)[0] <= limit and evaluate(q2)[2] <= stage2_at_q1 + EPS:
                candidate = q2
        final_stage1, _, final_stage2, _ = evaluate(candidate)
        self.q = candidate.copy()
        self.losses = (float(stage1_loss), float(final_stage1), float(final_stage2))
        return self.q.copy()

    def pause(self):
        self.q = self.model.seed.copy()
        self.losses = None
