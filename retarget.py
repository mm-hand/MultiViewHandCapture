import json
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np

from config import (
    HUMAN_TO_ROBOT_PALM,
    IK_EPSILON,
    IK_ITERATIONS,
    IK_MAX_RMS,
    IK_MAX_STEP,
    IK_NO_IMPROVEMENT,
    IK_PINV_RCOND,
    IK_STOP_RMS,
    IK_WEIGHTS,
    JOINT_FILTER,
    ROBOT_BASE_LINK,
    ROBOT_CHAINS,
    ROBOT_FE_INDICES,
    ROBOT_TIP_LINKS,
    ROBOT_TIP_OFFSETS,
    STANDARD_PALM_SIZE,
    URDF_CONTRACT_PATH,
    URDF_PATH,
)
from hand_core import OneEuro

EPS = 1e-9


def _values(text, default):
    return np.fromstring(text if text is not None else default, sep=" ")


def _rotation(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / max(np.linalg.norm(axis), EPS)
    cross = np.array(((0, -axis[2], axis[1]), (axis[2], 0, -axis[0]), (-axis[1], axis[0], 0)))
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


def _origin(element):
    xyz = _values(element.get("xyz"), "0 0 0")
    roll, pitch, yaw = _values(element.get("rpy"), "0 0 0")
    rx, ry, rz = _rotation((1, 0, 0), roll), _rotation((0, 1, 0), pitch), _rotation((0, 0, 1), yaw)
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rz @ ry @ rx, xyz
    return transform


def _unit(vector, fallback):
    norm = np.linalg.norm(vector)
    return vector / norm if norm > EPS else fallback / max(np.linalg.norm(fallback), EPS)


class RobotModel:
    def __init__(self, urdf_path=URDF_PATH, contract_path=URDF_CONTRACT_PATH):
        contract = json.loads(contract_path.read_text())
        self.topic = contract["topic"]
        self.names = tuple(joint["name"] for joint in contract["joints"])
        self.layout = (
            f"mmhand:J00-J20:{contract['input_space']}:{contract['mapping_id']}:"
            f"v{contract['mapping_version']}:{contract['urdf_config_crc']}"
        )
        limits = np.radians([joint["limit_deg"] for joint in contract["joints"]])
        limits[list(ROBOT_FE_INDICES), 0] = 0
        self.lower, self.upper = limits.T

        self.joints, self.children = {}, defaultdict(list)
        for node in ET.parse(urdf_path).getroot().findall("joint"):
            origin = node.find("origin")
            joint = {
                "type": node.get("type"),
                "parent": node.find("parent").get("link"),
                "child": node.find("child").get("link"),
                "origin": _origin(origin) if origin is not None else np.eye(4),
                "axis": _values(node.find("axis").get("xyz"), "1 0 0")
                if node.find("axis") is not None
                else np.array((1.0, 0, 0)),
            }
            self.joints[node.get("name")] = joint
            self.children[joint["parent"]].append(node.get("name"))
        missing = set(self.names) - self.joints.keys()
        if missing:
            raise ValueError(f"URDF is missing contract joints: {sorted(missing)}")
        self.index = {name: i for i, name in enumerate(self.names)}
        self.fingers = tuple(ROBOT_CHAINS)

        self.neutral = self.points(np.zeros(21))
        long_mcps = [self.neutral[name][0] for name in self.fingers if name != "thumb"]
        forward = _unit(np.mean(long_mcps, axis=0) - self.neutral["palm"], np.array((1.0, 0, 0)))
        width = _unit(
            self.neutral["pinky"][0] - self.neutral["index"][0], np.array((0, 1.0, 0))
        )
        normal = _unit(np.cross(forward, width), np.array((0, 0, 1.0)))
        width = _unit(np.cross(normal, forward), width)
        self.palm_rotation = np.column_stack((forward, width, normal))
        self.human_to_base = self.palm_rotation @ np.asarray(HUMAN_TO_ROBOT_PALM, float)
        self.palm_size = np.mean(np.linalg.norm(np.asarray(long_mcps) - self.neutral["palm"], axis=1))
        self.palm_scale = self.palm_size / STANDARD_PALM_SIZE
        self.local_lengths = {
            name: np.linalg.norm(self._local(self.neutral, name), axis=1) for name in self.fingers
        }
        self.anchor_lengths = {
            name: np.linalg.norm(
                self.neutral[name][-1] - self.neutral[name][1 if name == "thumb" else 0]
            )
            for name in self.fingers
        }
        self.hand_lengths = {
            name: np.linalg.norm(self.neutral[name][-1] - self.neutral["palm"])
            for name in self.fingers
        }

    def fk(self, q):
        link_transforms, joint_origins = {ROBOT_BASE_LINK: np.eye(4)}, {}
        stack = [ROBOT_BASE_LINK]
        while stack:
            parent = stack.pop()
            for name in self.children[parent]:
                joint = self.joints[name]
                frame = link_transforms[parent] @ joint["origin"]
                joint_origins[name] = frame[:3, 3].copy()
                child = frame
                if joint["type"] in ("revolute", "continuous"):
                    motion = np.eye(4)
                    motion[:3, :3] = _rotation(joint["axis"], q[self.index[name]])
                    child = frame @ motion
                link_transforms[joint["child"]] = child
                stack.append(joint["child"])
        return link_transforms, joint_origins

    def points(self, q):
        links, origins = self.fk(q)
        output = {"palm": links[ROBOT_BASE_LINK][:3, 3].copy()}
        for name, (_, indices) in ROBOT_CHAINS.items():
            joints = [self.names[index] for index in indices]
            start = 1
            landmarks = [origins[joints[i]] for i in range(start, start + 3)]
            offset = np.r_[ROBOT_TIP_OFFSETS[name], 1.0]
            landmarks.append((links[ROBOT_TIP_LINKS[name]] @ offset)[:3])
            output[name] = np.asarray(landmarks)
        return output

    @staticmethod
    def _local(points, name):
        finger, palm = points[name], points["palm"]
        if name == "thumb":
            return np.asarray((finger[0] - palm, finger[1] - palm, finger[2] - finger[1], finger[3] - finger[2]))
        return np.asarray((finger[0] - palm, finger[1] - finger[0], finger[2] - finger[1], finger[3] - finger[2]))


class Retargeter:
    def __init__(self, model=None):
        self.model = RobotModel() if model is None else model
        self.raw_q = np.zeros(21)
        self.filter = OneEuro(*JOINT_FILTER)

    def _direction(self, human_vector, robot_vector, length):
        vector = self.model.human_to_base @ human_vector
        return _unit(vector, robot_vector) * length

    def targets(self, human):
        human = np.asarray(human, float)
        target = {"local": {}, "anchor": {}, "hand": {}, "thumb_tip": {}}
        for name, (indices, _) in ROBOT_CHAINS.items():
            finger = human[list(indices)]
            if name == "thumb":
                vectors = (finger[0] - human[0], finger[1] - human[0], finger[2] - finger[1], finger[3] - finger[2])
                anchor = finger[3] - finger[1]
            else:
                vectors = (finger[0] - human[0], finger[1] - finger[0], finger[2] - finger[1], finger[3] - finger[2])
                anchor = finger[3] - finger[0]
            neutral = self.model._local(self.model.neutral, name)
            target["local"][name] = np.asarray(
                [
                    self._direction(vector, reference, length)
                    for vector, reference, length in zip(vectors, neutral, self.model.local_lengths[name])
                ]
            )
            robot_anchor = self.model.neutral[name][-1] - self.model.neutral[name][
                1 if name == "thumb" else 0
            ]
            target["anchor"][name] = self._direction(
                anchor, robot_anchor, self.model.anchor_lengths[name]
            )
            target["hand"][name] = self._direction(
                finger[-1] - human[0],
                self.model.neutral[name][-1] - self.model.neutral["palm"],
                self.model.hand_lengths[name],
            )
        thumb_tip = human[ROBOT_CHAINS["thumb"][0][-1]]
        for name in self.model.fingers:
            if name != "thumb":
                tip = human[ROBOT_CHAINS[name][0][-1]]
                target["thumb_tip"][name] = self.model.human_to_base @ (tip - thumb_tip) * self.model.palm_scale
        return target

    def residual(self, q, target):
        points, residuals = self.model.points(q), []
        for name in self.model.fingers:
            local = (self.model._local(points, name) - target["local"][name]) / self.model.local_lengths[name][:, None]
            residuals.append(np.sqrt(IK_WEIGHTS["local"]) * local.ravel())
            anchor_index = 1 if name == "thumb" else 0
            anchor = points[name][-1] - points[name][anchor_index]
            residuals.append(
                np.sqrt(IK_WEIGHTS["anchor_tip"])
                * (anchor - target["anchor"][name])
                / self.model.anchor_lengths[name]
            )
            hand = points[name][-1] - points["palm"]
            residuals.append(
                np.sqrt(IK_WEIGHTS["hand_tip"])
                * (hand - target["hand"][name])
                / self.model.hand_lengths[name]
            )
        for name, expected in target["thumb_tip"].items():
            actual = points[name][-1] - points["thumb"][-1]
            residuals.append(
                np.sqrt(IK_WEIGHTS["thumb_tip"]) * (actual - expected) / self.model.palm_size
            )
        return np.concatenate(residuals)

    def jacobian(self, q, target, residual=None):
        residual = self.residual(q, target) if residual is None else residual
        jacobian = np.empty((len(residual), 21))
        for index in range(21):
            step = IK_EPSILON if q[index] < self.model.upper[index] else -IK_EPSILON
            sample = q.copy()
            sample[index] = np.clip(sample[index] + step, self.model.lower[index], self.model.upper[index])
            delta = sample[index] - q[index]
            jacobian[:, index] = (
                (self.residual(sample, target) - residual) / delta if abs(delta) > EPS else 0
            )
        return jacobian

    def solve(self, human, timestamp):
        human = np.asarray(human, float)
        if human.shape != (21, 3) or not np.isfinite(human).all():
            return self._failure(float("inf"))
        target, q = self.targets(human), self.raw_q.copy()
        residual = self.residual(q, target)
        best_q, best_rms = q.copy(), float(np.sqrt(np.mean(residual**2)))
        stalled = 0
        try:
            for _ in range(IK_ITERATIONS):
                if best_rms <= IK_STOP_RMS:
                    break
                step = -np.linalg.pinv(
                    self.jacobian(q, target, residual), rcond=IK_PINV_RCOND
                ) @ residual
                peak = np.max(np.abs(step))
                if peak > IK_MAX_STEP:
                    step *= IK_MAX_STEP / peak
                q = np.clip(q + step, self.model.lower, self.model.upper)
                residual = self.residual(q, target)
                rms = float(np.sqrt(np.mean(residual**2)))
                if rms + 1e-7 < best_rms:
                    best_q, best_rms, stalled = q.copy(), rms, 0
                else:
                    stalled += 1
                    if stalled >= IK_NO_IMPROVEMENT:
                        break
        except np.linalg.LinAlgError:
            return self._failure(best_rms)
        self.raw_q = best_q
        success = np.isfinite(best_rms) and best_rms <= IK_MAX_RMS
        filtered = np.clip(self.filter(best_q, timestamp), self.model.lower, self.model.upper) if success else None
        return {
            "success": success,
            "q": filtered,
            "q_raw": best_q,
            "q_deg": None if filtered is None else np.degrees(filtered),
            "rms": best_rms,
            "points": self.model.points(best_q if filtered is None else filtered),
        }

    def _failure(self, rms):
        return {
            "success": False,
            "q": None,
            "q_raw": self.raw_q.copy(),
            "q_deg": None,
            "rms": rms,
            "points": self.model.points(self.raw_q),
        }

    def pause(self):
        self.filter.reset()
