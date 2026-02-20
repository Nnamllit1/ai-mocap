from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class JointCacheItem:
    xyz: np.ndarray
    confidence: float
    timestamp: float


class JointStateTracker:
    def __init__(
        self,
        hold_ms: int = 250,
        max_jump_m: float = 0.35,
        jump_reject_conf: float = 0.85,
    ):
        self.hold_ms = max(0, int(hold_ms))
        self.max_jump_m = max(0.0, float(max_jump_m))
        self.jump_reject_conf = max(0.0, min(1.0, float(jump_reject_conf)))
        self._cache: Dict[int, JointCacheItem] = {}

    def reset(self) -> None:
        self._cache.clear()

    def stabilize(
        self,
        timestamp: float,
        measured_pose3d: Dict[int, np.ndarray],
        measured_confidences: Dict[int, float],
        measured_states: Dict[int, str] | None = None,
    ) -> dict[int, dict]:
        out: dict[int, dict] = {}
        now = float(timestamp)

        measured_states = measured_states or {}
        for joint_idx, xyz in measured_pose3d.items():
            conf = float(measured_confidences.get(joint_idx, 0.0))
            point = np.array(xyz, dtype=float)
            prev = self._cache.get(joint_idx)
            if prev is not None:
                jump_m = float(np.linalg.norm(point - np.array(prev.xyz, dtype=float)))
                if jump_m > self.max_jump_m and conf < self.jump_reject_conf:
                    age_ms = int(max(0.0, (now - float(prev.timestamp)) * 1000.0))
                    if age_ms <= self.hold_ms:
                        out[joint_idx] = {
                            "xyz": np.array(prev.xyz, dtype=float),
                            "confidence": float(prev.confidence),
                            "state": "held",
                            "age_ms": age_ms,
                        }
                    continue

            state = str(measured_states.get(joint_idx, "measured"))
            self._cache[joint_idx] = JointCacheItem(
                xyz=point, confidence=conf, timestamp=now
            )
            out[joint_idx] = {
                "xyz": point,
                "confidence": conf,
                "state": state,
                "age_ms": 0,
            }

        for joint_idx, item in self._cache.items():
            if joint_idx in out:
                continue
            age_ms = int(max(0.0, (now - float(item.timestamp)) * 1000.0))
            if age_ms > self.hold_ms:
                continue
            out[joint_idx] = {
                "xyz": np.array(item.xyz, dtype=float),
                "confidence": float(item.confidence),
                "state": "held",
                "age_ms": age_ms,
            }

        return out
