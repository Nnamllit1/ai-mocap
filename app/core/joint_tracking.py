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
    def __init__(self, hold_ms: int = 250):
        self.hold_ms = max(0, int(hold_ms))
        self._cache: Dict[int, JointCacheItem] = {}

    def reset(self) -> None:
        self._cache.clear()

    def stabilize(
        self,
        timestamp: float,
        measured_pose3d: Dict[int, np.ndarray],
        measured_confidences: Dict[int, float],
    ) -> dict[int, dict]:
        out: dict[int, dict] = {}
        now = float(timestamp)

        for joint_idx, xyz in measured_pose3d.items():
            conf = float(measured_confidences.get(joint_idx, 0.0))
            point = np.array(xyz, dtype=float)
            self._cache[joint_idx] = JointCacheItem(
                xyz=point, confidence=conf, timestamp=now
            )
            out[joint_idx] = {
                "xyz": point,
                "confidence": conf,
                "state": "measured",
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
