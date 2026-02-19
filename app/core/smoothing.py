from __future__ import annotations

from typing import Dict

import numpy as np


class JointSmoother:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.state: Dict[int, np.ndarray] = {}

    def update(self, joint_idx: int, xyz: np.ndarray) -> np.ndarray:
        if joint_idx not in self.state:
            self.state[joint_idx] = xyz.copy()
            return xyz
        prev = self.state[joint_idx]
        filtered = self.alpha * xyz + (1.0 - self.alpha) * prev
        self.state[joint_idx] = filtered
        return filtered
