from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from ultralytics import YOLO

from app.core.constants import COCO_JOINTS
from app.models.config import ModelConfig


class PoseEstimator:
    def __init__(self, cfg: ModelConfig):
        self.model = YOLO(cfg.path)
        self.conf = cfg.conf
        self.iou = cfg.iou
        self.device = cfg.device

    def detect(self, frame: np.ndarray) -> Dict[int, Tuple[float, float, float]]:
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        if not results:
            return {}

        result = results[0]
        kpts = result.keypoints
        if kpts is None or kpts.xy is None:
            return {}

        xy = kpts.xy.cpu().numpy()
        if xy.size == 0:
            return {}

        if kpts.conf is None:
            conf = np.ones((xy.shape[0], xy.shape[1]), dtype=np.float32)
        else:
            conf = kpts.conf.cpu().numpy()

        person_idx = int(np.argmax(conf.mean(axis=1)))
        person_xy = xy[person_idx]
        person_conf = conf[person_idx]

        joints: Dict[int, Tuple[float, float, float]] = {}
        for joint_idx in range(min(len(COCO_JOINTS), person_xy.shape[0])):
            x, y = person_xy[joint_idx]
            c = float(person_conf[joint_idx])
            if np.isfinite(x) and np.isfinite(y):
                joints[joint_idx] = (float(x), float(y), c)
        return joints
