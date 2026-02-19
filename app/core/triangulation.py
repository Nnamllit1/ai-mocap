from __future__ import annotations

import itertools
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from app.models.config import TriangulationConfig
from app.services.calibration_store import CalibrationStore


class TriangulationEngine:
    def __init__(self, calibration: CalibrationStore, cfg: TriangulationConfig):
        self.calibration = calibration
        self.cfg = cfg
        self.projections: Dict[str, np.ndarray] = {}
        self._build_projection_matrices()

    def _build_projection_matrices(self) -> None:
        self.projections = {}
        for cam_id in self.calibration.intrinsics:
            if cam_id not in self.calibration.extrinsics:
                continue
            k_mat, _ = self.calibration.intrinsics[cam_id]
            r_mat, t_vec = self.calibration.extrinsics[cam_id]
            self.projections[cam_id] = k_mat @ np.hstack([r_mat, t_vec])

    def _project(self, cam_id: str, xyz: np.ndarray) -> np.ndarray:
        proj = self.projections[cam_id]
        xyz_h = np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float64)
        uvw = proj @ xyz_h
        if abs(float(uvw[2])) < 1e-8:
            return np.array([np.nan, np.nan], dtype=np.float64)
        return np.array([uvw[0] / uvw[2], uvw[1] / uvw[2]], dtype=np.float64)

    def _reprojection_error(
        self, xyz: np.ndarray, observations: Dict[str, Tuple[float, float, float]]
    ) -> float:
        errors = []
        for cam_id, (x, y, conf) in observations.items():
            if cam_id not in self.projections or conf < self.cfg.pair_conf_threshold:
                continue
            uv = self._project(cam_id, xyz)
            if np.any(np.isnan(uv)):
                continue
            errors.append(float(np.linalg.norm(uv - np.array([x, y]))))
        if not errors:
            return float("inf")
        return float(np.median(errors))

    def triangulate_joint(
        self, observations: Dict[str, Tuple[float, float, float]]
    ) -> Optional[np.ndarray]:
        valid = {
            cid: obs
            for cid, obs in observations.items()
            if cid in self.projections and obs[2] >= self.cfg.pair_conf_threshold
        }
        if len(valid) < self.cfg.min_views:
            return None

        candidates = []
        for cam_a, cam_b in itertools.combinations(valid.keys(), 2):
            p1 = self.projections[cam_a]
            p2 = self.projections[cam_b]
            x1, y1, _ = valid[cam_a]
            x2, y2, _ = valid[cam_b]
            xyz_h = cv2.triangulatePoints(
                p1,
                p2,
                np.array([[x1], [y1]], dtype=np.float64),
                np.array([[x2], [y2]], dtype=np.float64),
            )
            if abs(float(xyz_h[3, 0])) < 1e-9:
                continue
            xyz = (xyz_h[:3, 0] / xyz_h[3, 0]).astype(np.float64)
            err = self._reprojection_error(xyz, valid)
            if err <= self.cfg.reproj_error_max:
                candidates.append(xyz)

        if not candidates:
            return None
        return np.median(np.stack(candidates, axis=0), axis=0)
