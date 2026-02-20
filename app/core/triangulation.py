from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from app.models.config import TriangulationConfig
from app.services.calibration_store import CalibrationStore


@dataclass
class JointEstimate:
    xyz: np.ndarray
    mode: str
    inlier_camera_ids: list[str]
    reprojection_error: float | None


class TriangulationEngine:
    def __init__(self, calibration: CalibrationStore, cfg: TriangulationConfig):
        self.calibration = calibration
        self.cfg = cfg
        self.projections: Dict[str, np.ndarray] = {}
        self.rotations: Dict[str, np.ndarray] = {}
        self.translations: Dict[str, np.ndarray] = {}
        self.k_inv: Dict[str, np.ndarray] = {}
        self._build_projection_matrices()

    def _build_projection_matrices(self) -> None:
        self.projections = {}
        self.rotations = {}
        self.translations = {}
        self.k_inv = {}
        for cam_id in self.calibration.intrinsics:
            if cam_id not in self.calibration.extrinsics:
                continue
            k_mat, _ = self.calibration.intrinsics[cam_id]
            r_mat, t_vec = self.calibration.extrinsics[cam_id]
            k = np.array(k_mat, dtype=np.float64)
            r = np.array(r_mat, dtype=np.float64)
            t = np.array(t_vec, dtype=np.float64).reshape(3, 1)
            self.projections[cam_id] = k @ np.hstack([r, t])
            self.rotations[cam_id] = r
            self.translations[cam_id] = t
            self.k_inv[cam_id] = np.linalg.inv(k)

    def _project(self, cam_id: str, xyz: np.ndarray) -> np.ndarray:
        proj = self.projections[cam_id]
        xyz_h = np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float64)
        uvw = proj @ xyz_h
        if abs(float(uvw[2])) < 1e-8:
            return np.array([np.nan, np.nan], dtype=np.float64)
        return np.array([uvw[0] / uvw[2], uvw[1] / uvw[2]], dtype=np.float64)

    def _reprojection_errors(
        self, xyz: np.ndarray, observations: Dict[str, Tuple[float, float, float]]
    ) -> dict[str, float]:
        errors: dict[str, float] = {}
        for cam_id, (x, y, conf) in observations.items():
            if cam_id not in self.projections or conf < self.cfg.pair_conf_threshold:
                continue
            uv = self._project(cam_id, xyz)
            if np.any(np.isnan(uv)):
                continue
            errors[cam_id] = float(np.linalg.norm(uv - np.array([x, y], dtype=np.float64)))
        return errors

    def _triangulate_pair(
        self,
        cam_a: str,
        cam_b: str,
        obs_a: Tuple[float, float, float],
        obs_b: Tuple[float, float, float],
    ) -> Optional[np.ndarray]:
        p1 = self.projections[cam_a]
        p2 = self.projections[cam_b]
        x1, y1, _ = obs_a
        x2, y2, _ = obs_b
        xyz_h = cv2.triangulatePoints(
            p1,
            p2,
            np.array([[x1], [y1]], dtype=np.float64),
            np.array([[x2], [y2]], dtype=np.float64),
        )
        if abs(float(xyz_h[3, 0])) < 1e-9:
            return None
        xyz = (xyz_h[:3, 0] / xyz_h[3, 0]).astype(np.float64)
        if not np.all(np.isfinite(xyz)):
            return None
        return xyz

    def _triangulate_weighted(
        self,
        observations: Dict[str, Tuple[float, float, float]],
    ) -> Optional[np.ndarray]:
        rows = []
        for cam_id, (x, y, conf) in observations.items():
            if cam_id not in self.projections:
                continue
            p = self.projections[cam_id]
            w = max(1e-4, float(conf))
            rows.append(w * ((float(x) * p[2, :]) - p[0, :]))
            rows.append(w * ((float(y) * p[2, :]) - p[1, :]))
        if len(rows) < 4:
            return None
        a_mat = np.array(rows, dtype=np.float64)
        _, _, vt = np.linalg.svd(a_mat, full_matrices=False)
        xyz_h = vt[-1, :]
        if abs(float(xyz_h[3])) < 1e-9:
            return None
        xyz = (xyz_h[:3] / xyz_h[3]).astype(np.float64)
        if not np.all(np.isfinite(xyz)):
            return None
        return xyz

    def _single_view_fallback(
        self,
        cam_id: str,
        observation: Tuple[float, float, float],
        prior_xyz: np.ndarray | None,
        prior_timestamp: float | None,
        timestamp: float | None,
    ) -> Optional[np.ndarray]:
        if not bool(self.cfg.allow_single_view_fallback):
            return None
        if prior_xyz is None or prior_timestamp is None or timestamp is None:
            return None
        age_ms = max(0.0, (float(timestamp) - float(prior_timestamp)) * 1000.0)
        if age_ms > float(self.cfg.single_view_max_age_ms):
            return None
        if cam_id not in self.k_inv or cam_id not in self.rotations or cam_id not in self.translations:
            return None

        x, y, _ = observation
        pixel = np.array([float(x), float(y), 1.0], dtype=np.float64)
        ray = self.k_inv[cam_id] @ pixel
        if not np.all(np.isfinite(ray)):
            return None
        if abs(float(ray[2])) < 1e-8:
            return None
        ray = ray / float(ray[2])

        prior = np.array(prior_xyz, dtype=np.float64).reshape(3, 1)
        r_mat = self.rotations[cam_id]
        t_vec = self.translations[cam_id]
        prior_cam = r_mat @ prior + t_vec
        depth = float(prior_cam[2, 0])
        if depth <= 1e-6:
            return None

        cam_point = (ray * depth).reshape(3, 1)
        world_point = r_mat.T @ (cam_point - t_vec)
        xyz = world_point.reshape(3).astype(np.float64)
        if not np.all(np.isfinite(xyz)):
            return None
        return xyz

    def estimate_joint(
        self,
        observations: Dict[str, Tuple[float, float, float]],
        *,
        prior_xyz: np.ndarray | None = None,
        prior_timestamp: float | None = None,
        timestamp: float | None = None,
    ) -> Optional[JointEstimate]:
        valid = {}
        for cid, obs in observations.items():
            if cid not in self.projections:
                continue
            x, y, conf = obs
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            if float(conf) < self.cfg.pair_conf_threshold:
                continue
            valid[cid] = (float(x), float(y), float(conf))

        if not valid:
            return None

        if len(valid) == 1:
            cam_id, obs = next(iter(valid.items()))
            xyz = self._single_view_fallback(
                cam_id,
                obs,
                prior_xyz=prior_xyz,
                prior_timestamp=prior_timestamp,
                timestamp=timestamp,
            )
            if xyz is None:
                return None
            return JointEstimate(
                xyz=xyz,
                mode="single_view",
                inlier_camera_ids=[cam_id],
                reprojection_error=None,
            )

        best_xyz: np.ndarray | None = None
        best_inliers: list[str] = []
        best_med_err = float("inf")
        best_mean_conf = -1.0

        for cam_a, cam_b in itertools.combinations(valid.keys(), 2):
            xyz = self._triangulate_pair(cam_a, cam_b, valid[cam_a], valid[cam_b])
            if xyz is None:
                continue
            errors = self._reprojection_errors(xyz, valid)
            inliers = [cid for cid, err in errors.items() if err <= self.cfg.reproj_error_max]
            if not inliers:
                continue
            med_err = float(np.median([errors[cid] for cid in inliers]))
            mean_conf = float(np.mean([valid[cid][2] for cid in inliers]))
            if len(inliers) > len(best_inliers):
                best_xyz = xyz
                best_inliers = inliers
                best_med_err = med_err
                best_mean_conf = mean_conf
                continue
            if len(inliers) == len(best_inliers) and med_err < best_med_err:
                best_xyz = xyz
                best_inliers = inliers
                best_med_err = med_err
                best_mean_conf = mean_conf
                continue
            if (
                len(inliers) == len(best_inliers)
                and abs(med_err - best_med_err) <= 1e-9
                and mean_conf > best_mean_conf
            ):
                best_xyz = xyz
                best_inliers = inliers
                best_med_err = med_err
                best_mean_conf = mean_conf

        if best_xyz is None:
            return None

        if len(best_inliers) >= self.cfg.min_views:
            inlier_obs = {cid: valid[cid] for cid in best_inliers}
            refined = self._triangulate_weighted(inlier_obs)
            if refined is None:
                refined = best_xyz
            errors = self._reprojection_errors(refined, valid)
            inliers = [cid for cid, err in errors.items() if err <= self.cfg.reproj_error_max]
            if len(inliers) >= self.cfg.min_views:
                inlier_obs2 = {cid: valid[cid] for cid in inliers}
                refined2 = self._triangulate_weighted(inlier_obs2)
                if refined2 is None:
                    refined2 = refined
                errors2 = self._reprojection_errors(refined2, valid)
                inliers2 = [cid for cid, err in errors2.items() if err <= self.cfg.reproj_error_max]
                if len(inliers2) >= self.cfg.min_views:
                    reproj = float(np.median([errors2[cid] for cid in inliers2]))
                    return JointEstimate(
                        xyz=refined2,
                        mode="measured",
                        inlier_camera_ids=sorted(inliers2),
                        reprojection_error=reproj,
                    )
                if len(inliers2) == 1:
                    cid = inliers2[0]
                    fallback = self._single_view_fallback(
                        cid,
                        valid[cid],
                        prior_xyz=prior_xyz,
                        prior_timestamp=prior_timestamp,
                        timestamp=timestamp,
                    )
                    if fallback is not None:
                        return JointEstimate(
                            xyz=fallback,
                            mode="single_view",
                            inlier_camera_ids=[cid],
                            reprojection_error=errors2.get(cid),
                        )
                return None
            if len(inliers) == 1:
                cid = inliers[0]
                fallback = self._single_view_fallback(
                    cid,
                    valid[cid],
                    prior_xyz=prior_xyz,
                    prior_timestamp=prior_timestamp,
                    timestamp=timestamp,
                )
                if fallback is None:
                    return None
                return JointEstimate(
                    xyz=fallback,
                    mode="single_view",
                    inlier_camera_ids=[cid],
                    reprojection_error=errors.get(cid),
                )
            return None

        if len(best_inliers) == 1:
            cam_id = best_inliers[0]
            fallback = self._single_view_fallback(
                cam_id,
                valid[cam_id],
                prior_xyz=prior_xyz,
                prior_timestamp=prior_timestamp,
                timestamp=timestamp,
            )
            if fallback is None:
                return None
            return JointEstimate(
                xyz=fallback,
                mode="single_view",
                inlier_camera_ids=[cam_id],
                reprojection_error=best_med_err if np.isfinite(best_med_err) else None,
            )

        return None

    def triangulate_joint(
        self,
        observations: Dict[str, Tuple[float, float, float]],
        *,
        prior_xyz: np.ndarray | None = None,
        prior_timestamp: float | None = None,
        timestamp: float | None = None,
    ) -> Optional[np.ndarray]:
        estimate = self.estimate_joint(
            observations,
            prior_xyz=prior_xyz,
            prior_timestamp=prior_timestamp,
            timestamp=timestamp,
        )
        if estimate is None:
            return None
        return estimate.xyz
