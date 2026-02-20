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
        self.projections_norm: Dict[str, np.ndarray] = {}
        self.rotations: Dict[str, np.ndarray] = {}
        self.translations: Dict[str, np.ndarray] = {}
        self.intrinsics: Dict[str, np.ndarray] = {}
        self.distortions: Dict[str, np.ndarray] = {}
        self.rvecs: Dict[str, np.ndarray] = {}
        self._build_projection_matrices()

    def _build_projection_matrices(self) -> None:
        self.projections = {}
        self.projections_norm = {}
        self.rotations = {}
        self.translations = {}
        self.intrinsics = {}
        self.distortions = {}
        self.rvecs = {}

        for cam_id in self.calibration.intrinsics:
            if cam_id not in self.calibration.extrinsics:
                continue

            k_mat, dist = self.calibration.intrinsics[cam_id]
            r_mat, t_vec = self.calibration.extrinsics[cam_id]
            k = np.array(k_mat, dtype=np.float64)
            d = np.array(dist, dtype=np.float64).reshape(-1, 1)
            r = np.array(r_mat, dtype=np.float64)
            t = np.array(t_vec, dtype=np.float64).reshape(3, 1)

            p_norm = np.hstack([r, t])
            self.projections_norm[cam_id] = p_norm
            self.projections[cam_id] = k @ p_norm
            self.rotations[cam_id] = r
            self.translations[cam_id] = t
            self.intrinsics[cam_id] = k
            self.distortions[cam_id] = d
            rvec, _ = cv2.Rodrigues(r)
            self.rvecs[cam_id] = rvec

    @staticmethod
    def _is_finite_xy(x: float, y: float) -> bool:
        return bool(np.isfinite(x) and np.isfinite(y))

    def _required_views(self) -> int:
        return max(2, int(self.cfg.min_views))

    def _undistort_normalized(self, cam_id: str, x: float, y: float) -> Optional[np.ndarray]:
        if cam_id not in self.intrinsics or cam_id not in self.distortions:
            return None
        points = np.array([[[float(x), float(y)]]], dtype=np.float64)
        norm = cv2.undistortPoints(
            points,
            self.intrinsics[cam_id],
            self.distortions[cam_id],
            P=None,
        )
        if norm is None or norm.shape[0] == 0:
            return None
        uv = np.array(norm[0, 0, :], dtype=np.float64)
        if not np.all(np.isfinite(uv)):
            return None
        return uv

    def _project(self, cam_id: str, xyz: np.ndarray) -> np.ndarray:
        if (
            cam_id not in self.intrinsics
            or cam_id not in self.distortions
            or cam_id not in self.rvecs
            or cam_id not in self.translations
        ):
            return np.array([np.nan, np.nan], dtype=np.float64)
        point = np.array(xyz, dtype=np.float64).reshape(1, 1, 3)
        try:
            image_points, _ = cv2.projectPoints(
                point,
                self.rvecs[cam_id],
                self.translations[cam_id],
                self.intrinsics[cam_id],
                self.distortions[cam_id],
            )
        except cv2.error:
            return np.array([np.nan, np.nan], dtype=np.float64)
        uv = np.array(image_points[0, 0, :], dtype=np.float64)
        if not np.all(np.isfinite(uv)):
            return np.array([np.nan, np.nan], dtype=np.float64)
        return uv

    def _depth_in_camera(self, cam_id: str, xyz: np.ndarray) -> float:
        if cam_id not in self.rotations or cam_id not in self.translations:
            return float("nan")
        world = np.array(xyz, dtype=np.float64).reshape(3, 1)
        cam = self.rotations[cam_id] @ world + self.translations[cam_id]
        return float(cam[2, 0])

    def _reprojection_errors(
        self, xyz: np.ndarray, observations: Dict[str, Tuple[float, float, float]]
    ) -> dict[str, float]:
        errors: dict[str, float] = {}
        for cam_id, (x, y, conf) in observations.items():
            if cam_id not in self.projections:
                continue
            if float(conf) < self.cfg.pair_conf_threshold:
                continue
            if self._depth_in_camera(cam_id, xyz) <= 1e-6:
                errors[cam_id] = float("inf")
                continue
            uv = self._project(cam_id, xyz)
            if not np.all(np.isfinite(uv)):
                errors[cam_id] = float("inf")
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
        if cam_a not in self.projections_norm or cam_b not in self.projections_norm:
            return None

        x1, y1, _ = obs_a
        x2, y2, _ = obs_b
        uv1 = self._undistort_normalized(cam_a, x1, y1)
        uv2 = self._undistort_normalized(cam_b, x2, y2)
        if uv1 is None or uv2 is None:
            return None

        xyz_h = cv2.triangulatePoints(
            self.projections_norm[cam_a],
            self.projections_norm[cam_b],
            uv1.reshape(2, 1),
            uv2.reshape(2, 1),
        )
        if abs(float(xyz_h[3, 0])) < 1e-9:
            return None
        xyz = (xyz_h[:3, 0] / xyz_h[3, 0]).astype(np.float64)
        if not np.all(np.isfinite(xyz)):
            return None
        if self._depth_in_camera(cam_a, xyz) <= 1e-6 or self._depth_in_camera(cam_b, xyz) <= 1e-6:
            return None
        return xyz

    def _triangulate_weighted(
        self,
        observations: Dict[str, Tuple[float, float, float]],
    ) -> Optional[np.ndarray]:
        rows = []
        for cam_id, (x, y, conf) in observations.items():
            if cam_id not in self.projections_norm:
                continue
            uv = self._undistort_normalized(cam_id, x, y)
            if uv is None:
                continue
            p = self.projections_norm[cam_id]
            w = max(1e-4, float(conf))
            rows.append(w * ((float(uv[0]) * p[2, :]) - p[0, :]))
            rows.append(w * ((float(uv[1]) * p[2, :]) - p[1, :]))

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
        if cam_id not in self.rotations or cam_id not in self.translations:
            return None

        x, y, _ = observation
        uv = self._undistort_normalized(cam_id, x, y)
        if uv is None:
            return None
        ray_cam = np.array([float(uv[0]), float(uv[1]), 1.0], dtype=np.float64).reshape(3, 1)
        if not np.all(np.isfinite(ray_cam)):
            return None

        prior = np.array(prior_xyz, dtype=np.float64).reshape(3, 1)
        r_mat = self.rotations[cam_id]
        t_vec = self.translations[cam_id]
        prior_cam = r_mat @ prior + t_vec
        depth = float(prior_cam[2, 0])
        if depth <= 1e-6:
            return None

        cam_point = ray_cam * depth
        world_point = r_mat.T @ (cam_point - t_vec)
        xyz = world_point.reshape(3).astype(np.float64)
        if not np.all(np.isfinite(xyz)):
            return None
        if self._depth_in_camera(cam_id, xyz) <= 1e-6:
            return None
        return xyz

    def _select_best_candidate(
        self, valid: Dict[str, Tuple[float, float, float]]
    ) -> tuple[np.ndarray | None, list[str], float]:
        best_xyz: np.ndarray | None = None
        best_inliers: list[str] = []
        best_med_err = float("inf")
        best_mean_conf = -1.0

        for cam_a, cam_b in itertools.combinations(valid.keys(), 2):
            xyz = self._triangulate_pair(cam_a, cam_b, valid[cam_a], valid[cam_b])
            if xyz is None:
                continue
            errors = self._reprojection_errors(xyz, valid)
            inliers = sorted(
                cid
                for cid, err in errors.items()
                if np.isfinite(err) and err <= float(self.cfg.reproj_error_max)
            )
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

        if best_xyz is not None:
            return best_xyz, best_inliers, best_med_err

        weighted = self._triangulate_weighted(valid)
        if weighted is None:
            return None, [], float("inf")
        errors = self._reprojection_errors(weighted, valid)
        inliers = sorted(
            cid
            for cid, err in errors.items()
            if np.isfinite(err) and err <= float(self.cfg.reproj_error_max)
        )
        if not inliers:
            return None, [], float("inf")
        med_err = float(np.median([errors[cid] for cid in inliers]))
        return weighted, inliers, med_err

    def estimate_joint(
        self,
        observations: Dict[str, Tuple[float, float, float]],
        *,
        prior_xyz: np.ndarray | None = None,
        prior_timestamp: float | None = None,
        timestamp: float | None = None,
    ) -> Optional[JointEstimate]:
        valid: Dict[str, Tuple[float, float, float]] = {}
        for cid, obs in observations.items():
            if cid not in self.projections:
                continue
            x, y, conf = obs
            if not self._is_finite_xy(float(x), float(y)):
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

        best_xyz, best_inliers, _ = self._select_best_candidate(valid)
        if best_xyz is None:
            return None

        inlier_ids = list(best_inliers)
        refined_xyz = np.array(best_xyz, dtype=np.float64)
        errors = self._reprojection_errors(refined_xyz, valid)

        for _ in range(2):
            if len(inlier_ids) >= 2:
                refined_obs = {cid: valid[cid] for cid in inlier_ids}
                weighted = self._triangulate_weighted(refined_obs)
                if weighted is not None:
                    refined_xyz = weighted
            errors = self._reprojection_errors(refined_xyz, valid)
            new_inliers = sorted(
                cid
                for cid, err in errors.items()
                if np.isfinite(err) and err <= float(self.cfg.reproj_error_max)
            )
            if new_inliers == inlier_ids:
                break
            inlier_ids = new_inliers

        required_views = self._required_views()
        if len(inlier_ids) >= required_views:
            final_obs = {cid: valid[cid] for cid in inlier_ids}
            final_xyz = self._triangulate_weighted(final_obs)
            if final_xyz is None:
                final_xyz = refined_xyz
            final_errors = self._reprojection_errors(final_xyz, valid)
            final_inliers = sorted(
                cid
                for cid, err in final_errors.items()
                if np.isfinite(err) and err <= float(self.cfg.reproj_error_max)
            )
            if len(final_inliers) >= required_views:
                reproj = float(np.median([final_errors[cid] for cid in final_inliers]))
                return JointEstimate(
                    xyz=final_xyz,
                    mode="measured",
                    inlier_camera_ids=final_inliers,
                    reprojection_error=reproj,
                )
            inlier_ids = final_inliers
            errors = final_errors

        if len(inlier_ids) == 1:
            cam_id = inlier_ids[0]
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
                reprojection_error=errors.get(cam_id),
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
