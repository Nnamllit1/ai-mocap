from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np

from app.core.capture import CaptureHub
from app.models.config import AppConfig
from app.services.calibration_store import CalibrationStore


@dataclass
class CalibrationSession:
    active: bool = False
    camera_ids: List[str] = field(default_factory=list)
    object_points: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    image_points: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    image_sizes: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    captures: int = 0
    report: Dict = field(default_factory=dict)
    failure_streak: int = 0
    recommended_fps_cap: int | None = None
    sync_skew_history_ms: List[float] = field(default_factory=list)


class CalibrationService:
    _detect_timeout_s = 0.35

    def __init__(self, cfg: AppConfig, capture_hub: CaptureHub):
        self.cfg = cfg
        self.capture_hub = capture_hub
        self.session = CalibrationSession()
        workers = max(2, min(4, int(os.cpu_count() or 2)))
        self._detector_pool = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="checkerboard",
        )

    def _detect_many(
        self,
        frames_by_cam: Dict[str, np.ndarray],
        board_size: Tuple[int, int],
    ) -> Dict[str, Tuple[bool, np.ndarray | None]]:
        if not frames_by_cam:
            return {}
        futures = {
            cam_id: self._detector_pool.submit(self._detect_checkerboard, frame, board_size)
            for cam_id, frame in frames_by_cam.items()
        }
        out: Dict[str, Tuple[bool, np.ndarray | None]] = {}
        for cam_id, future in futures.items():
            try:
                out[cam_id] = future.result(timeout=self._detect_timeout_s)
            except FuturesTimeoutError:
                future.cancel()
                out[cam_id] = (False, None)
            except Exception:
                out[cam_id] = (False, None)
        return out

    def _detect_checkerboard(self, frame: np.ndarray, board_size: Tuple[int, int]) -> Tuple[bool, np.ndarray | None]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scales = (1.0, 1.5, 2.0, 2.75)

        # Prefer the SB detector when available; it is more robust on perspective tilt and low contrast.
        normalize_flag = int(getattr(cv2, "CALIB_CB_NORMALIZE_IMAGE", 0))
        exhaustive_flag = int(getattr(cv2, "CALIB_CB_EXHAUSTIVE", 0))
        accuracy_flag = int(getattr(cv2, "CALIB_CB_ACCURACY", 0))
        larger_flag = int(getattr(cv2, "CALIB_CB_LARGER", 0))
        sb_flag_sets = (
            normalize_flag,
            normalize_flag | exhaustive_flag | accuracy_flag,
            normalize_flag | exhaustive_flag | accuracy_flag | larger_flag,
        )
        sb_finder = getattr(cv2, "findChessboardCornersSB", None)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        proc_variants = (
            gray,
            cv2.equalizeHist(gray),
            clahe.apply(gray),
            cv2.GaussianBlur(gray, (5, 5), 0),
        )
        if callable(sb_finder):
            for processed in proc_variants:
                for scale in scales:
                    scaled = (
                        processed
                        if scale == 1.0
                        else cv2.resize(
                            processed, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
                        )
                    )
                    for sb_flags in sb_flag_sets:
                        found, corners = sb_finder(scaled, board_size, sb_flags)
                        if found and corners is not None:
                            corners = corners.astype(np.float32)
                            if scale != 1.0:
                                corners /= float(scale)
                            return True, corners

        legacy_flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_FILTER_QUADS
        )
        for processed in proc_variants:
            for scale in scales:
                scaled = (
                    processed
                    if scale == 1.0
                    else cv2.resize(processed, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                )
                found, corners = cv2.findChessboardCorners(scaled, board_size, legacy_flags)
                if found and corners is not None:
                    corners = corners.astype(np.float32)
                    if scale != 1.0:
                        corners /= float(scale)
                    # Refine on original image for calibration quality.
                    cv2.cornerSubPix(
                        gray,
                        corners,
                        winSize=(11, 11),
                        zeroZone=(-1, -1),
                        criteria=(
                            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                            30,
                            1e-3,
                        ),
                    )
                    return True, corners
        return False, None

    def start(self, camera_ids: List[str]) -> Dict:
        if len(camera_ids) < 2:
            raise ValueError("Calibration requires at least 2 cameras.")
        self.session = CalibrationSession(
            active=True,
            camera_ids=list(camera_ids),
            object_points={cid: [] for cid in camera_ids},
            image_points={cid: [] for cid in camera_ids},
            report={},
            recommended_fps_cap=self.cfg.ingest.client_fps_cap,
        )
        return {"ok": True, "camera_ids": camera_ids}

    def _compute_effective_latency_ms(self) -> int:
        sync_cfg = self.cfg.calibration_sync
        base = max(int(self.cfg.runtime.max_latency_ms), int(sync_cfg.min_latency_ms))
        if not sync_cfg.adaptive_enabled:
            return min(base, int(sync_cfg.max_latency_ms))
        hist = self.session.sync_skew_history_ms[-sync_cfg.history_size :]
        if hist:
            sorted_hist = sorted(hist)
            idx = max(0, int(len(sorted_hist) * 0.9) - 1)
            p90 = sorted_hist[idx]
        else:
            p90 = 0.0
        margin = float(sync_cfg.jitter_factor) * float(p90)
        value = int(round(base + margin))
        return max(int(sync_cfg.min_latency_ms), min(value, int(sync_cfg.max_latency_ms)))

    def _score_calibration(
        self,
        rms_by_camera: Dict[str, float],
        pair_rms: Dict[str, float],
        store: CalibrationStore,
    ) -> Dict:
        tips: List[str] = []
        intr_values = list(rms_by_camera.values())
        pair_values = list(pair_rms.values())
        intr_avg = float(sum(intr_values) / len(intr_values)) if intr_values else None
        pair_avg = float(sum(pair_values) / len(pair_values)) if pair_values else None
        pair_max = float(max(pair_values)) if pair_values else None

        dist_max_abs = 0.0
        bad_dist_cams: List[str] = []
        for cid, (_, dist) in store.intrinsics.items():
            vals = [float(v) for v in np.array(dist).reshape(-1).tolist()]
            cam_abs = max((abs(v) for v in vals), default=0.0)
            dist_max_abs = max(dist_max_abs, cam_abs)
            if len(vals) >= 5 and (abs(vals[1]) > 2.5 or abs(vals[4]) > 8.0):
                bad_dist_cams.append(cid)

        score = 100.0
        if intr_avg is not None:
            score -= max(0.0, (intr_avg - 0.6) * 14.0)
            if intr_avg > 1.4:
                tips.append("Increase board pose diversity (corners + near/far + tilt) to reduce intrinsic RMS.")
        if pair_avg is not None:
            score -= max(0.0, (pair_avg - 0.8) * 18.0)
        if pair_max is not None and pair_max > 2.5:
            score -= min(20.0, (pair_max - 2.5) * 8.0)
            tips.append("Some camera pairs have high reprojection error. Re-capture with both boards fully visible.")
        if bad_dist_cams:
            score -= 12.0
            tips.append(f"Distortion looks overfit on: {', '.join(bad_dist_cams)}. Keep board larger in frame.")
        if self.session.captures < max(15, self.cfg.calibration.min_captures):
            tips.append("Use more captures (20-30) for better stability.")

        score = max(0.0, min(100.0, score))
        if score >= 90:
            rating = "A"
            verdict = "excellent"
        elif score >= 80:
            rating = "B"
            verdict = "good"
        elif score >= 70:
            rating = "C"
            verdict = "ok"
        elif score >= 55:
            rating = "D"
            verdict = "weak"
        else:
            rating = "E"
            verdict = "poor"

        if not tips:
            tips.append("Calibration looks stable. Keep this camera setup fixed for best tracking consistency.")

        return {
            "score": int(round(score)),
            "rating": rating,
            "verdict": verdict,
            "intrinsic_rms_avg": intr_avg,
            "pair_rms_avg": pair_avg,
            "pair_rms_max": pair_max,
            "distortion_max_abs": float(dist_max_abs),
            "tips": tips[:5],
        }

    def _maybe_downshift_fps(self) -> None:
        threshold = int(self.cfg.calibration_sync.fps_downshift_failure_streak)
        if threshold <= 0:
            return
        if self.session.failure_streak < threshold:
            return
        steps = [self.cfg.ingest.client_fps_cap, 15, 12, 10]
        steps = sorted(set(int(v) for v in steps if int(v) > 0), reverse=True)
        current = int(self.session.recommended_fps_cap or self.cfg.ingest.client_fps_cap)
        lower = [v for v in steps if v < current]
        if lower:
            self.session.recommended_fps_cap = lower[0]
        self.session.failure_streak = 0

    def readiness(self) -> Dict:
        if not self.session.active:
            return {
                "active": False,
                "camera_ids": [],
                "per_camera": {},
                "all_cameras_ready": False,
                "effective_latency_ms": self._compute_effective_latency_ms(),
                "sync_skew_ms": 0.0,
                "recommended_fps_cap": self.session.recommended_fps_cap,
            }

        camera_ids = self.session.camera_ids
        effective_latency_ms = self._compute_effective_latency_ms()
        frame_diag = self.capture_hub.get_frame_diagnostics(camera_ids, effective_latency_ms)
        latest_frames = self.capture_hub.get_latest_frames(camera_ids)
        chess_cols, chess_rows = self.cfg.calibration.chessboard
        board_size = (chess_cols, chess_rows)
        per_camera = frame_diag["per_camera"]
        frames_to_detect = {
            cid: frame_packet.frame
            for cid in camera_ids
            for frame_packet in [latest_frames.get(cid)]
            if frame_packet is not None
        }
        detect_results = self._detect_many(frames_to_detect, board_size)
        for cid in camera_ids:
            frame_packet = latest_frames.get(cid)
            checkerboard_detected = False
            if frame_packet is not None:
                found, _ = detect_results.get(cid, (False, None))
                checkerboard_detected = bool(found)
            per_camera[cid]["checkerboard_detected"] = checkerboard_detected

        all_ready = all(
            per_camera[cid]["connected"]
            and per_camera[cid]["in_sync"]
            and per_camera[cid]["checkerboard_detected"]
            for cid in camera_ids
        )
        return {
            "active": True,
            "camera_ids": camera_ids,
            "per_camera": per_camera,
            "all_cameras_ready": all_ready,
            "effective_latency_ms": effective_latency_ms,
            "sync_skew_ms": frame_diag["sync_skew_ms"],
            "recommended_fps_cap": self.session.recommended_fps_cap,
            "captures": self.session.captures,
            "required": self.cfg.calibration.min_captures,
        }

    def capture(self) -> Dict:
        if not self.session.active:
            raise RuntimeError("Calibration session is not active.")
        effective_latency_ms = self._compute_effective_latency_ms()
        frames = self.capture_hub.get_synced_frames(
            min_sources=len(self.session.camera_ids),
            max_latency_ms=effective_latency_ms,
        )
        frame_diag = self.capture_hub.get_frame_diagnostics(self.session.camera_ids, effective_latency_ms)
        self.session.sync_skew_history_ms.append(float(frame_diag["sync_skew_ms"]))
        self.session.sync_skew_history_ms = self.session.sync_skew_history_ms[
            -self.cfg.calibration_sync.history_size :
        ]
        if not all(cid in frames for cid in self.session.camera_ids):
            self.session.failure_streak += 1
            self._maybe_downshift_fps()
            return {
                "ok": False,
                "reason": "not_all_cameras_ready",
                "effective_latency_ms": effective_latency_ms,
                "sync_skew_ms": frame_diag["sync_skew_ms"],
                "recommended_fps_cap": self.session.recommended_fps_cap,
                "per_camera": frame_diag["per_camera"],
            }

        chess_cols, chess_rows = self.cfg.calibration.chessboard
        board_size = (chess_cols, chess_rows)
        square_size = self.cfg.calibration.square_size_m
        objp = np.zeros((chess_cols * chess_rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:chess_cols, 0:chess_rows].T.reshape(-1, 2)
        objp *= square_size

        corners_by_cam = {}
        detect_results = self._detect_many(
            {cid: frames[cid].frame for cid in self.session.camera_ids},
            board_size,
        )
        for cid in self.session.camera_ids:
            frame = frames[cid].frame
            found, corners = detect_results.get(cid, (False, None))
            if not found or corners is None:
                self.session.failure_streak = 0
                return {
                    "ok": False,
                    "reason": f"checkerboard_not_found:{cid}",
                    "effective_latency_ms": effective_latency_ms,
                    "sync_skew_ms": frame_diag["sync_skew_ms"],
                    "recommended_fps_cap": self.session.recommended_fps_cap,
                    "per_camera": frame_diag["per_camera"],
                }
            corners_by_cam[cid] = corners.astype(np.float32)
            self.session.image_sizes[cid] = (frame.shape[1], frame.shape[0])

        for cid in self.session.camera_ids:
            self.session.object_points[cid].append(objp.copy())
            self.session.image_points[cid].append(corners_by_cam[cid])

        self.session.captures += 1
        self.session.failure_streak = 0
        return {
            "ok": True,
            "captures": self.session.captures,
            "effective_latency_ms": effective_latency_ms,
            "sync_skew_ms": frame_diag["sync_skew_ms"],
            "recommended_fps_cap": self.session.recommended_fps_cap,
            "per_camera": frame_diag["per_camera"],
        }

    def solve(self) -> Dict:
        if not self.session.active:
            raise RuntimeError("Calibration session is not active.")
        min_required = self.cfg.calibration.min_captures
        if self.session.captures < min_required:
            return {
                "ok": False,
                "reason": "not_enough_captures",
                "captures": self.session.captures,
                "required": min_required,
            }

        store = CalibrationStore()
        rms_by_camera = {}
        camera_ids = self.session.camera_ids

        for cid in camera_ids:
            rms, k_mat, dist, _, _ = cv2.calibrateCamera(
                self.session.object_points[cid],
                self.session.image_points[cid],
                self.session.image_sizes[cid],
                None,
                None,
            )
            store.intrinsics[cid] = (k_mat, dist)
            rms_by_camera[cid] = float(rms)

        ref_id = camera_ids[0]
        store.extrinsics[ref_id] = (np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64))

        pair_rms = {}
        for cid in camera_ids[1:]:
            k1, d1 = store.intrinsics[ref_id]
            k2, d2 = store.intrinsics[cid]
            rms, _, _, _, _, r_mat, t_vec, _, _ = cv2.stereoCalibrate(
                self.session.object_points[ref_id],
                self.session.image_points[ref_id],
                self.session.image_points[cid],
                k1,
                d1,
                k2,
                d2,
                self.session.image_sizes[ref_id],
                criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5),
                flags=cv2.CALIB_FIX_INTRINSIC,
            )
            store.extrinsics[cid] = (r_mat, t_vec)
            pair_rms[f"{ref_id}->{cid}"] = float(rms)

        path = self.cfg.calibration_path()
        store.save(path)
        self.session.report = {
            "ok": True,
            "captures": self.session.captures,
            "intrinsic_rms": rms_by_camera,
            "pair_rms": pair_rms,
            "calibration_score": self._score_calibration(rms_by_camera, pair_rms, store),
            "path": str(path),
        }
        return self.session.report

    def report(self) -> Dict:
        return {
            "active": self.session.active,
            "camera_ids": self.session.camera_ids,
            "captures": self.session.captures,
            "recommended_fps_cap": self.session.recommended_fps_cap,
            "result": self.session.report,
        }
