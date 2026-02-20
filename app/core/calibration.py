from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from app.core.capture import CaptureHub
from app.models.config import AppConfig
from app.services.calibration_store import CalibrationStore
from app.services.state_io import load_json, save_json_atomic


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
    last_accept_ts_ms: float | None = None
    last_accept_pose: Dict[str, float] | None = None
    stable_since_ts_ms: float | None = None
    last_readiness_metrics: Dict[str, Any] = field(default_factory=dict)
    last_pose_sample: Dict[str, float] | None = None
    resume_pending: bool = False
    resume_policy: str = "manual"
    resume_deadline_ts_ms: float | None = None
    resume_snapshot_ts_ms: float | None = None
    resume_auto_reset: bool = False
    resume_reason: str | None = None


class CalibrationService:
    _detect_timeout_s = 0.35

    def __init__(
        self,
        cfg: AppConfig,
        capture_hub: CaptureHub,
        session_state_path: str | Path | None = None,
    ):
        self.cfg = cfg
        self.capture_hub = capture_hub
        self._session_state_path = Path(session_state_path) if session_state_path else None
        self.session = CalibrationSession()
        workers = max(2, min(4, int(os.cpu_count() or 2)))
        self._detector_pool = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="checkerboard",
        )
        self._restore_session_snapshot()

    @staticmethod
    def _to_list_points(points: Dict[str, List[np.ndarray]]) -> Dict[str, List[list]]:
        out: Dict[str, List[list]] = {}
        for cid, arrs in points.items():
            out[cid] = [np.array(a).tolist() for a in arrs]
        return out

    @staticmethod
    def _from_list_points(points: Dict[str, List[list]], dtype: np.dtype) -> Dict[str, List[np.ndarray]]:
        out: Dict[str, List[np.ndarray]] = {}
        for cid, arrs in (points or {}).items():
            out[cid] = [np.array(a, dtype=dtype) for a in arrs]
        return out

    @staticmethod
    def _sanitize_resume_policy(policy: str | None) -> str:
        value = str(policy or "manual").lower().strip()
        return value if value in {"manual", "timeout"} else "manual"

    def _session_payload(self) -> dict:
        return {
            "active": bool(self.session.active),
            "camera_ids": list(self.session.camera_ids),
            "object_points": self._to_list_points(self.session.object_points),
            "image_points": self._to_list_points(self.session.image_points),
            "image_sizes": {cid: [int(size[0]), int(size[1])] for cid, size in self.session.image_sizes.items()},
            "captures": int(self.session.captures),
            "report": dict(self.session.report or {}),
            "failure_streak": int(self.session.failure_streak),
            "recommended_fps_cap": self.session.recommended_fps_cap,
            "sync_skew_history_ms": [float(v) for v in self.session.sync_skew_history_ms],
            "last_accept_ts_ms": self.session.last_accept_ts_ms,
            "last_accept_pose": self.session.last_accept_pose,
            "stable_since_ts_ms": self.session.stable_since_ts_ms,
            "last_readiness_metrics": dict(self.session.last_readiness_metrics or {}),
            "last_pose_sample": self.session.last_pose_sample,
            "resume_pending": bool(self.session.resume_pending),
            "resume_policy": str(self.session.resume_policy),
            "resume_deadline_ts_ms": self.session.resume_deadline_ts_ms,
            "resume_snapshot_ts_ms": self.session.resume_snapshot_ts_ms,
            "resume_auto_reset": bool(self.session.resume_auto_reset),
            "resume_reason": self.session.resume_reason,
        }

    def _save_session_snapshot(self) -> None:
        if self._session_state_path is None:
            return
        if not self.session.active and not self.session.resume_pending:
            save_json_atomic(self._session_state_path, {"active": False})
            return
        payload = self._session_payload()
        save_json_atomic(self._session_state_path, payload)

    def _restore_session_snapshot(self) -> None:
        if self._session_state_path is None:
            return
        payload = load_json(self._session_state_path, default={"active": False})
        if not isinstance(payload, dict) or not bool(payload.get("active", False)):
            return
        now_ms = time.time() * 1000.0
        policy = self._sanitize_resume_policy(self.cfg.calibration.resume_policy)
        deadline = (
            now_ms + (1000.0 * float(self.cfg.calibration.resume_timeout_s))
            if policy == "timeout"
            else None
        )
        self.session = CalibrationSession(
            active=True,
            camera_ids=[str(v) for v in payload.get("camera_ids", [])],
            object_points=self._from_list_points(payload.get("object_points", {}), np.float32),
            image_points=self._from_list_points(payload.get("image_points", {}), np.float32),
            image_sizes={
                str(cid): (int(size[0]), int(size[1]))
                for cid, size in (payload.get("image_sizes", {}) or {}).items()
                if isinstance(size, (list, tuple)) and len(size) == 2
            },
            captures=int(payload.get("captures", 0)),
            report=dict(payload.get("report", {}) or {}),
            failure_streak=int(payload.get("failure_streak", 0)),
            recommended_fps_cap=payload.get("recommended_fps_cap"),
            sync_skew_history_ms=[float(v) for v in payload.get("sync_skew_history_ms", [])],
            last_accept_ts_ms=payload.get("last_accept_ts_ms"),
            last_accept_pose=payload.get("last_accept_pose"),
            stable_since_ts_ms=payload.get("stable_since_ts_ms"),
            last_readiness_metrics=dict(payload.get("last_readiness_metrics", {}) or {}),
            last_pose_sample=payload.get("last_pose_sample"),
            resume_pending=True,
            resume_policy=policy,
            resume_deadline_ts_ms=deadline,
            resume_snapshot_ts_ms=now_ms,
            resume_auto_reset=False,
            resume_reason="server_restart",
        )
        self._save_session_snapshot()

    def _resume_connectivity_status(self) -> dict:
        camera_ids = list(self.session.camera_ids)
        diag = self.capture_hub.get_frame_diagnostics(
            camera_ids,
            max(1, int(self.cfg.runtime.max_latency_ms)),
        )
        per_camera = diag.get("per_camera", {})
        connected_ids = [cid for cid in camera_ids if bool(per_camera.get(cid, {}).get("connected", False))]
        missing_ids = [cid for cid in camera_ids if cid not in connected_ids]
        return {
            "connected_ids": connected_ids,
            "missing_ids": missing_ids,
            "all_connected": len(missing_ids) == 0 and len(camera_ids) > 0,
            "per_camera": per_camera,
        }

    def _activate_resumed_session(self, reason: str) -> None:
        self.session.resume_pending = False
        self.session.resume_deadline_ts_ms = None
        self.session.resume_auto_reset = False
        self.session.resume_reason = reason
        self._save_session_snapshot()

    def _reset_resumed_session(self, reason: str) -> None:
        self.session = CalibrationSession(
            active=False,
            resume_pending=False,
            resume_policy=self._sanitize_resume_policy(self.cfg.calibration.resume_policy),
            resume_reason=reason,
        )
        self._save_session_snapshot()

    def _apply_resume_policy(self) -> None:
        if not self.session.resume_pending:
            return
        status = self._resume_connectivity_status()
        if self.session.resume_policy == "timeout":
            now_ms = time.time() * 1000.0
            if status["all_connected"]:
                self._activate_resumed_session("auto_resumed")
                return
            deadline = self.session.resume_deadline_ts_ms
            if deadline is not None and now_ms >= float(deadline):
                self.session.resume_auto_reset = True
                self._reset_resumed_session("resume_timeout_reset")

    def resume_status(self) -> Dict:
        if not self.session.resume_pending:
            return {
                "resume_pending": False,
                "policy": self._sanitize_resume_policy(self.cfg.calibration.resume_policy),
                "timeout_s": int(self.cfg.calibration.resume_timeout_s),
            }
        self._apply_resume_policy()
        if not self.session.resume_pending:
            return {
                "resume_pending": False,
                "policy": self._sanitize_resume_policy(self.cfg.calibration.resume_policy),
                "timeout_s": int(self.cfg.calibration.resume_timeout_s),
                "resolved_reason": self.session.resume_reason,
            }
        status = self._resume_connectivity_status()
        now_ms = time.time() * 1000.0
        remaining_ms = None
        if self.session.resume_policy == "timeout" and self.session.resume_deadline_ts_ms is not None:
            remaining_ms = max(0, int(float(self.session.resume_deadline_ts_ms) - now_ms))
        return {
            "resume_pending": True,
            "policy": self.session.resume_policy,
            "timeout_s": int(self.cfg.calibration.resume_timeout_s),
            "deadline_ts_ms": self.session.resume_deadline_ts_ms,
            "remaining_ms": remaining_ms,
            "camera_ids": list(self.session.camera_ids),
            "connected_ids": status["connected_ids"],
            "missing_ids": status["missing_ids"],
            "all_connected": bool(status["all_connected"]),
            "captures": int(self.session.captures),
            "reason": self.session.resume_reason,
        }

    def resume_continue(self) -> Dict:
        self._apply_resume_policy()
        if not self.session.resume_pending:
            return {"ok": False, "resume_pending": False, "reason": "no_resume_pending"}
        status = self._resume_connectivity_status()
        if not status["all_connected"]:
            return {
                "ok": False,
                "resume_pending": True,
                "reason": "not_all_cameras_reconnected",
                "missing_ids": status["missing_ids"],
            }
        self._activate_resumed_session("manual_resumed")
        return {
            "ok": True,
            "resume_pending": False,
            "reason": "manual_resumed",
            "captures": int(self.session.captures),
            "camera_ids": list(self.session.camera_ids),
        }

    def resume_reset(self) -> Dict:
        self._reset_resumed_session("manual_reset")
        return {"ok": True, "resume_pending": False, "reason": "manual_reset"}

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
        policy = self._sanitize_resume_policy(self.cfg.calibration.resume_policy)
        self.session = CalibrationSession(
            active=True,
            camera_ids=list(camera_ids),
            object_points={cid: [] for cid in camera_ids},
            image_points={cid: [] for cid in camera_ids},
            report={},
            recommended_fps_cap=self.cfg.ingest.client_fps_cap,
            resume_pending=False,
            resume_policy=policy,
            resume_deadline_ts_ms=None,
            resume_snapshot_ts_ms=time.time() * 1000.0,
            resume_auto_reset=False,
            resume_reason="started",
        )
        self._save_session_snapshot()
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

    def _extract_pose_metrics(self, corners: np.ndarray, frame_shape: Tuple[int, int, int]) -> Dict[str, float]:
        h, w = frame_shape[:2]
        corners_2d = corners.reshape(-1, 2).astype(np.float32)
        centroid = np.mean(corners_2d, axis=0)
        hull = cv2.convexHull(corners_2d)
        area = float(cv2.contourArea(hull))
        frame_area = max(1.0, float(w * h))
        return {
            "cx": float(centroid[0]) / float(max(1, w)),
            "cy": float(centroid[1]) / float(max(1, h)),
            "area": area / frame_area,
        }

    def _aggregate_pose_metrics(self, poses_by_cam: Dict[str, Dict[str, float]]) -> Dict[str, float] | None:
        if not poses_by_cam:
            return None
        cx_vals = [float(p["cx"]) for p in poses_by_cam.values()]
        cy_vals = [float(p["cy"]) for p in poses_by_cam.values()]
        area_vals = [float(p["area"]) for p in poses_by_cam.values()]
        return {
            "cx": float(np.median(cx_vals)),
            "cy": float(np.median(cy_vals)),
            "area": float(np.median(area_vals)),
        }

    def _pose_distance(self, current: Dict[str, float] | None, previous: Dict[str, float] | None) -> float | None:
        if not current or not previous:
            return None
        dx = float(current["cx"] - previous["cx"])
        dy = float(current["cy"] - previous["cy"])
        da = abs(float(current["area"] - previous["area"]))
        return float(np.hypot(dx, dy) + (0.5 * da))

    def _update_stability(self, pose: Dict[str, float] | None, now_ms: float) -> float:
        stable_threshold = float(self.cfg.calibration.auto_stable_threshold_norm)
        if pose is None:
            self.session.stable_since_ts_ms = None
            self.session.last_pose_sample = None
            return 0.0
        sample_delta = self._pose_distance(pose, self.session.last_pose_sample)
        if self.session.stable_since_ts_ms is None:
            self.session.stable_since_ts_ms = now_ms
        elif sample_delta is not None and sample_delta > stable_threshold:
            self.session.stable_since_ts_ms = now_ms
        self.session.last_pose_sample = dict(pose)
        return max(0.0, float(now_ms - float(self.session.stable_since_ts_ms or now_ms)))

    def _evaluate_capture_gate(self, readiness_metrics: Dict[str, Any], now_ms: float, mode: str) -> Tuple[bool, str | None]:
        mode = (mode or "manual").lower()
        if not readiness_metrics.get("all_cameras_ready", False):
            return False, "not_ready"
        board_metrics = readiness_metrics.get("board_metrics", {}) or {}
        if not board_metrics.get("quality_ok", False):
            return False, "low_quality"
        if mode == "manual":
            return True, None

        min_interval_ms = int(self.cfg.calibration.auto_min_interval_ms)
        if self.session.last_accept_ts_ms is not None and (now_ms - self.session.last_accept_ts_ms) < float(min_interval_ms):
            return False, "min_interval"

        pose_delta = board_metrics.get("pose_delta")
        if self.session.last_accept_pose is not None:
            if pose_delta is None or float(pose_delta) < float(self.cfg.calibration.auto_motion_threshold_norm):
                return False, "insufficient_motion"
            if float(pose_delta) < float(self.cfg.calibration.auto_pose_delta_threshold):
                return False, "duplicate_pose"

        stable_ms = float(board_metrics.get("stable_ms", 0.0))
        if stable_ms < float(self.cfg.calibration.auto_hold_ms):
            return False, "not_stable"

        return True, None

    def readiness(self) -> Dict:
        self._apply_resume_policy()
        if not self.session.active:
            resume = self.resume_status()
            return {
                "active": False,
                "camera_ids": [],
                "per_camera": {},
                "all_cameras_ready": False,
                "effective_latency_ms": self._compute_effective_latency_ms(),
                "sync_skew_ms": 0.0,
                "recommended_fps_cap": self.session.recommended_fps_cap,
                "board_metrics": {
                    "centroid_xy_norm": None,
                    "board_area_norm": None,
                    "quality_ok": False,
                    "pose_delta": None,
                    "stable_ms": 0.0,
                    "board_area_norm_by_camera": {},
                    "board_quality_ok_by_camera": {},
                },
                "capture_block_reason": "session_inactive",
                "resume": resume,
            }

        camera_ids = self.session.camera_ids
        effective_latency_ms = self._compute_effective_latency_ms()
        if self.session.resume_pending:
            frame_diag_pending = self.capture_hub.get_frame_diagnostics(camera_ids, effective_latency_ms)
            resume = self.resume_status()
            return {
                "active": True,
                "camera_ids": camera_ids,
                "per_camera": frame_diag_pending["per_camera"],
                "all_cameras_ready": False,
                "effective_latency_ms": effective_latency_ms,
                "sync_skew_ms": frame_diag_pending["sync_skew_ms"],
                "recommended_fps_cap": self.session.recommended_fps_cap,
                "captures": self.session.captures,
                "required": self.cfg.calibration.min_captures,
                "board_metrics": {
                    "centroid_xy_norm": None,
                    "board_area_norm": None,
                    "quality_ok": False,
                    "pose_delta": None,
                    "stable_ms": 0.0,
                    "board_area_norm_by_camera": {},
                    "board_quality_ok_by_camera": {},
                },
                "capture_block_reason": "resume_pending",
                "resume": resume,
            }

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
        min_area_norm = float(self.cfg.calibration.auto_min_board_area_norm)
        poses_by_cam: Dict[str, Dict[str, float]] = {}
        for cid in camera_ids:
            frame_packet = latest_frames.get(cid)
            checkerboard_detected = False
            pose = None
            if frame_packet is not None:
                found, corners = detect_results.get(cid, (False, None))
                checkerboard_detected = bool(found and corners is not None)
                if checkerboard_detected:
                    pose = self._extract_pose_metrics(corners, frame_packet.frame.shape)
                    poses_by_cam[cid] = pose
            per_camera[cid]["checkerboard_detected"] = checkerboard_detected
            per_camera[cid]["board_area_norm"] = None if pose is None else float(pose["area"])
            per_camera[cid]["board_centroid_xy_norm"] = (
                None if pose is None else [float(pose["cx"]), float(pose["cy"])]
            )
            per_camera[cid]["board_quality_ok"] = bool(
                pose is not None and float(pose["area"]) >= min_area_norm
            )

        all_ready = all(
            per_camera[cid]["connected"]
            and per_camera[cid]["in_sync"]
            and per_camera[cid]["checkerboard_detected"]
            for cid in camera_ids
        )
        now_ms = time.time() * 1000.0
        aggregate_pose = self._aggregate_pose_metrics(poses_by_cam)
        stable_ms = self._update_stability(aggregate_pose, now_ms)
        pose_delta = self._pose_distance(aggregate_pose, self.session.last_accept_pose)
        board_area = float(aggregate_pose["area"]) if aggregate_pose else None
        quality_ok = bool(
            camera_ids
            and all(
                bool(per_camera[cid]["checkerboard_detected"] and per_camera[cid]["board_quality_ok"])
                for cid in camera_ids
            )
        )
        board_metrics = {
            "centroid_xy_norm": None
            if aggregate_pose is None
            else [float(aggregate_pose["cx"]), float(aggregate_pose["cy"])],
            "board_area_norm": board_area,
            "quality_ok": quality_ok,
            "pose_delta": pose_delta,
            "stable_ms": stable_ms,
            "board_area_norm_by_camera": {
                cid: per_camera[cid]["board_area_norm"] for cid in camera_ids
            },
            "board_quality_ok_by_camera": {
                cid: bool(per_camera[cid]["board_quality_ok"]) for cid in camera_ids
            },
        }
        readiness_metrics = {
            "all_cameras_ready": all_ready,
            "board_metrics": board_metrics,
        }
        _, capture_block_reason = self._evaluate_capture_gate(readiness_metrics, now_ms, mode="auto")
        self.session.last_readiness_metrics = {
            **readiness_metrics,
            "capture_block_reason": capture_block_reason,
            "timestamp_ms": now_ms,
        }
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
            "board_metrics": board_metrics,
            "capture_block_reason": capture_block_reason,
            "resume": self.resume_status(),
        }

    def capture(self, mode: str = "manual") -> Dict:
        mode = (mode or "manual").lower()
        if mode not in {"manual", "auto"}:
            mode = "manual"
        self._apply_resume_policy()
        if not self.session.active:
            raise RuntimeError("Calibration session is not active.")
        if self.session.resume_pending:
            resume = self.resume_status()
            return {
                "ok": False,
                "accepted": False,
                "capture_mode": mode,
                "rejection_reason": "resume_pending",
                "reason": "resume_pending",
                "captures": self.session.captures,
                "required": self.cfg.calibration.min_captures,
                "resume": resume,
            }
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
            self._save_session_snapshot()
            return {
                "ok": False,
                "accepted": False,
                "capture_mode": mode,
                "rejection_reason": "not_ready",
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
        per_camera = frame_diag["per_camera"]
        min_area_norm = float(self.cfg.calibration.auto_min_board_area_norm)
        poses_by_cam: Dict[str, Dict[str, float]] = {}
        for cid in self.session.camera_ids:
            frame = frames[cid].frame
            found, corners = detect_results.get(cid, (False, None))
            per_camera[cid]["checkerboard_detected"] = bool(found and corners is not None)
            per_camera[cid]["board_area_norm"] = None
            per_camera[cid]["board_centroid_xy_norm"] = None
            per_camera[cid]["board_quality_ok"] = False
            if not found or corners is None:
                self.session.failure_streak = 0
                self._save_session_snapshot()
                return {
                    "ok": False,
                    "accepted": False,
                    "capture_mode": mode,
                    "rejection_reason": "not_ready",
                    "reason": f"checkerboard_not_found:{cid}",
                    "effective_latency_ms": effective_latency_ms,
                    "sync_skew_ms": frame_diag["sync_skew_ms"],
                    "recommended_fps_cap": self.session.recommended_fps_cap,
                    "per_camera": per_camera,
                }
            corners_by_cam[cid] = corners.astype(np.float32)
            self.session.image_sizes[cid] = (frame.shape[1], frame.shape[0])
            pose = self._extract_pose_metrics(corners_by_cam[cid], frame.shape)
            poses_by_cam[cid] = pose
            per_camera[cid]["board_area_norm"] = float(pose["area"])
            per_camera[cid]["board_centroid_xy_norm"] = [float(pose["cx"]), float(pose["cy"])]
            per_camera[cid]["board_quality_ok"] = bool(float(pose["area"]) >= min_area_norm)

        now_ms = time.time() * 1000.0
        aggregate_pose = self._aggregate_pose_metrics(poses_by_cam)
        stable_ms = self._update_stability(aggregate_pose, now_ms)
        pose_delta = self._pose_distance(aggregate_pose, self.session.last_accept_pose)
        board_area = float(aggregate_pose["area"]) if aggregate_pose else None
        quality_ok = bool(
            self.session.camera_ids
            and all(
                bool(per_camera[cid]["checkerboard_detected"] and per_camera[cid]["board_quality_ok"])
                for cid in self.session.camera_ids
            )
        )
        board_metrics = {
            "centroid_xy_norm": None
            if aggregate_pose is None
            else [float(aggregate_pose["cx"]), float(aggregate_pose["cy"])],
            "board_area_norm": board_area,
            "quality_ok": quality_ok,
            "pose_delta": pose_delta,
            "stable_ms": stable_ms,
            "board_area_norm_by_camera": {
                cid: per_camera[cid]["board_area_norm"] for cid in self.session.camera_ids
            },
            "board_quality_ok_by_camera": {
                cid: bool(per_camera[cid]["board_quality_ok"]) for cid in self.session.camera_ids
            },
        }
        readiness_metrics = {
            "all_cameras_ready": True,
            "board_metrics": board_metrics,
        }
        accepted, rejection_reason = self._evaluate_capture_gate(readiness_metrics, now_ms, mode=mode)
        self.session.last_readiness_metrics = {
            **readiness_metrics,
            "capture_block_reason": rejection_reason,
            "timestamp_ms": now_ms,
        }
        if not accepted:
            self.session.failure_streak = 0
            self._save_session_snapshot()
            return {
                "ok": False,
                "accepted": False,
                "capture_mode": mode,
                "rejection_reason": rejection_reason,
                "reason": "capture_gate_blocked",
                "captures": self.session.captures,
                "required": self.cfg.calibration.min_captures,
                "effective_latency_ms": effective_latency_ms,
                "sync_skew_ms": frame_diag["sync_skew_ms"],
                "recommended_fps_cap": self.session.recommended_fps_cap,
                "per_camera": per_camera,
                "board_metrics": board_metrics,
            }

        for cid in self.session.camera_ids:
            self.session.object_points[cid].append(objp.copy())
            self.session.image_points[cid].append(corners_by_cam[cid])

        self.session.captures += 1
        self.session.failure_streak = 0
        self.session.last_accept_ts_ms = now_ms
        self.session.last_accept_pose = None if aggregate_pose is None else dict(aggregate_pose)
        self._save_session_snapshot()
        return {
            "ok": True,
            "accepted": True,
            "capture_mode": mode,
            "rejection_reason": None,
            "captures": self.session.captures,
            "required": self.cfg.calibration.min_captures,
            "effective_latency_ms": effective_latency_ms,
            "sync_skew_ms": frame_diag["sync_skew_ms"],
            "recommended_fps_cap": self.session.recommended_fps_cap,
            "per_camera": per_camera,
            "board_metrics": board_metrics,
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

        ref_id = sorted(camera_ids)[0]
        store.extrinsics[ref_id] = (np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64))

        pair_rms = {}
        for cid in sorted(cid for cid in camera_ids if cid != ref_id):
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
        self._save_session_snapshot()
        return self.session.report

    def report(self) -> Dict:
        self._apply_resume_policy()
        return {
            "active": self.session.active,
            "camera_ids": self.session.camera_ids,
            "captures": self.session.captures,
            "recommended_fps_cap": self.session.recommended_fps_cap,
            "result": self.session.report,
            "resume": self.resume_status(),
        }
