import unittest
from pathlib import Path
import tempfile
import time

import numpy as np

from app.core.calibration import CalibrationService
from app.core.capture import CameraFrame, CaptureHub
from app.models.config import AppConfig, CalibrationConfig, CalibrationSyncConfig, RuntimeConfig


class CalibrationAdaptiveSyncTests(unittest.TestCase):
    def _cfg(self, *, resume_policy: str = "manual", resume_timeout_s: int = 90):
        return AppConfig(
            runtime=RuntimeConfig(max_latency_ms=120),
            calibration=CalibrationConfig(
                resume_policy=resume_policy,
                resume_timeout_s=resume_timeout_s,
            ),
            calibration_sync=CalibrationSyncConfig(
                adaptive_enabled=True,
                min_latency_ms=120,
                max_latency_ms=450,
                jitter_factor=2.0,
                history_size=60,
                fps_downshift_failure_streak=3,
            ),
        )

    def test_effective_window_grows_with_skew_and_clamps(self):
        service = CalibrationService(self._cfg(), CaptureHub(heartbeat_timeout_s=6.0))
        service.start(["cam_a", "cam_b"])
        service.session.sync_skew_history_ms = [30.0, 40.0, 50.0, 300.0]
        effective = service._compute_effective_latency_ms()
        self.assertGreaterEqual(effective, 120)
        self.assertLessEqual(effective, 450)

        service.session.sync_skew_history_ms = [1000.0] * 100
        effective2 = service._compute_effective_latency_ms()
        self.assertEqual(effective2, 450)

    def test_fps_downshift_after_failure_streak(self):
        service = CalibrationService(self._cfg(), CaptureHub(heartbeat_timeout_s=6.0))
        service.start(["cam_a", "cam_b"])
        self.assertEqual(service.session.recommended_fps_cap, 20)
        service.session.failure_streak = 3
        service._maybe_downshift_fps()
        self.assertEqual(service.session.recommended_fps_cap, 15)

    def test_capture_returns_sync_diagnostics_on_not_ready(self):
        service = CalibrationService(self._cfg(), CaptureHub(heartbeat_timeout_s=6.0))
        service.start(["cam_a", "cam_b"])
        out = service.capture()
        self.assertFalse(out["ok"])
        self.assertEqual(out["reason"], "not_all_cameras_ready")
        self.assertIn("effective_latency_ms", out)
        self.assertIn("sync_skew_ms", out)
        self.assertIn("per_camera", out)

    def _make_corners(self, x0: float, y0: float, step: float = 2.0) -> np.ndarray:
        points = []
        for y in range(6):
            for x in range(9):
                points.append([x0 + (x * step), y0 + (y * step)])
        return np.array(points, dtype=np.float32).reshape(-1, 1, 2)

    def _ready_service(self):
        service = CalibrationService(self._cfg(), CaptureHub(heartbeat_timeout_s=6.0))
        service.start(["cam_a", "cam_b"])
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        now = time.time()
        frames = {
            "cam_a": CameraFrame("cam_a", now, frame, 1),
            "cam_b": CameraFrame("cam_b", now, frame, 1),
        }
        service.capture_hub.get_synced_frames = lambda min_sources, max_latency_ms: frames
        service.capture_hub.get_frame_diagnostics = lambda camera_ids, max_latency_ms: {
            "sync_skew_ms": 0.0,
            "per_camera": {
                cid: {"connected": True, "in_sync": True, "latest_frame_age_ms": 10.0, "seq": 1}
                for cid in camera_ids
            },
        }
        return service, frames

    def test_auto_capture_rejects_when_motion_below_threshold(self):
        service, _ = self._ready_service()
        corners = self._make_corners(20, 20)
        service._detect_many = lambda frames_by_cam, board_size: {cid: (True, corners) for cid in frames_by_cam}
        pose = service._extract_pose_metrics(corners, (100, 100, 3))
        service.session.last_accept_pose = dict(pose)
        service.session.last_pose_sample = dict(pose)
        service.session.stable_since_ts_ms = (time.time() * 1000.0) - 1000.0
        service.session.last_accept_ts_ms = (time.time() * 1000.0) - 5000.0

        out = service.capture(mode="auto")
        self.assertFalse(out["ok"])
        self.assertFalse(out["accepted"])
        self.assertEqual(out["capture_mode"], "auto")
        self.assertEqual(out["rejection_reason"], "insufficient_motion")

    def test_auto_capture_accepts_after_movement_and_hold(self):
        service, _ = self._ready_service()
        corners = self._make_corners(60, 60)
        service._detect_many = lambda frames_by_cam, board_size: {cid: (True, corners) for cid in frames_by_cam}
        pose = service._extract_pose_metrics(corners, (100, 100, 3))
        service.session.last_accept_pose = {"cx": 0.1, "cy": 0.1, "area": 0.02}
        service.session.last_pose_sample = dict(pose)
        service.session.stable_since_ts_ms = (time.time() * 1000.0) - 1000.0
        service.session.last_accept_ts_ms = (time.time() * 1000.0) - 5000.0

        out = service.capture(mode="auto")
        self.assertTrue(out["ok"])
        self.assertTrue(out["accepted"])
        self.assertEqual(out["capture_mode"], "auto")
        self.assertIsNone(out["rejection_reason"])
        self.assertEqual(out["captures"], 1)

    def test_auto_capture_enforces_min_interval(self):
        service, _ = self._ready_service()
        corners = self._make_corners(60, 60)
        service._detect_many = lambda frames_by_cam, board_size: {cid: (True, corners) for cid in frames_by_cam}
        pose = service._extract_pose_metrics(corners, (100, 100, 3))
        service.session.last_accept_pose = {"cx": 0.1, "cy": 0.1, "area": 0.02}
        service.session.last_pose_sample = dict(pose)
        service.session.stable_since_ts_ms = (time.time() * 1000.0) - 1000.0
        service.session.last_accept_ts_ms = (time.time() * 1000.0) - 100.0

        out = service.capture(mode="auto")
        self.assertFalse(out["ok"])
        self.assertFalse(out["accepted"])
        self.assertEqual(out["capture_mode"], "auto")
        self.assertEqual(out["rejection_reason"], "min_interval")

    def test_readiness_quality_uses_all_cameras(self):
        service, _ = self._ready_service()
        corners_big = self._make_corners(40, 40, step=2.0)
        corners_small = self._make_corners(2, 2, step=0.2)
        service._detect_many = lambda frames_by_cam, board_size: {
            "cam_a": (True, corners_big),
            "cam_b": (True, corners_small),
        }

        readiness = service.readiness()
        self.assertTrue(readiness["all_cameras_ready"])
        self.assertFalse(readiness["board_metrics"]["quality_ok"])
        self.assertIn("board_quality_ok_by_camera", readiness["board_metrics"])
        self.assertTrue(readiness["board_metrics"]["board_quality_ok_by_camera"]["cam_a"])
        self.assertFalse(readiness["board_metrics"]["board_quality_ok_by_camera"]["cam_b"])

    def test_aggregate_pose_metrics_is_order_independent(self):
        service = CalibrationService(self._cfg(), CaptureHub(heartbeat_timeout_s=6.0))
        pose_a = {"cx": 0.2, "cy": 0.3, "area": 0.02}
        pose_b = {"cx": 0.6, "cy": 0.7, "area": 0.04}
        agg1 = service._aggregate_pose_metrics({"cam_a": pose_a, "cam_b": pose_b})
        agg2 = service._aggregate_pose_metrics({"cam_b": pose_b, "cam_a": pose_a})
        self.assertEqual(agg1, agg2)

    @staticmethod
    def _connected_diag(camera_ids):
        return {
            "sync_skew_ms": 0.0,
            "per_camera": {
                cid: {"connected": True, "in_sync": True, "latest_frame_age_ms": 10.0, "seq": 1}
                for cid in camera_ids
            },
        }

    @staticmethod
    def _disconnected_diag(camera_ids):
        return {
            "sync_skew_ms": 0.0,
            "per_camera": {
                cid: {"connected": False, "in_sync": False, "latest_frame_age_ms": None, "seq": 0}
                for cid in camera_ids
            },
        }

    def test_calibration_snapshot_restores_as_resume_pending_after_restart(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "calibration_session.json"
            service = CalibrationService(
                self._cfg(),
                CaptureHub(heartbeat_timeout_s=6.0),
                session_state_path=state_path,
            )
            service.start(["cam_a", "cam_b"])
            restarted = CalibrationService(
                self._cfg(),
                CaptureHub(heartbeat_timeout_s=6.0),
                session_state_path=state_path,
            )
            status = restarted.resume_status()
            self.assertTrue(restarted.session.active)
            self.assertTrue(restarted.session.resume_pending)
            self.assertTrue(status["resume_pending"])
            self.assertEqual(sorted(status["camera_ids"]), ["cam_a", "cam_b"])

    def test_manual_resume_requires_explicit_continue(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "calibration_session.json"
            cfg = self._cfg(resume_policy="manual")
            service = CalibrationService(
                cfg,
                CaptureHub(heartbeat_timeout_s=6.0),
                session_state_path=state_path,
            )
            service.start(["cam_a", "cam_b"])

            hub = CaptureHub(heartbeat_timeout_s=6.0)
            restarted = CalibrationService(
                cfg,
                hub,
                session_state_path=state_path,
            )
            hub.get_frame_diagnostics = (
                lambda camera_ids, max_latency_ms: self._connected_diag(camera_ids)
            )

            readiness = restarted.readiness()
            self.assertTrue(restarted.session.resume_pending)
            self.assertEqual(readiness["capture_block_reason"], "resume_pending")

            resumed = restarted.resume_continue()
            self.assertTrue(resumed["ok"])
            self.assertFalse(restarted.session.resume_pending)

    def test_timeout_resume_auto_resumes_if_all_cameras_reconnect(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "calibration_session.json"
            cfg = self._cfg(resume_policy="timeout", resume_timeout_s=5)
            service = CalibrationService(
                cfg,
                CaptureHub(heartbeat_timeout_s=6.0),
                session_state_path=state_path,
            )
            service.start(["cam_a", "cam_b"])

            hub = CaptureHub(heartbeat_timeout_s=6.0)
            restarted = CalibrationService(
                cfg,
                hub,
                session_state_path=state_path,
            )
            hub.get_frame_diagnostics = (
                lambda camera_ids, max_latency_ms: self._connected_diag(camera_ids)
            )

            status = restarted.resume_status()
            self.assertFalse(status["resume_pending"])
            self.assertEqual(status.get("resolved_reason"), "auto_resumed")
            self.assertFalse(restarted.session.resume_pending)

    def test_timeout_resume_resets_if_cameras_do_not_reconnect(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "calibration_session.json"
            cfg = self._cfg(resume_policy="timeout", resume_timeout_s=0)
            service = CalibrationService(
                cfg,
                CaptureHub(heartbeat_timeout_s=6.0),
                session_state_path=state_path,
            )
            service.start(["cam_a", "cam_b"])

            hub = CaptureHub(heartbeat_timeout_s=6.0)
            restarted = CalibrationService(
                cfg,
                hub,
                session_state_path=state_path,
            )
            hub.get_frame_diagnostics = (
                lambda camera_ids, max_latency_ms: self._disconnected_diag(camera_ids)
            )

            status = restarted.resume_status()
            self.assertFalse(status["resume_pending"])
            self.assertEqual(status.get("resolved_reason"), "resume_timeout_reset")
            self.assertFalse(restarted.session.active)


if __name__ == "__main__":
    unittest.main()
