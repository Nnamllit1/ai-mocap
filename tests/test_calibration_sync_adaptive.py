import unittest

from app.core.calibration import CalibrationService
from app.core.capture import CaptureHub
from app.models.config import AppConfig, CalibrationSyncConfig, RuntimeConfig


class CalibrationAdaptiveSyncTests(unittest.TestCase):
    def _cfg(self):
        return AppConfig(
            runtime=RuntimeConfig(max_latency_ms=120),
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


if __name__ == "__main__":
    unittest.main()
