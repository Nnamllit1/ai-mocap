import unittest

try:
    from fastapi.testclient import TestClient
except Exception:  # noqa: BLE001
    TestClient = None

from app.main import create_app


@unittest.skipUnless(TestClient is not None, "fastapi testclient/httpx not installed")
class CalibrationReadinessApiTests(unittest.TestCase):
    def test_readiness_requires_token(self):
        app = create_app()
        client = TestClient(app)
        response = client.get("/api/calibration/readiness")
        self.assertEqual(response.status_code, 401)

    def test_readiness_shape_with_token(self):
        app = create_app()
        client = TestClient(app)
        token = app.state.runtime.config_store.config.server.token
        response = client.get(
            "/api/calibration/readiness",
            headers={"x-access-token": token},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("active", payload)
        self.assertIn("all_cameras_ready", payload)
        self.assertIn("effective_latency_ms", payload)
        self.assertIn("sync_skew_ms", payload)
        self.assertIn("recommended_fps_cap", payload)
        self.assertIn("board_metrics", payload)
        self.assertIn("capture_block_reason", payload)
        self.assertIn("quality_ok", payload["board_metrics"])
        self.assertIn("pose_delta", payload["board_metrics"])
        self.assertIn("stable_ms", payload["board_metrics"])

    def test_capture_endpoint_accepts_default_and_manual_mode_payload(self):
        app = create_app()
        client = TestClient(app)
        token = app.state.runtime.config_store.config.server.token
        response_default = client.post(
            "/api/calibration/capture",
            headers={"x-access-token": token},
        )
        self.assertEqual(response_default.status_code, 400)
        self.assertIn("Calibration session is not active.", response_default.json().get("detail", ""))

        response_manual = client.post(
            "/api/calibration/capture",
            headers={"x-access-token": token},
            json={"mode": "manual"},
        )
        self.assertEqual(response_manual.status_code, 400)
        self.assertIn("Calibration session is not active.", response_manual.json().get("detail", ""))


if __name__ == "__main__":
    unittest.main()
