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


if __name__ == "__main__":
    unittest.main()
