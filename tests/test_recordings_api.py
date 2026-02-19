import unittest

try:
    from fastapi.testclient import TestClient
except Exception:  # noqa: BLE001
    TestClient = None

try:
    import cv2  # noqa: F401
except Exception:  # noqa: BLE001
    cv2 = None


@unittest.skipUnless(TestClient is not None, "fastapi testclient/httpx not installed")
@unittest.skipUnless(cv2 is not None, "opencv-python not installed")
class RecordingsApiTests(unittest.TestCase):
    def setUp(self):
        from app.main import create_app

        self.app = create_app()
        self.client = TestClient(self.app)
        self.token = self.app.state.runtime.config_store.config.server.token
        self.headers = {"x-access-token": self.token}

    def test_recordings_status_requires_token(self):
        response = self.client.get("/api/recordings/status")
        self.assertEqual(response.status_code, 401)

    def test_start_stop_list_export_happy_path(self):
        start_resp = self.client.post("/api/recordings/start", headers=self.headers)
        self.assertEqual(start_resp.status_code, 200)
        clip_id = start_resp.json()["clip"]["clip_id"]

        stop_resp = self.client.post("/api/recordings/stop", headers=self.headers)
        self.assertEqual(stop_resp.status_code, 200)
        self.assertEqual(stop_resp.json()["clip"]["clip_id"], clip_id)

        list_resp = self.client.get("/api/recordings", headers=self.headers)
        self.assertEqual(list_resp.status_code, 200)
        self.assertTrue(any(item["clip_id"] == clip_id for item in list_resp.json()))

        export_resp = self.client.post(
            f"/api/recordings/{clip_id}/export",
            headers=self.headers,
        )
        self.assertEqual(export_resp.status_code, 200)
        payload = export_resp.json()
        self.assertIn("json", payload["paths"])
        self.assertIn("csv", payload["paths"])

    def test_export_missing_clip_404(self):
        response = self.client.post(
            "/api/recordings/clip_missing/export",
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "clip_not_found")

    def test_export_active_clip_400(self):
        start_resp = self.client.post("/api/recordings/start", headers=self.headers)
        clip_id = start_resp.json()["clip"]["clip_id"]
        export_resp = self.client.post(
            f"/api/recordings/{clip_id}/export",
            headers=self.headers,
        )
        self.assertEqual(export_resp.status_code, 400)
        self.assertEqual(export_resp.json()["detail"], "clip_still_recording")
        self.client.post("/api/recordings/stop", headers=self.headers)


if __name__ == "__main__":
    unittest.main()
