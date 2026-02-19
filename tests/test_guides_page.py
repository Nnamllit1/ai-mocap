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
class GuidesPageTests(unittest.TestCase):
    def setUp(self):
        from app.main import create_app

        self.app = create_app()
        self.client = TestClient(self.app)
        self.token = self.app.state.runtime.config_store.config.server.token

    def test_guides_requires_auth_redirect(self):
        response = self.client.get("/guides", follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers.get("location"), "/login")

    def test_guides_renders_sections_with_auth_cookie(self):
        self.client.cookies.set("portal_token", self.token)
        response = self.client.get("/guides")
        self.assertEqual(response.status_code, 200)
        body = response.text
        self.assertIn("Setup Guide", body)
        self.assertIn("Calibration Guide", body)
        self.assertIn("/guides", body)


if __name__ == "__main__":
    unittest.main()
