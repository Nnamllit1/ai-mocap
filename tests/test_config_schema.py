import unittest

from app.models.config import AppConfig


class ConfigSchemaTests(unittest.TestCase):
    def test_defaults_load(self):
        cfg = AppConfig()
        self.assertEqual(cfg.triangulation.min_views, 2)
        self.assertTrue(cfg.triangulation.allow_single_view_fallback)
        self.assertGreater(cfg.runtime.max_joint_jump_m, 0.0)
        self.assertFalse(hasattr(cfg.calibration, "auto_primary_camera_strategy"))
        self.assertEqual(cfg.calibration.resume_policy, "manual")
        self.assertGreaterEqual(cfg.calibration.resume_timeout_s, 0)
        self.assertTrue(cfg.persistence.invites_path)
        self.assertTrue(cfg.persistence.camera_registry_path)
        self.assertTrue(cfg.persistence.calibration_session_path)
        self.assertTrue(cfg.server.token)


if __name__ == "__main__":
    unittest.main()
