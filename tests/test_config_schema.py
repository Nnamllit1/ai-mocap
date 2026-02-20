import unittest

from app.models.config import AppConfig


class ConfigSchemaTests(unittest.TestCase):
    def test_defaults_load(self):
        cfg = AppConfig()
        self.assertEqual(cfg.triangulation.min_views, 2)
        self.assertTrue(cfg.triangulation.allow_single_view_fallback)
        self.assertGreater(cfg.runtime.max_joint_jump_m, 0.0)
        self.assertTrue(cfg.runtime.bone_length_guard_enabled)
        self.assertGreaterEqual(cfg.runtime.bone_length_soft_rel_tol, 0.0)
        self.assertGreaterEqual(
            cfg.runtime.bone_length_hard_rel_tol,
            cfg.runtime.bone_length_soft_rel_tol,
        )
        self.assertGreaterEqual(cfg.runtime.bone_length_ema_alpha, 0.0)
        self.assertLessEqual(cfg.runtime.bone_length_ema_alpha, 1.0)
        self.assertGreaterEqual(cfg.runtime.bone_length_learn_conf, 0.0)
        self.assertLessEqual(cfg.runtime.bone_length_learn_conf, 1.0)
        self.assertFalse(hasattr(cfg.calibration, "auto_primary_camera_strategy"))
        self.assertEqual(cfg.calibration.resume_policy, "manual")
        self.assertGreaterEqual(cfg.calibration.resume_timeout_s, 0)
        self.assertTrue(cfg.persistence.invites_path)
        self.assertTrue(cfg.persistence.camera_registry_path)
        self.assertTrue(cfg.persistence.calibration_session_path)
        self.assertTrue(cfg.server.token)


if __name__ == "__main__":
    unittest.main()
