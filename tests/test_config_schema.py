import unittest

from app.models.config import AppConfig


class ConfigSchemaTests(unittest.TestCase):
    def test_defaults_load(self):
        cfg = AppConfig()
        self.assertEqual(cfg.triangulation.min_views, 2)
        self.assertTrue(cfg.server.token)


if __name__ == "__main__":
    unittest.main()
