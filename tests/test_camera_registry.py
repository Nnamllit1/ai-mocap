from pathlib import Path
import tempfile
import unittest

from app.services.camera_registry import CameraRegistry


class CameraRegistryTests(unittest.TestCase):
    def test_register_and_reconnect(self):
        reg = CameraRegistry()
        first = reg.upsert_from_device(
            device_uid="d1",
            device_name="Pixel",
            platform="android",
            label="Phone A",
        )
        second = reg.upsert_from_device(
            device_uid="d1",
            device_name="Pixel v2",
            platform="android",
            label=None,
        )
        self.assertEqual(first.camera_id, second.camera_id)
        self.assertNotEqual(first.ws_token, second.ws_token)

    def test_registry_persists_and_connected_resets_on_restart(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "camera_registry.json"
            reg = CameraRegistry(state_path=path)
            first = reg.upsert_from_device(
                device_uid="d1",
                device_name="Pixel",
                platform="android",
                label="Phone A",
            )
            reg.set_connected(first.camera_id, True)
            reg.heartbeat(first.camera_id)

            restarted = CameraRegistry(state_path=path)
            loaded = restarted.get(first.camera_id)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.camera_id, first.camera_id)
            self.assertEqual(loaded.ws_token, first.ws_token)
            self.assertFalse(bool(loaded.connected))
            self.assertTrue(restarted.validate_ws_token(first.camera_id, first.ws_token))

            second = restarted.upsert_from_device(
                device_uid="d1",
                device_name="Pixel v2",
                platform="android",
                label=None,
            )
            self.assertEqual(second.camera_id, first.camera_id)
            self.assertNotEqual(second.ws_token, first.ws_token)

    def test_soft_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "camera_registry.json"
            reg = CameraRegistry(state_path=path)
            rec = reg.upsert_from_device(
                device_uid="d2",
                device_name="iPhone",
                platform="ios",
                label="Phone B",
            )
            self.assertTrue(reg.soft_delete(rec.camera_id))
            records = reg.list_records()
            self.assertEqual(len(records), 0)


if __name__ == "__main__":
    unittest.main()
