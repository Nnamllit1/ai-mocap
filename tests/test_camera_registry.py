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

    def test_soft_delete(self):
        reg = CameraRegistry()
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
