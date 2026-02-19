import time
import unittest

import cv2
import numpy as np

from app.core.capture import CaptureHub


class CaptureSyncTests(unittest.TestCase):
    def _jpg(self):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", frame)
        self.assertTrue(ok)
        return encoded.tobytes()

    def test_sync_window(self):
        hub = CaptureHub(heartbeat_timeout_s=5.0)
        jpg = self._jpg()
        hub.ingest_jpeg("a", jpg)
        time.sleep(0.02)
        hub.ingest_jpeg("b", jpg)
        synced = hub.get_synced_frames(min_sources=2, max_latency_ms=100)
        self.assertEqual(len(synced), 2)


if __name__ == "__main__":
    unittest.main()
