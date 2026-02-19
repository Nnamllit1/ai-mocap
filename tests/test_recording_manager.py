import tempfile
import unittest

import numpy as np

from app.services.recording_manager import RecordingManager


class RecordingManagerTests(unittest.TestCase):
    def test_start_stop_transitions(self):
        mgr = RecordingManager()
        start = mgr.start(active_cameras=2)
        self.assertEqual(start["message"], "started")
        self.assertEqual(mgr.status()["state"], "recording")
        stop = mgr.stop()
        self.assertEqual(stop["message"], "stopped")
        self.assertEqual(mgr.status()["state"], "idle")

    def test_frame_append_counts(self):
        mgr = RecordingManager()
        mgr.start(active_cameras=1)
        mgr.append_frame(
            timestamp=10.0,
            joint_states={
                0: {
                    "xyz": np.array([1.0, 2.0, 3.0]),
                    "confidence": 0.9,
                    "state": "measured",
                    "age_ms": 0,
                }
            },
            metrics={"active_cameras": 1, "valid_joints": 1},
        )
        clip = mgr.stop()["clip"]
        self.assertEqual(clip["frame_count"], 1)
        self.assertEqual(clip["joint_samples"], 1)

    def test_export_requires_stopped_clip(self):
        mgr = RecordingManager()
        clip_id = mgr.start(active_cameras=1)["clip"]["clip_id"]
        with self.assertRaises(ValueError):
            mgr.export_clip(clip_id)
        mgr.stop()
        with tempfile.TemporaryDirectory() as tmp:
            out = mgr.export_clip(clip_id, out_dir=tmp)
            self.assertTrue(out["paths"]["json"].endswith(f"{clip_id}.json"))
            self.assertTrue(out["paths"]["csv"].endswith(f"{clip_id}.csv"))


if __name__ == "__main__":
    unittest.main()
