import unittest

import numpy as np

from app.core.joint_tracking import JointStateTracker


class JointTrackingTests(unittest.TestCase):
    def test_measured_joint_pass_through(self):
        tracker = JointStateTracker(hold_ms=250)
        out = tracker.stabilize(
            timestamp=10.0,
            measured_pose3d={0: np.array([1.0, 2.0, 3.0])},
            measured_confidences={0: 0.9},
        )
        self.assertIn(0, out)
        self.assertEqual(out[0]["state"], "measured")
        self.assertEqual(out[0]["age_ms"], 0)

    def test_hold_joint_within_window(self):
        tracker = JointStateTracker(hold_ms=250)
        tracker.stabilize(
            timestamp=10.0,
            measured_pose3d={0: np.array([1.0, 2.0, 3.0])},
            measured_confidences={0: 0.8},
        )
        out = tracker.stabilize(
            timestamp=10.2,
            measured_pose3d={},
            measured_confidences={},
        )
        self.assertIn(0, out)
        self.assertEqual(out[0]["state"], "held")
        self.assertGreaterEqual(out[0]["age_ms"], 1)

    def test_drop_joint_after_hold_window(self):
        tracker = JointStateTracker(hold_ms=250)
        tracker.stabilize(
            timestamp=10.0,
            measured_pose3d={0: np.array([1.0, 2.0, 3.0])},
            measured_confidences={0: 0.8},
        )
        out = tracker.stabilize(
            timestamp=10.3,
            measured_pose3d={},
            measured_confidences={},
        )
        self.assertNotIn(0, out)


if __name__ == "__main__":
    unittest.main()
