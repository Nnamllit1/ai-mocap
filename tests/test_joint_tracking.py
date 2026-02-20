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

    def test_single_view_state_is_preserved(self):
        tracker = JointStateTracker(hold_ms=250)
        out = tracker.stabilize(
            timestamp=10.0,
            measured_pose3d={0: np.array([1.0, 2.0, 3.0])},
            measured_confidences={0: 0.5},
            measured_states={0: "single_view"},
        )
        self.assertIn(0, out)
        self.assertEqual(out[0]["state"], "single_view")

    def test_low_confidence_large_jump_is_rejected(self):
        tracker = JointStateTracker(
            hold_ms=500,
            max_jump_m=0.1,
            jump_reject_conf=0.9,
        )
        tracker.stabilize(
            timestamp=10.0,
            measured_pose3d={0: np.array([0.0, 0.0, 0.0])},
            measured_confidences={0: 0.95},
        )
        out = tracker.stabilize(
            timestamp=10.05,
            measured_pose3d={0: np.array([1.0, 0.0, 0.0])},
            measured_confidences={0: 0.2},
        )
        self.assertIn(0, out)
        self.assertEqual(out[0]["state"], "held")
        np.testing.assert_allclose(out[0]["xyz"], np.array([0.0, 0.0, 0.0]), rtol=0, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
