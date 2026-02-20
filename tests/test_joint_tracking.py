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

    def test_bone_length_baseline_learns_from_stable_measured_pairs(self):
        tracker = JointStateTracker(
            hold_ms=500,
            bone_length_guard_enabled=True,
            bone_length_learn_conf=0.65,
        )
        tracker.stabilize(
            timestamp=10.0,
            measured_pose3d={
                5: np.array([0.0, 0.0, 0.0]),
                7: np.array([1.0, 0.0, 0.0]),
            },
            measured_confidences={5: 0.95, 7: 0.95},
            measured_states={5: "measured", 7: "measured"},
        )
        self.assertIn((5, 7), tracker._bone_len_ema)
        self.assertAlmostEqual(float(tracker._bone_len_ema[(5, 7)]), 1.0, places=5)

    def test_bone_length_moderate_deviation_is_clamped(self):
        tracker = JointStateTracker(
            hold_ms=500,
            bone_length_guard_enabled=True,
            bone_length_soft_rel_tol=0.15,
            bone_length_hard_rel_tol=0.35,
        )
        tracker.stabilize(
            timestamp=10.0,
            measured_pose3d={
                5: np.array([0.0, 0.0, 0.0]),
                7: np.array([1.0, 0.0, 0.0]),
            },
            measured_confidences={5: 0.95, 7: 0.95},
        )
        out = tracker.stabilize(
            timestamp=10.05,
            measured_pose3d={
                5: np.array([0.0, 0.0, 0.0]),
                7: np.array([1.25, 0.0, 0.0]),
            },
            measured_confidences={5: 0.95, 7: 0.95},
        )
        self.assertIn(7, out)
        self.assertEqual(out[7]["state"], "measured")
        self.assertGreaterEqual(tracker.last_bone_guard_clamped_count, 1)
        np.testing.assert_allclose(
            out[7]["xyz"],
            np.array([1.15, 0.0, 0.0]),
            rtol=0,
            atol=1e-6,
        )

    def test_bone_length_severe_low_confidence_rejects_and_holds(self):
        tracker = JointStateTracker(
            hold_ms=500,
            max_jump_m=10.0,
            bone_length_guard_enabled=True,
            bone_length_soft_rel_tol=0.15,
            bone_length_hard_rel_tol=0.35,
            jump_reject_conf=0.85,
        )
        tracker.stabilize(
            timestamp=10.0,
            measured_pose3d={
                5: np.array([0.0, 0.0, 0.0]),
                7: np.array([1.0, 0.0, 0.0]),
            },
            measured_confidences={5: 0.95, 7: 0.95},
            measured_states={5: "measured", 7: "measured"},
        )
        out = tracker.stabilize(
            timestamp=10.05,
            measured_pose3d={
                5: np.array([0.0, 0.0, 0.0]),
                7: np.array([1.8, 0.0, 0.0]),
            },
            measured_confidences={5: 0.95, 7: 0.20},
            measured_states={5: "measured", 7: "measured"},
        )
        self.assertIn(7, out)
        self.assertEqual(out[7]["state"], "held")
        self.assertGreaterEqual(tracker.last_bone_guard_rejected_count, 1)
        np.testing.assert_allclose(
            out[7]["xyz"],
            np.array([1.0, 0.0, 0.0]),
            rtol=0,
            atol=1e-6,
        )

    def test_bone_length_severe_high_confidence_clamps(self):
        tracker = JointStateTracker(
            hold_ms=500,
            bone_length_guard_enabled=True,
            bone_length_soft_rel_tol=0.15,
            bone_length_hard_rel_tol=0.35,
            jump_reject_conf=0.85,
        )
        tracker.stabilize(
            timestamp=10.0,
            measured_pose3d={
                5: np.array([0.0, 0.0, 0.0]),
                7: np.array([1.0, 0.0, 0.0]),
            },
            measured_confidences={5: 0.95, 7: 0.95},
        )
        out = tracker.stabilize(
            timestamp=10.05,
            measured_pose3d={
                5: np.array([0.0, 0.0, 0.0]),
                7: np.array([1.8, 0.0, 0.0]),
            },
            measured_confidences={5: 0.95, 7: 0.95},
        )
        self.assertIn(7, out)
        self.assertEqual(out[7]["state"], "measured")
        self.assertEqual(tracker.last_bone_guard_rejected_count, 0)
        self.assertGreaterEqual(tracker.last_bone_guard_clamped_count, 1)
        np.testing.assert_allclose(
            out[7]["xyz"],
            np.array([1.15, 0.0, 0.0]),
            rtol=0,
            atol=1e-6,
        )

    def test_bone_length_baseline_not_learned_for_single_view_or_held(self):
        tracker = JointStateTracker(
            hold_ms=500,
            bone_length_guard_enabled=True,
            bone_length_learn_conf=0.65,
        )
        tracker.stabilize(
            timestamp=10.0,
            measured_pose3d={
                5: np.array([0.0, 0.0, 0.0]),
                7: np.array([1.0, 0.0, 0.0]),
            },
            measured_confidences={5: 0.95, 7: 0.95},
            measured_states={5: "measured", 7: "single_view"},
        )
        self.assertNotIn((5, 7), tracker._bone_len_ema)

        tracker.stabilize(
            timestamp=10.1,
            measured_pose3d={
                5: np.array([0.0, 0.0, 0.0]),
                7: np.array([1.0, 0.0, 0.0]),
            },
            measured_confidences={5: 0.95, 7: 0.95},
            measured_states={5: "measured", 7: "measured"},
        )
        baseline_before = float(tracker._bone_len_ema[(5, 7)])
        tracker.stabilize(
            timestamp=10.2,
            measured_pose3d={},
            measured_confidences={},
        )
        self.assertAlmostEqual(float(tracker._bone_len_ema[(5, 7)]), baseline_before, places=6)


if __name__ == "__main__":
    unittest.main()
