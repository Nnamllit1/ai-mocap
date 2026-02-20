import unittest

from integrations.blender.mocap_live_sync_addon import (
    JointCacheEntry,
    resolve_joint_output_position,
    update_joint_cache_entry,
)


class BlenderConfidenceHoldTests(unittest.TestCase):
    def test_high_confidence_updates_last_good(self):
        entry = JointCacheEntry()
        update_joint_cache_entry(
            entry,
            mapped_xyz=(1.0, 2.0, 3.0),
            confidence=0.9,
            packet_timestamp=10.0,
            now_monotonic=100.0,
            confidence_threshold=0.6,
        )
        self.assertEqual(entry.last_good_position, (1.0, 2.0, 3.0))
        pos, stale = resolve_joint_output_position(entry, now_monotonic=100.1, stale_timeout_s=0.5)
        self.assertEqual(pos, (1.0, 2.0, 3.0))
        self.assertFalse(stale)

    def test_low_confidence_holds_last_good(self):
        entry = JointCacheEntry()
        update_joint_cache_entry(
            entry,
            mapped_xyz=(1.0, 1.0, 1.0),
            confidence=0.95,
            packet_timestamp=1.0,
            now_monotonic=50.0,
            confidence_threshold=0.6,
        )
        update_joint_cache_entry(
            entry,
            mapped_xyz=(5.0, 5.0, 5.0),
            confidence=0.2,
            packet_timestamp=2.0,
            now_monotonic=50.1,
            confidence_threshold=0.6,
        )
        pos, _stale = resolve_joint_output_position(entry, now_monotonic=50.2, stale_timeout_s=1.0)
        self.assertEqual(pos, (1.0, 1.0, 1.0))

    def test_stale_timeout_flagging(self):
        entry = JointCacheEntry()
        update_joint_cache_entry(
            entry,
            mapped_xyz=(0.0, 0.0, 0.0),
            confidence=0.9,
            packet_timestamp=1.0,
            now_monotonic=10.0,
            confidence_threshold=0.5,
        )
        _pos, stale = resolve_joint_output_position(entry, now_monotonic=11.5, stale_timeout_s=1.0)
        self.assertTrue(stale)


if __name__ == "__main__":
    unittest.main()
