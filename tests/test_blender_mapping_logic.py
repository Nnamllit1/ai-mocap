import unittest

from integrations.blender.mocap_live_sync_addon import (
    AXIS_PRESET_PASSTHROUGH,
    AXIS_PRESET_Z_UP,
    apply_axis_preset,
)


class BlenderMappingLogicTests(unittest.TestCase):
    def test_passthrough_mapping(self):
        out = apply_axis_preset((1.0, 2.0, 3.0), AXIS_PRESET_PASSTHROUGH, scale=1.0)
        self.assertEqual(out, (1.0, 2.0, 3.0))

    def test_scale_multiplier_is_applied(self):
        out = apply_axis_preset((1.0, -2.0, 0.5), AXIS_PRESET_PASSTHROUGH, scale=2.0)
        self.assertEqual(out, (2.0, -4.0, 1.0))

    def test_z_up_mapping(self):
        out = apply_axis_preset((1.0, 2.0, 3.0), AXIS_PRESET_Z_UP, scale=1.0)
        self.assertEqual(out, (1.0, 3.0, -2.0))

    def test_unknown_mapping_falls_back_to_passthrough(self):
        out = apply_axis_preset((4.0, 5.0, 6.0), "unknown_mapping", scale=1.0)
        self.assertEqual(out, (4.0, 5.0, 6.0))


if __name__ == "__main__":
    unittest.main()
