import struct
import unittest

from integrations.blender.mocap_live_sync_addon import (
    decode_joint_packet,
    decode_osc_message,
    decode_status_packet,
    read_osc_padded_string,
)


def _osc_pad(text: str) -> bytes:
    raw = text.encode("utf-8") + b"\x00"
    pad = (4 - (len(raw) % 4)) % 4
    return raw + (b"\x00" * pad)


def _build_float_packet(address: str, values: list[float], endian: str) -> bytes:
    tags = "," + ("f" * len(values))
    payload = b"".join(struct.pack(endian + "f", float(v)) for v in values)
    return _osc_pad(address) + _osc_pad(tags) + payload


class BlenderOscDecoderTests(unittest.TestCase):
    def test_decode_joint_packet_little_endian(self):
        packet = _build_float_packet(
            "/mocap/joint/nose",
            [1.0, 2.0, 3.0, 0.9, 1771589427.0],
            "<",
        )
        out = decode_joint_packet(packet, "/mocap")
        self.assertIsNotNone(out)
        self.assertEqual(out["joint_name"], "nose")
        self.assertAlmostEqual(out["xyz"][0], 1.0, places=5)
        self.assertAlmostEqual(out["confidence"], 0.9, places=4)

    def test_decode_status_packet_big_endian(self):
        packet = _build_float_packet("/mocap/status", [2.0, 17.0], ">")
        out = decode_status_packet(packet, "/mocap")
        self.assertIsNotNone(out)
        self.assertAlmostEqual(out["active_cameras"], 2.0, places=5)
        self.assertAlmostEqual(out["valid_joints"], 17.0, places=5)

    def test_reject_malformed_short_payload(self):
        packet = _osc_pad("/mocap/joint/nose") + _osc_pad(",fffff") + b"\x00\x00\x00\x00"
        self.assertIsNone(decode_osc_message(packet))
        self.assertIsNone(decode_joint_packet(packet, "/mocap"))

    def test_read_osc_padded_string_alignment(self):
        payload = _osc_pad("/abc") + _osc_pad(",ff")
        text, offset = read_osc_padded_string(payload, 0)
        self.assertEqual(text, "/abc")
        self.assertEqual(offset % 4, 0)
        tags, _ = read_osc_padded_string(payload, offset)
        self.assertEqual(tags, ",ff")


if __name__ == "__main__":
    unittest.main()
