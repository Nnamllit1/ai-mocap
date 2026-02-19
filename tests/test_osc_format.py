import unittest

import numpy as np

from app.core.osc import BlenderOscSink
from app.models.config import OscConfig


class DummySocket:
    def __init__(self):
        self.sent = []

    def sendto(self, payload, addr):
        self.sent.append((payload, addr))


class OscTests(unittest.TestCase):
    def test_send_joint_packet(self):
        sink = BlenderOscSink(OscConfig(host="127.0.0.1", port=9000, address_prefix="/mocap"))
        sink.sock.close()
        dummy = DummySocket()
        sink.sock = dummy
        sink.send_joint("nose", np.array([1.0, 2.0, 3.0]), 0.9, 10.0)
        self.assertEqual(len(dummy.sent), 1)
        payload, _ = dummy.sent[0]
        self.assertIn(b"/mocap/joint/nose", payload)


if __name__ == "__main__":
    unittest.main()
