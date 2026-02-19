from __future__ import annotations

import socket

import numpy as np

from app.models.config import OscConfig


class BlenderOscSink:
    def __init__(self, cfg: OscConfig):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = (cfg.host, int(cfg.port))
        self.prefix = cfg.address_prefix.rstrip("/")

    @staticmethod
    def _pad4(data: bytes) -> bytes:
        pad = (4 - (len(data) % 4)) % 4
        return data + (b"\x00" * pad)

    def _osc_str(self, value: str) -> bytes:
        return self._pad4(value.encode("utf-8") + b"\x00")

    @staticmethod
    def _osc_float(value: float) -> bytes:
        return np.float32(value).tobytes()

    def send_joint(self, joint_name: str, xyz, confidence: float, timestamp: float) -> None:
        address = f"{self.prefix}/joint/{joint_name}"
        payload = b"".join(
            [
                self._osc_str(address),
                self._osc_str(",fffff"),
                self._osc_float(float(xyz[0])),
                self._osc_float(float(xyz[1])),
                self._osc_float(float(xyz[2])),
                self._osc_float(float(confidence)),
                self._osc_float(float(timestamp)),
            ]
        )
        self.sock.sendto(payload, self.addr)

    def send_status(self, active_cameras: int, valid_joints: int) -> None:
        address = f"{self.prefix}/status"
        payload = b"".join(
            [
                self._osc_str(address),
                self._osc_str(",ff"),
                self._osc_float(float(active_cameras)),
                self._osc_float(float(valid_joints)),
            ]
        )
        self.sock.sendto(payload, self.addr)
