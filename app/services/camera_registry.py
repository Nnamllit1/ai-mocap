from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class CameraRecord:
    camera_id: str
    label: str
    device_uid: str
    device_name: str
    platform: str
    created_at: float
    last_seen: float
    connected: bool
    enabled: bool = True
    stream_profile: dict = field(default_factory=dict)
    ws_token: str = ""
    deleted: bool = False


class CameraRegistry:
    def __init__(self):
        self._records: Dict[str, CameraRecord] = {}
        self._device_to_camera: Dict[str, str] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _new_camera_id() -> str:
        return f"cam_{secrets.token_urlsafe(5)}"

    @staticmethod
    def _new_ws_token() -> str:
        return secrets.token_urlsafe(16)

    def upsert_from_device(
        self,
        *,
        device_uid: str,
        device_name: str,
        platform: str,
        label: str | None = None,
    ) -> CameraRecord:
        now = time.time()
        with self._lock:
            existing_id = self._device_to_camera.get(device_uid)
            if existing_id and existing_id in self._records:
                rec = self._records[existing_id]
                rec.device_name = device_name
                rec.platform = platform
                rec.last_seen = now
                rec.connected = False
                rec.deleted = False
                if label:
                    rec.label = label
                rec.ws_token = self._new_ws_token()
                return CameraRecord(**rec.__dict__)

            camera_id = self._new_camera_id()
            rec = CameraRecord(
                camera_id=camera_id,
                label=label or f"Camera {len(self._records) + 1}",
                device_uid=device_uid,
                device_name=device_name,
                platform=platform,
                created_at=now,
                last_seen=now,
                connected=False,
                ws_token=self._new_ws_token(),
            )
            self._records[camera_id] = rec
            self._device_to_camera[device_uid] = camera_id
            return CameraRecord(**rec.__dict__)

    def list_records(self, include_deleted: bool = False) -> list[dict]:
        with self._lock:
            records = list(self._records.values())
        out = []
        for rec in records:
            if rec.deleted and not include_deleted:
                continue
            item = rec.__dict__.copy()
            item.pop("ws_token", None)
            out.append(item)
        return sorted(out, key=lambda x: x["created_at"])

    def get(self, camera_id: str) -> Optional[CameraRecord]:
        with self._lock:
            rec = self._records.get(camera_id)
            if rec is None:
                return None
            return CameraRecord(**rec.__dict__)

    def set_connected(self, camera_id: str, connected: bool) -> None:
        with self._lock:
            rec = self._records.get(camera_id)
            if rec is None:
                return
            rec.connected = connected
            rec.last_seen = time.time()

    def heartbeat(self, camera_id: str) -> None:
        with self._lock:
            rec = self._records.get(camera_id)
            if rec is None:
                return
            rec.connected = True
            rec.last_seen = time.time()

    def update(self, camera_id: str, *, label: str | None = None, enabled: bool | None = None) -> Optional[dict]:
        with self._lock:
            rec = self._records.get(camera_id)
            if rec is None:
                return None
            if label is not None:
                rec.label = label
            if enabled is not None:
                rec.enabled = bool(enabled)
            item = rec.__dict__.copy()
            item.pop("ws_token", None)
            return item

    def soft_delete(self, camera_id: str) -> bool:
        with self._lock:
            rec = self._records.get(camera_id)
            if rec is None:
                return False
            rec.deleted = True
            rec.connected = False
            return True

    def validate_ws_token(self, camera_id: str, ws_token: str) -> bool:
        with self._lock:
            rec = self._records.get(camera_id)
            if rec is None or rec.deleted:
                return False
            return bool(ws_token) and rec.ws_token == ws_token
