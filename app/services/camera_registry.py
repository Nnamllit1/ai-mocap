from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from app.services.state_io import load_json, save_json_atomic


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
    def __init__(self, state_path: str | Path | None = None):
        self._records: Dict[str, CameraRecord] = {}
        self._device_to_camera: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._state_path = Path(state_path) if state_path else None
        self._load_state()

    @staticmethod
    def _to_payload(record: CameraRecord) -> dict:
        return {
            "camera_id": str(record.camera_id),
            "label": str(record.label),
            "device_uid": str(record.device_uid),
            "device_name": str(record.device_name),
            "platform": str(record.platform),
            "created_at": float(record.created_at),
            "last_seen": float(record.last_seen),
            "connected": bool(record.connected),
            "enabled": bool(record.enabled),
            "stream_profile": dict(record.stream_profile or {}),
            "ws_token": str(record.ws_token),
            "deleted": bool(record.deleted),
        }

    @staticmethod
    def _from_payload(payload: dict) -> CameraRecord:
        return CameraRecord(
            camera_id=str(payload["camera_id"]),
            label=str(payload.get("label", "")),
            device_uid=str(payload.get("device_uid", "")),
            device_name=str(payload.get("device_name", "")),
            platform=str(payload.get("platform", "unknown")),
            created_at=float(payload.get("created_at", time.time())),
            last_seen=float(payload.get("last_seen", 0.0)),
            connected=bool(payload.get("connected", False)),
            enabled=bool(payload.get("enabled", True)),
            stream_profile=dict(payload.get("stream_profile", {})),
            ws_token=str(payload.get("ws_token", "")),
            deleted=bool(payload.get("deleted", False)),
        )

    def _save_state_locked(self) -> None:
        if self._state_path is None:
            return
        payload = {
            "records": [self._to_payload(rec) for rec in self._records.values()],
            "device_to_camera": dict(self._device_to_camera),
        }
        save_json_atomic(self._state_path, payload)

    def _load_state(self) -> None:
        if self._state_path is None:
            return
        payload = load_json(self._state_path, default={"records": [], "device_to_camera": {}})
        records = payload.get("records", []) if isinstance(payload, dict) else []
        mapping = payload.get("device_to_camera", {}) if isinstance(payload, dict) else {}
        with self._lock:
            self._records = {}
            for item in records:
                try:
                    rec = self._from_payload(item)
                except Exception:  # noqa: BLE001
                    continue
                rec.connected = False
                self._records[rec.camera_id] = rec
            self._device_to_camera = {}
            if isinstance(mapping, dict):
                for device_uid, camera_id in mapping.items():
                    if camera_id in self._records:
                        self._device_to_camera[str(device_uid)] = str(camera_id)
            if not self._device_to_camera:
                for rec in self._records.values():
                    if rec.device_uid:
                        self._device_to_camera[rec.device_uid] = rec.camera_id
            self._save_state_locked()

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
                self._save_state_locked()
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
            self._save_state_locked()
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

    def is_known_device_uid(self, device_uid: str) -> bool:
        key = str(device_uid or "")
        if not key:
            return False
        with self._lock:
            camera_id = self._device_to_camera.get(key)
            if not camera_id:
                return False
            rec = self._records.get(camera_id)
            return bool(rec is not None and not rec.deleted)

    def set_connected(self, camera_id: str, connected: bool) -> None:
        with self._lock:
            rec = self._records.get(camera_id)
            if rec is None:
                return
            rec.connected = connected
            rec.last_seen = time.time()
            self._save_state_locked()

    def heartbeat(self, camera_id: str) -> None:
        with self._lock:
            rec = self._records.get(camera_id)
            if rec is None:
                return
            rec.connected = True
            rec.last_seen = time.time()
            self._save_state_locked()

    def update(self, camera_id: str, *, label: str | None = None, enabled: bool | None = None) -> Optional[dict]:
        with self._lock:
            rec = self._records.get(camera_id)
            if rec is None:
                return None
            if label is not None:
                rec.label = label
            if enabled is not None:
                rec.enabled = bool(enabled)
            self._save_state_locked()
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
            self._save_state_locked()
            return True

    def validate_ws_token(self, camera_id: str, ws_token: str) -> bool:
        with self._lock:
            rec = self._records.get(camera_id)
            if rec is None or rec.deleted:
                return False
            return bool(ws_token) and rec.ws_token == ws_token
