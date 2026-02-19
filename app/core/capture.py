from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock
from typing import Dict

import cv2
import numpy as np


@dataclass
class CameraFrame:
    camera_id: str
    timestamp: float
    frame: np.ndarray
    seq: int


@dataclass
class CameraHealth:
    connected: bool
    last_seen: float
    seq: int


class CaptureHub:
    def __init__(self, heartbeat_timeout_s: float):
        self._frames: Dict[str, CameraFrame] = {}
        self._health: Dict[str, CameraHealth] = {}
        self._lock = Lock()
        self._heartbeat_timeout_s = heartbeat_timeout_s

    def ingest_jpeg(self, camera_id: str, payload: bytes) -> bool:
        np_bytes = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            return False
        now = time.time()
        with self._lock:
            seq = self._health.get(camera_id, CameraHealth(False, 0.0, 0)).seq + 1
            self._frames[camera_id] = CameraFrame(
                camera_id=camera_id, timestamp=now, frame=frame, seq=seq
            )
            self._health[camera_id] = CameraHealth(True, now, seq)
        return True

    def heartbeat(self, camera_id: str) -> None:
        now = time.time()
        with self._lock:
            prev = self._health.get(camera_id, CameraHealth(False, 0.0, 0))
            self._health[camera_id] = CameraHealth(True, now, prev.seq)

    def mark_disconnected(self, camera_id: str) -> None:
        with self._lock:
            if camera_id in self._health:
                current = self._health[camera_id]
                self._health[camera_id] = CameraHealth(False, current.last_seen, current.seq)

    def get_synced_frames(self, min_sources: int, max_latency_ms: int) -> Dict[str, CameraFrame]:
        with self._lock:
            frames = dict(self._frames)
        if len(frames) < min_sources:
            return {}
        newest = max(frame.timestamp for frame in frames.values())
        window_s = max_latency_ms / 1000.0
        selected = {
            cid: frame
            for cid, frame in frames.items()
            if newest - frame.timestamp <= window_s
        }
        if len(selected) < min_sources:
            return {}
        return selected

    def get_latest_frames(self, camera_ids: list[str]) -> Dict[str, CameraFrame]:
        with self._lock:
            frames = dict(self._frames)
        return {cid: frames[cid] for cid in camera_ids if cid in frames}

    def get_frame_diagnostics(self, camera_ids: list[str], max_latency_ms: int) -> dict:
        now = time.time()
        with self._lock:
            frames = dict(self._frames)
            health = dict(self._health)
        newest_ts = max((frames[cid].timestamp for cid in camera_ids if cid in frames), default=0.0)
        window_s = max_latency_ms / 1000.0
        per_camera = {}
        available_ts = []
        for cid in camera_ids:
            frame = frames.get(cid)
            cam_health = health.get(cid, CameraHealth(False, 0.0, 0))
            if frame is not None:
                age_ms = max(0.0, (now - frame.timestamp) * 1000.0)
                in_sync = (newest_ts - frame.timestamp) <= window_s if newest_ts > 0 else False
                available_ts.append(frame.timestamp)
            else:
                age_ms = None
                in_sync = False
            timed_out = (now - cam_health.last_seen) > self._heartbeat_timeout_s
            connected = bool(cam_health.connected and not timed_out)
            per_camera[cid] = {
                "connected": connected,
                "latest_frame_age_ms": age_ms,
                "in_sync": in_sync,
                "seq": cam_health.seq,
            }
        if len(available_ts) >= 2:
            skew_ms = (max(available_ts) - min(available_ts)) * 1000.0
        else:
            skew_ms = 0.0
        return {"per_camera": per_camera, "sync_skew_ms": skew_ms}

    def latest_jpeg(self, camera_id: str) -> bytes | None:
        with self._lock:
            frame = self._frames.get(camera_id)
        if frame is None:
            return None
        ok, jpg = cv2.imencode(".jpg", frame.frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ok:
            return None
        return jpg.tobytes()

    def health_snapshot(self) -> Dict[str, dict]:
        now = time.time()
        with self._lock:
            snapshot = dict(self._health)
        out = {}
        for cid, health in snapshot.items():
            timed_out = (now - health.last_seen) > self._heartbeat_timeout_s
            out[cid] = {
                "connected": bool(health.connected and not timed_out),
                "last_seen": health.last_seen,
                "seq": health.seq,
            }
        return out
