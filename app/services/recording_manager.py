from __future__ import annotations

import csv
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

from app.core.constants import COCO_JOINTS


class RecordingManager:
    def __init__(self, max_history: int = 50):
        self.max_history = max(1, int(max_history))
        self._lock = threading.Lock()
        self._clips: list[dict] = []
        self._active_clip: Optional[dict] = None

    def status(self) -> dict:
        with self._lock:
            active = self._active_clip
            return {
                "state": "recording" if active else "idle",
                "active_clip": self._clip_summary(active) if active else None,
            }

    def list_clips(self) -> list[dict]:
        with self._lock:
            items = [self._clip_summary(c) for c in self._clips]
            items.sort(key=lambda item: float(item.get("started_at", 0.0)), reverse=True)
            return items

    def start(self, active_cameras: int = 0) -> dict:
        with self._lock:
            if self._active_clip is not None:
                return {"ok": True, "message": "already_recording", "clip": self._clip_summary(self._active_clip)}
            clip = {
                "clip_id": f"clip_{uuid.uuid4().hex[:12]}",
                "started_at": float(time.time()),
                "stopped_at": None,
                "frame_count": 0,
                "joint_samples": 0,
                "status": "recording",
                "exported_paths": {},
                "frames": [],
                "start_active_cameras": int(active_cameras),
            }
            self._active_clip = clip
            return {"ok": True, "message": "started", "clip": self._clip_summary(clip)}

    def stop(self) -> dict:
        with self._lock:
            if self._active_clip is None:
                return {"ok": True, "message": "already_stopped", "clip": None}
            clip = self._active_clip
            clip["status"] = "stopped"
            if clip["stopped_at"] is None:
                clip["stopped_at"] = float(time.time())
            self._clips.append(clip)
            self._active_clip = None
            if len(self._clips) > self.max_history:
                self._clips = self._clips[-self.max_history :]
            return {"ok": True, "message": "stopped", "clip": self._clip_summary(clip)}

    def append_frame(self, timestamp: float, joint_states: Dict[int, dict], metrics: dict) -> None:
        with self._lock:
            if self._active_clip is None:
                return
            clip = self._active_clip
            ts = float(timestamp)
            serial_joint_states = {}
            for joint_idx, entry in (joint_states or {}).items():
                xyz = entry.get("xyz", [0.0, 0.0, 0.0])
                serial_joint_states[str(joint_idx)] = {
                    "xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                    "confidence": float(entry.get("confidence", 0.0)),
                    "state": str(entry.get("state", "measured")),
                    "age_ms": int(entry.get("age_ms", 0)),
                }

            clip["frames"].append(
                {
                    "timestamp": ts,
                    "joint_states": serial_joint_states,
                    "metrics": {
                        "active_cameras": int(metrics.get("active_cameras", 0)),
                        "valid_joints": int(metrics.get("valid_joints", 0)),
                    },
                }
            )
            clip["frame_count"] += 1
            clip["joint_samples"] += len(serial_joint_states)
            clip["stopped_at"] = ts

    def export_clip(
        self,
        clip_id: str,
        out_dir: str = "data/exports/clips",
    ) -> dict:
        with self._lock:
            clip = self._find_clip(clip_id)
            if clip is None:
                raise KeyError("clip_not_found")
            if clip["status"] == "recording":
                raise ValueError("clip_still_recording")
            payload = {
                "clip_id": clip["clip_id"],
                "started_at": clip["started_at"],
                "stopped_at": clip["stopped_at"],
                "frame_count": clip["frame_count"],
                "joint_samples": clip["joint_samples"],
                "status": clip["status"],
                "frames": clip["frames"],
            }

        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)
        json_path = root / f"{clip_id}.json"
        csv_path = root / f"{clip_id}.csv"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "clip_id",
                    "timestamp",
                    "joint_id",
                    "joint_name",
                    "x",
                    "y",
                    "z",
                    "confidence",
                    "state",
                    "age_ms",
                    "valid",
                ],
            )
            writer.writeheader()
            for frame in clip["frames"]:
                ts = float(frame.get("timestamp", 0.0))
                joint_states = frame.get("joint_states", {})
                for joint_idx_str, state in joint_states.items():
                    joint_idx = int(joint_idx_str)
                    xyz = state.get("xyz", [0.0, 0.0, 0.0])
                    writer.writerow(
                        {
                            "clip_id": clip["clip_id"],
                            "timestamp": ts,
                            "joint_id": joint_idx,
                            "joint_name": COCO_JOINTS[joint_idx],
                            "x": float(xyz[0]),
                            "y": float(xyz[1]),
                            "z": float(xyz[2]),
                            "confidence": float(state.get("confidence", 0.0)),
                            "state": str(state.get("state", "measured")),
                            "age_ms": int(state.get("age_ms", 0)),
                            "valid": True,
                        }
                    )

        with self._lock:
            clip = self._find_clip(clip_id)
            if clip is not None:
                clip["exported_paths"] = {
                    "json": str(json_path).replace("\\", "/"),
                    "csv": str(csv_path).replace("\\", "/"),
                }
                return {
                    "ok": True,
                    "clip_id": clip_id,
                    "paths": clip["exported_paths"],
                }
        return {"ok": True, "clip_id": clip_id, "paths": {}}

    @staticmethod
    def _clip_summary(clip: Optional[dict]) -> Optional[dict]:
        if clip is None:
            return None
        return {
            "clip_id": clip["clip_id"],
            "started_at": clip["started_at"],
            "stopped_at": clip["stopped_at"],
            "frame_count": clip["frame_count"],
            "joint_samples": clip["joint_samples"],
            "status": clip["status"],
            "exported_paths": clip.get("exported_paths", {}),
            "start_active_cameras": int(clip.get("start_active_cameras", 0)),
        }

    def _find_clip(self, clip_id: str) -> Optional[dict]:
        if self._active_clip and self._active_clip.get("clip_id") == clip_id:
            return self._active_clip
        for clip in self._clips:
            if clip.get("clip_id") == clip_id:
                return clip
        return None
