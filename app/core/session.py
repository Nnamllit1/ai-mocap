from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from app.core.capture import CaptureHub
from app.core.constants import COCO_JOINTS
from app.core.events import EventBus, Pose3DEvent, SessionStatusEvent
from app.core.osc import BlenderOscSink
from app.core.pose import PoseEstimator
from app.core.smoothing import JointSmoother
from app.core.triangulation import TriangulationEngine
from app.models.config import AppConfig
from app.services.calibration_store import CalibrationStore
from app.services.export_manager import ExportManager


@dataclass
class SessionState:
    running: bool = False
    active_cameras: int = 0
    valid_joints: int = 0
    message: str = "idle"
    last_pose3d: dict = field(default_factory=dict)
    last_joint_confidences: dict = field(default_factory=dict)
    last_timestamp: float = 0.0
    loop_fps: float = 0.0
    loop_ms: float = 0.0
    dropped_cycles: int = 0
    last_active_camera_ids: list[str] = field(default_factory=list)
    joint_conf_avg: float | None = None


class SessionManager:
    def __init__(self, cfg: AppConfig, capture_hub: CaptureHub, event_bus: EventBus):
        self.cfg = cfg
        self.capture_hub = capture_hub
        self.event_bus = event_bus
        self.state = SessionState()
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()

    def _emit_joints_to_sinks(
        self,
        osc_sink: BlenderOscSink,
        pose3d: Dict[int, np.ndarray],
        joint_confidences: Dict[int, float],
        timestamp: float,
    ) -> None:
        for joint_idx, xyz in pose3d.items():
            conf = float(joint_confidences.get(joint_idx, 0.0))
            osc_sink.send_joint(COCO_JOINTS[joint_idx], xyz, conf, timestamp)

    def _emit_status_to_sinks(
        self,
        osc_sink: BlenderOscSink,
        active_cameras: int,
        valid_joints: int,
    ) -> None:
        osc_sink.send_status(active_cameras=active_cameras, valid_joints=valid_joints)

    def _load_calibration(self) -> CalibrationStore:
        path = Path(self.cfg.calibration.output_path)
        if not path.exists():
            raise FileNotFoundError(f"Calibration not found: {path}")
        return CalibrationStore.load(path)

    def start(self) -> dict:
        with self._lock:
            if self.state.running:
                return {"ok": True, "message": "already_running"}
            self._stop_evt.clear()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            self.state.running = True
            self.state.message = "starting"
        return {"ok": True, "message": "started"}

    def stop(self) -> dict:
        with self._lock:
            if not self.state.running:
                return {"ok": True, "message": "already_stopped"}
            self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        with self._lock:
            self.state.running = False
            self.state.message = "stopped"
        return {"ok": True, "message": "stopped"}

    def status(self) -> dict:
        health = self.capture_hub.health_snapshot()
        return {
            "running": self.state.running,
            "active_cameras": self.state.active_cameras,
            "valid_joints": self.state.valid_joints,
            "message": self.state.message,
            "cameras": health,
            "last_timestamp": self.state.last_timestamp,
            "last_pose3d": self.state.last_pose3d,
            "last_joint_confidences": self.state.last_joint_confidences,
            "loop_fps": self.state.loop_fps,
            "loop_ms": self.state.loop_ms,
            "dropped_cycles": self.state.dropped_cycles,
            "last_active_camera_ids": self.state.last_active_camera_ids,
            "joint_conf_avg": self.state.joint_conf_avg,
        }

    def _run_loop(self) -> None:
        try:
            calibration = self._load_calibration()
            triangulator = TriangulationEngine(calibration, self.cfg.triangulation)
            estimator = PoseEstimator(self.cfg.model)
            smoother = JointSmoother(self.cfg.runtime.ema_alpha)
            osc_sink = BlenderOscSink(self.cfg.osc)
            exporter = ExportManager(self.cfg.export)
            target_dt = 1.0 / max(self.cfg.runtime.target_fps, 1)
            self.state.message = "running"
            prev_cycle_end = time.time()

            while not self._stop_evt.is_set():
                t0 = time.time()
                frames = self.capture_hub.get_synced_frames(
                    min_sources=self.cfg.triangulation.min_views,
                    max_latency_ms=self.cfg.runtime.max_latency_ms,
                )
                self.state.last_active_camera_ids = list(frames.keys())
                if not frames:
                    self.state.active_cameras = 0
                    self.state.valid_joints = 0
                    self.state.joint_conf_avg = None
                    self.state.dropped_cycles += 1
                    time.sleep(0.01)
                    continue

                joints_by_source: Dict[str, Dict[int, tuple]] = {}
                for source_id, frame in frames.items():
                    joints_by_source[source_id] = estimator.detect(frame.frame)

                pose3d = {}
                joint_confidences: Dict[int, float] = {}
                timestamp = max(frame.timestamp for frame in frames.values())
                for joint_idx in range(len(COCO_JOINTS)):
                    obs = {}
                    for source_id, joints in joints_by_source.items():
                        if joint_idx in joints:
                            obs[source_id] = joints[joint_idx]
                    xyz = triangulator.triangulate_joint(obs)
                    if xyz is None:
                        continue
                    xyz = smoother.update(joint_idx, xyz)
                    pose3d[joint_idx] = xyz
                    conf = float(np.mean([item[2] for item in obs.values()])) if obs else 0.0
                    joint_confidences[joint_idx] = conf

                self._emit_joints_to_sinks(osc_sink, pose3d, joint_confidences, timestamp)
                self._emit_status_to_sinks(
                    osc_sink, active_cameras=len(frames), valid_joints=len(pose3d)
                )
                if self.cfg.export.enable_live_export and pose3d:
                    exporter.append(timestamp, pose3d)

                serial = {
                    str(idx): [float(v[0]), float(v[1]), float(v[2])] for idx, v in pose3d.items()
                }
                serial_conf = {str(idx): float(val) for idx, val in joint_confidences.items()}
                self.state.last_pose3d = serial
                self.state.last_joint_confidences = serial_conf
                self.state.last_timestamp = timestamp
                self.state.active_cameras = len(frames)
                self.state.valid_joints = len(pose3d)
                self.state.joint_conf_avg = (
                    float(np.mean(list(joint_confidences.values())))
                    if joint_confidences
                    else None
                )

                self.event_bus.publish("pose3d", Pose3DEvent(timestamp=timestamp, joints=pose3d))
                self.event_bus.publish(
                    "status",
                    SessionStatusEvent(
                        running=True,
                        active_cameras=len(frames),
                        valid_joints=len(pose3d),
                        message="running",
                    ),
                )

                elapsed = time.time() - t0
                elapsed_ms = elapsed * 1000.0
                if self.state.loop_ms <= 0.0:
                    self.state.loop_ms = elapsed_ms
                else:
                    self.state.loop_ms = (0.8 * self.state.loop_ms) + (0.2 * elapsed_ms)

                now = time.time()
                cycle_dt = max(1e-6, now - prev_cycle_end)
                inst_fps = 1.0 / cycle_dt
                if self.state.loop_fps <= 0.0:
                    self.state.loop_fps = inst_fps
                else:
                    self.state.loop_fps = (0.8 * self.state.loop_fps) + (0.2 * inst_fps)
                prev_cycle_end = now

                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)
        except Exception as exc:  # noqa: BLE001
            self.state.message = f"error: {exc}"
        finally:
            self.state.running = False
