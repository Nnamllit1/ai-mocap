from __future__ import annotations

from pathlib import Path

import cv2

from app.core.constants import COCO_JOINTS
from app.core.pose import PoseEstimator
from app.core.smoothing import JointSmoother
from app.core.triangulation import TriangulationEngine
from app.models.config import AppConfig
from app.services.calibration_store import CalibrationStore
from app.services.export_manager import ExportManager


class OfflineProcessor:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def run_export(self) -> dict:
        enabled = [source for source in self.cfg.offline_sources if source.enabled]
        if len(enabled) < 2:
            return {"ok": False, "message": "need at least 2 enabled offline_sources"}

        cal_path = Path(self.cfg.calibration.output_path)
        if not cal_path.exists():
            return {"ok": False, "message": f"calibration missing: {cal_path}"}

        calibration = CalibrationStore.load(cal_path)
        triangulator = TriangulationEngine(calibration, self.cfg.triangulation)
        estimator = PoseEstimator(self.cfg.model)
        smoother = JointSmoother(self.cfg.runtime.ema_alpha)
        exporter = ExportManager(self.cfg.export)

        captures = {source.id: cv2.VideoCapture(source.path) for source in enabled}
        try:
            for source_id, cap in captures.items():
                if not cap.isOpened():
                    return {"ok": False, "message": f"cannot open {source_id}"}

            frame_idx = 0
            written = 0
            while True:
                frames = {}
                for source_id, cap in captures.items():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        exporter.flush_offline()
                        return {
                            "ok": True,
                            "frames": frame_idx,
                            "rows": len(exporter.rows),
                            "joints_written": written,
                        }
                    frames[source_id] = frame

                joints_by_cam = {source_id: estimator.detect(frame) for source_id, frame in frames.items()}
                pose3d = {}
                for joint_idx in range(len(COCO_JOINTS)):
                    obs = {}
                    for source_id, joints in joints_by_cam.items():
                        if joint_idx in joints:
                            obs[source_id] = joints[joint_idx]
                    xyz = triangulator.triangulate_joint(obs)
                    if xyz is None:
                        continue
                    pose3d[joint_idx] = smoother.update(joint_idx, xyz)
                if pose3d:
                    exporter.append(float(frame_idx), pose3d)
                    written += len(pose3d)
                frame_idx += 1
        finally:
            for cap in captures.values():
                cap.release()
