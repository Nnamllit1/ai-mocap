from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from app.core.constants import COCO_JOINTS
from app.models.config import ExportConfig


class ExportManager:
    def __init__(self, cfg: ExportConfig):
        self.cfg = cfg
        self.rows: List[dict] = []

    def append(self, timestamp: float, joints: Dict[int, np.ndarray]) -> None:
        for joint_idx, xyz in joints.items():
            self.rows.append(
                {
                    "timestamp": float(timestamp),
                    "joint_id": int(joint_idx),
                    "joint_name": COCO_JOINTS[joint_idx],
                    "x": float(xyz[0]),
                    "y": float(xyz[1]),
                    "z": float(xyz[2]),
                    "confidence": 1.0,
                    "valid": True,
                }
            )

    def flush_live(self) -> None:
        self._flush(Path(self.cfg.live_json_path), Path(self.cfg.live_csv_path))

    def flush_offline(self) -> None:
        self._flush(Path(self.cfg.offline_json_path), Path(self.cfg.offline_csv_path))

    def _flush(self, json_path: Path, csv_path: Path) -> None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(self.rows, indent=2), encoding="utf-8")
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "timestamp",
                    "joint_id",
                    "joint_name",
                    "x",
                    "y",
                    "z",
                    "confidence",
                    "valid",
                ],
            )
            writer.writeheader()
            writer.writerows(self.rows)

    def clear(self) -> None:
        self.rows.clear()
