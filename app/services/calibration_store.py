from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


class CalibrationStore:
    def __init__(self):
        self.intrinsics: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.extrinsics: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def save(self, path: Path) -> None:
        payload = {"intrinsics": {}, "extrinsics": {}}
        for cam_id, (k_mat, dist) in self.intrinsics.items():
            payload["intrinsics"][cam_id] = {
                "K": k_mat.tolist(),
                "dist": dist.reshape(-1).tolist(),
            }
        for cam_id, (r_mat, t_vec) in self.extrinsics.items():
            payload["extrinsics"][cam_id] = {
                "R": r_mat.tolist(),
                "t": t_vec.reshape(-1).tolist(),
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "CalibrationStore":
        payload = json.loads(path.read_text(encoding="utf-8"))
        store = CalibrationStore()
        for cam_id, values in payload.get("intrinsics", {}).items():
            store.intrinsics[cam_id] = (
                np.array(values["K"], dtype=np.float64),
                np.array(values["dist"], dtype=np.float64).reshape(-1, 1),
            )
        for cam_id, values in payload.get("extrinsics", {}).items():
            store.extrinsics[cam_id] = (
                np.array(values["R"], dtype=np.float64),
                np.array(values["t"], dtype=np.float64).reshape(3, 1),
            )
        return store
