from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    token: str = "change-me"


class ModelConfig(BaseModel):
    path: str = "yolo26n-pose.pt"
    conf: float = 0.25
    iou: float = 0.45
    device: str = "cpu"


class RuntimeConfig(BaseModel):
    target_fps: int = 24
    max_latency_ms: int = 120
    ema_alpha: float = 0.6
    show_preview: bool = True
    missing_joint_hold_ms: int = 250
    max_joint_jump_m: float = 0.35
    jump_reject_conf: float = 0.85


class TriangulationConfig(BaseModel):
    min_views: int = 2
    pair_conf_threshold: float = 0.25
    reproj_error_max: float = 25.0
    allow_single_view_fallback: bool = True
    single_view_conf_scale: float = 0.55
    single_view_max_age_ms: int = 800


class OscConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 9000
    address_prefix: str = "/mocap"


class IngestConfig(BaseModel):
    jpeg_quality: int = 70
    max_frame_size: int = 1_500_000
    client_fps_cap: int = 20
    heartbeat_timeout_s: float = 6.0


class ExportConfig(BaseModel):
    enable_live_export: bool = True
    enable_offline_export: bool = True
    live_json_path: str = "data/exports/live_mocap.json"
    live_csv_path: str = "data/exports/live_mocap.csv"
    offline_json_path: str = "data/exports/offline_mocap.json"
    offline_csv_path: str = "data/exports/offline_mocap.csv"


class CalibrationConfig(BaseModel):
    output_path: str = "data/calibration/cameras.json"
    chessboard: list[int] = Field(default_factory=lambda: [9, 6])
    square_size_m: float = 0.025
    min_captures: int = 20
    auto_enabled_default: bool = False
    auto_poll_ms: int = 400
    auto_min_interval_ms: int = 1200
    auto_hold_ms: int = 450
    auto_motion_threshold_norm: float = 0.035
    auto_stable_threshold_norm: float = 0.010
    auto_min_board_area_norm: float = 0.008
    auto_pose_delta_threshold: float = 0.12

    @field_validator("chessboard")
    @classmethod
    def _validate_chessboard(cls, value: list[int]) -> list[int]:
        if len(value) != 2:
            raise ValueError("chessboard must contain [cols, rows]")
        return value


class CalibrationSyncConfig(BaseModel):
    adaptive_enabled: bool = True
    min_latency_ms: int = 120
    max_latency_ms: int = 450
    jitter_factor: float = 2.0
    history_size: int = 60
    fps_downshift_failure_streak: int = 3


class OfflineSource(BaseModel):
    id: str
    path: str
    enabled: bool = True


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    triangulation: TriangulationConfig = Field(default_factory=TriangulationConfig)
    osc: OscConfig = Field(default_factory=OscConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    calibration_sync: CalibrationSyncConfig = Field(default_factory=CalibrationSyncConfig)
    offline_sources: list[OfflineSource] = Field(default_factory=list)

    def calibration_path(self) -> Path:
        return Path(self.calibration.output_path)

    def maybe_masked_dump(self, mask_token: bool = True) -> dict:
        data = self.model_dump()
        if mask_token:
            token = data["server"].get("token", "")
            if token:
                data["server"]["token"] = "*" * max(4, len(token))
        return data


class ConfigUpdate(BaseModel):
    server: Optional[ServerConfig] = None
    model: Optional[ModelConfig] = None
    runtime: Optional[RuntimeConfig] = None
    triangulation: Optional[TriangulationConfig] = None
    osc: Optional[OscConfig] = None
    ingest: Optional[IngestConfig] = None
    export: Optional[ExportConfig] = None
    calibration: Optional[CalibrationConfig] = None
    calibration_sync: Optional[CalibrationSyncConfig] = None
