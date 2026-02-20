from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.calibration import CalibrationService
from app.core.capture import CaptureHub
from app.core.events import EventBus
from app.core.session import SessionManager
from app.services.camera_registry import CameraRegistry
from app.services.config_store import ConfigStore
from app.services.join_invites import JoinInviteService
from app.services.recording_manager import RecordingManager


@dataclass
class RuntimeContext:
    config_store: ConfigStore
    capture_hub: CaptureHub
    calibration_service: CalibrationService
    session_manager: SessionManager
    event_bus: EventBus
    camera_registry: CameraRegistry
    join_invites: JoinInviteService
    recording_manager: RecordingManager


def build_runtime(config_path: Path) -> RuntimeContext:
    config_store = ConfigStore(config_path)
    cfg = config_store.config
    event_bus = EventBus()
    capture_hub = CaptureHub(cfg.ingest.heartbeat_timeout_s)
    calibration_service = CalibrationService(
        cfg,
        capture_hub,
        session_state_path=cfg.persistence.calibration_session_path,
    )
    recording_manager = RecordingManager()
    session_manager = SessionManager(cfg, capture_hub, event_bus, recording_manager)
    camera_registry = CameraRegistry(state_path=cfg.persistence.camera_registry_path)
    join_invites = JoinInviteService(
        default_ttl_s=120,
        state_path=cfg.persistence.invites_path,
    )
    return RuntimeContext(
        config_store=config_store,
        capture_hub=capture_hub,
        calibration_service=calibration_service,
        session_manager=session_manager,
        event_bus=event_bus,
        camera_registry=camera_registry,
        join_invites=join_invites,
        recording_manager=recording_manager,
    )
