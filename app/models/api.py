from typing import Literal

from pydantic import BaseModel, Field


class SessionActionResponse(BaseModel):
    ok: bool
    message: str


class CalibrationStartRequest(BaseModel):
    camera_ids: list[str] = Field(default_factory=list)


class CalibrationCaptureRequest(BaseModel):
    mode: Literal["manual", "auto"] = "manual"


class CreateInviteRequest(BaseModel):
    ttl_s: int = 120
    preset_label: str | None = None


class RegisterCameraRequest(BaseModel):
    ticket_id: str
    device_uid: str
    device_name: str = "unknown-device"
    platform: str = "unknown"
    preferred_label: str | None = None


class UpdateCameraRequest(BaseModel):
    label: str | None = None
    enabled: bool | None = None
