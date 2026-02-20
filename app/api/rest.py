from __future__ import annotations

import socket
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from app.api.auth import require_http_token
from app.core.constants import COCO_JOINTS
from app.services.checkerboard_pdf import CheckerboardSpec, generate_checkerboard_pdf
from app.models.api import (
    CalibrationCaptureRequest,
    CalibrationStartRequest,
    CreateInviteRequest,
    RegisterCameraRequest,
    SessionActionResponse,
    UpdateCameraRequest,
)
from app.models.config import ConfigUpdate

router = APIRouter(prefix="/api")


def _runtime(request: Request):
    return request.app.state.runtime


def _detect_lan_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # No packets are sent; this picks the outbound interface IP.
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        if ip and not ip.startswith("127."):
            return ip
    except OSError:
        pass
    finally:
        sock.close()
    return "127.0.0.1"


def _resolve_public_host(configured_host: str) -> str:
    host = (configured_host or "").strip()
    if host in {"", "0.0.0.0", "::", "localhost", "127.0.0.1"}:
        return _detect_lan_ip()
    return host


@router.get("/config")
def get_config(request: Request):
    runtime = _runtime(request)
    require_http_token(request, runtime.config_store.config.server.token)
    return runtime.config_store.config.maybe_masked_dump(mask_token=True)


@router.put("/config")
def put_config(request: Request, payload: ConfigUpdate):
    runtime = _runtime(request)
    require_http_token(request, runtime.config_store.config.server.token)
    cfg = runtime.config_store.update(payload)
    # Rebuild dependent services on config change.
    runtime.capture_hub._heartbeat_timeout_s = cfg.ingest.heartbeat_timeout_s  # noqa: SLF001
    runtime.calibration_service.cfg = cfg
    runtime.session_manager.cfg = cfg
    return cfg.maybe_masked_dump(mask_token=True)


@router.get("/client/env")
def client_env(request: Request):
    runtime = _runtime(request)
    cfg = runtime.config_store.config.server
    host = _resolve_public_host(cfg.host)
    port = int(cfg.port)
    scheme = request.url.scheme
    port_suffix = f":{port}" if port not in (80, 443) else ""
    return {
        "origin": f"{scheme}://{host}{port_suffix}",
        "host": host,
        "port": port,
    }


@router.post("/session/start", response_model=SessionActionResponse)
def start_session(request: Request):
    runtime = _runtime(request)
    require_http_token(request, runtime.config_store.config.server.token)
    out = runtime.session_manager.start()
    return SessionActionResponse(**out)


@router.post("/session/stop", response_model=SessionActionResponse)
def stop_session(request: Request):
    runtime = _runtime(request)
    require_http_token(request, runtime.config_store.config.server.token)
    out = runtime.session_manager.stop()
    return SessionActionResponse(**out)


@router.get("/session/status")
def session_status(request: Request):
    runtime = _runtime(request)
    require_http_token(request, runtime.config_store.config.server.token)
    return runtime.session_manager.status()


@router.get("/cameras")
def cameras(request: Request):
    runtime = _runtime(request)
    require_http_token(request, runtime.config_store.config.server.token)
    health = runtime.capture_hub.health_snapshot()
    roster = runtime.camera_registry.list_records()
    out = {}
    for item in roster:
        cam_id = item["camera_id"]
        h = health.get(cam_id, {})
        out[cam_id] = {
            "connected": bool(h.get("connected", item.get("connected", False))),
            "last_seen": h.get("last_seen", item.get("last_seen", 0.0)),
            "seq": h.get("seq", 0),
        }
    for cam_id, h in health.items():
        if cam_id in out:
            continue
        out[cam_id] = {
            "connected": bool(h.get("connected", False)),
            "last_seen": h.get("last_seen", 0.0),
            "seq": h.get("seq", 0),
        }
    return out


@router.get("/cameras/roster")
def camera_roster(request: Request):
    runtime = _runtime(request)
    require_http_token(request, runtime.config_store.config.server.token)
    health = runtime.capture_hub.health_snapshot()
    roster = runtime.camera_registry.list_records()
    for item in roster:
        h = health.get(item["camera_id"], {})
        if h:
            item["connected"] = bool(h.get("connected", item["connected"]))
            item["last_seen"] = h.get("last_seen", item["last_seen"])
            item["seq"] = h.get("seq", 0)
        else:
            item["seq"] = 0
    return roster


@router.post("/cameras/invites")
def create_camera_invite(request: Request, payload: CreateInviteRequest):
    runtime = _runtime(request)
    require_http_token(request, runtime.config_store.config.server.token)
    invite = runtime.join_invites.create(
        issued_by="admin",
        ttl_s=payload.ttl_s,
        preset_label=payload.preset_label,
    )
    cfg = runtime.config_store.config.server
    host = _resolve_public_host(cfg.host)
    scheme = request.url.scheme
    port = f":{cfg.port}" if cfg.port not in (80, 443) else ""
    join_url = f"{scheme}://{host}{port}/join?ticket={quote(invite.ticket_id)}"
    return {
        "ticket_id": invite.ticket_id,
        "expires_at": invite.expires_at,
        "used": invite.used,
        "join_url": join_url,
    }


@router.get("/cameras/invites/{ticket_id}")
def get_camera_invite(request: Request, ticket_id: str):
    runtime = _runtime(request)
    require_http_token(request, runtime.config_store.config.server.token)
    invite = runtime.join_invites.get(ticket_id)
    if invite is None:
        raise HTTPException(status_code=404, detail="invite_not_found")
    return {
        "ticket_id": invite.ticket_id,
        "expires_at": invite.expires_at,
        "used": invite.used,
        "valid": runtime.join_invites.is_valid(ticket_id),
        "preset_label": invite.preset_label,
    }


@router.post("/cameras/register")
def register_camera(request: Request, payload: RegisterCameraRequest):
    runtime = _runtime(request)
    try:
        invite = runtime.join_invites.consume(payload.ticket_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    rec = runtime.camera_registry.upsert_from_device(
        device_uid=payload.device_uid,
        device_name=payload.device_name,
        platform=payload.platform,
        label=payload.preferred_label or invite.preset_label,
    )

    ws_proto = "wss" if request.url.scheme == "https" else "ws"
    ws_url = (
        f"{ws_proto}://{request.url.netloc}/ws/camera/"
        f"{quote(rec.camera_id)}?token={quote(rec.ws_token)}"
    )
    return {
        "ok": True,
        "camera_id": rec.camera_id,
        "label": rec.label,
        "ws_url": ws_url,
        "ws_token": rec.ws_token,
    }


@router.patch("/cameras/{camera_id}")
def update_camera(request: Request, camera_id: str, payload: UpdateCameraRequest):
    runtime = _runtime(request)
    require_http_token(request, runtime.config_store.config.server.token)
    updated = runtime.camera_registry.update(
        camera_id, label=payload.label, enabled=payload.enabled
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="camera_not_found")
    return updated


@router.delete("/cameras/{camera_id}")
def delete_camera(request: Request, camera_id: str):
    runtime = _runtime(request)
    require_http_token(request, runtime.config_store.config.server.token)
    ok = runtime.camera_registry.soft_delete(camera_id)
    if not ok:
        raise HTTPException(status_code=404, detail="camera_not_found")
    return {"ok": True}


@router.post("/calibration/start")
def calibration_start(request: Request, payload: CalibrationStartRequest):
    runtime = _runtime(request)
    require_http_token(request, runtime.config_store.config.server.token)
    try:
        return runtime.calibration_service.start(payload.camera_ids)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/calibration/capture")
def calibration_capture(request: Request, payload: CalibrationCaptureRequest | None = None):
    runtime = _runtime(request)
    require_http_token(request, runtime.config_store.config.server.token)
    try:
        mode = payload.mode if payload is not None else "manual"
        return runtime.calibration_service.capture(mode=mode)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/calibration/solve")
def calibration_solve(request: Request):
    runtime = _runtime(request)
    require_http_token(request, runtime.config_store.config.server.token)
    try:
        return runtime.calibration_service.solve()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/calibration/report")
def calibration_report(request: Request):
    runtime = _runtime(request)
    require_http_token(request, runtime.config_store.config.server.token)
    return runtime.calibration_service.report()


@router.get("/calibration/readiness")
def calibration_readiness(request: Request):
    runtime = _runtime(request)
    require_http_token(request, runtime.config_store.config.server.token)
    return runtime.calibration_service.readiness()


@router.get("/calibration/checkerboard.pdf")
def calibration_checkerboard_pdf(request: Request):
    runtime = _runtime(request)
    require_http_token(request, runtime.config_store.config.server.token)
    cfg = runtime.config_store.config.calibration
    try:
        cols, rows = cfg.chessboard
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="invalid chessboard config") from exc

    try:
        spec = CheckerboardSpec(
            inside_cols=int(cols),
            inside_rows=int(rows),
            square_size_m=float(cfg.square_size_m),
            paper="A4",
        )
        payload = generate_checkerboard_pdf(spec)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    mm = int(round(spec.square_size_mm))
    filename = f"checkerboard_{spec.inside_cols}x{spec.inside_rows}_{mm}mm_a4.pdf"
    return Response(
        content=payload,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


@router.get("/preview/pose3d")
def preview_pose3d(request: Request):
    runtime = _runtime(request)
    require_http_token(request, runtime.config_store.config.server.token)
    status = runtime.session_manager.status()
    return {
        "timestamp": status["last_timestamp"],
        "joints": status["last_pose3d"],
        "schema": "coco17",
        "joint_names": COCO_JOINTS,
        "confidences": status.get("last_joint_confidences", {}),
        "joint_states": status.get("last_joint_states", {}),
        "metrics": {
            "valid_joints": status.get("valid_joints", 0),
            "active_cameras": status.get("active_cameras", 0),
            "loop_fps": status.get("loop_fps", 0.0),
            "loop_ms": status.get("loop_ms", 0.0),
            "running": status.get("running", False),
            "joint_conf_avg": status.get("joint_conf_avg"),
            "dropped_cycles": status.get("dropped_cycles", 0),
        },
    }


@router.get("/recordings/status")
def recordings_status(request: Request):
    runtime = _runtime(request)
    require_http_token(request, runtime.config_store.config.server.token)
    return runtime.recording_manager.status()


@router.post("/recordings/start")
def recordings_start(request: Request):
    runtime = _runtime(request)
    require_http_token(request, runtime.config_store.config.server.token)
    active = runtime.session_manager.status().get("active_cameras", 0)
    out = runtime.recording_manager.start(active_cameras=active)
    if int(active) <= 0:
        out["warning"] = "recording_with_0_active_cameras"
    return out


@router.post("/recordings/stop")
def recordings_stop(request: Request):
    runtime = _runtime(request)
    require_http_token(request, runtime.config_store.config.server.token)
    return runtime.recording_manager.stop()


@router.get("/recordings")
def recordings_list(request: Request):
    runtime = _runtime(request)
    require_http_token(request, runtime.config_store.config.server.token)
    return runtime.recording_manager.list_clips()


@router.post("/recordings/{clip_id}/export")
def recordings_export(request: Request, clip_id: str):
    runtime = _runtime(request)
    require_http_token(request, runtime.config_store.config.server.token)
    try:
        return runtime.recording_manager.export_clip(clip_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc.args[0])) from exc


@router.get("/preview/camera/{camera_id}")
def preview_camera(request: Request, camera_id: str):
    runtime = _runtime(request)
    require_http_token(request, runtime.config_store.config.server.token)
    jpg = runtime.capture_hub.latest_jpeg(camera_id)
    if jpg is None:
        raise HTTPException(status_code=404, detail="no frame")
    return Response(content=jpg, media_type="image/jpeg")


@router.post("/offline/export")
def offline_export_placeholder(request: Request):
    runtime = _runtime(request)
    require_http_token(request, runtime.config_store.config.server.token)
    from app.core.offline import OfflineProcessor

    processor = OfflineProcessor(runtime.config_store.config)
    return processor.run_export()
