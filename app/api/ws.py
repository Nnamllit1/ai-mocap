from __future__ import annotations

import json
import time

from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect

from app.api.auth import require_ws_token_flexible

router = APIRouter()


@router.websocket("/ws/camera/{camera_id}")
async def ws_camera_ingest(websocket: WebSocket, camera_id: str):
    runtime = websocket.app.state.runtime
    master_token = runtime.config_store.config.server.token

    def _validator(token: str) -> bool:
        return runtime.camera_registry.validate_ws_token(camera_id, token)

    try:
        await require_ws_token_flexible(
            websocket,
            master_token=master_token,
            token_validator=_validator,
        )
    except RuntimeError:
        return

    await websocket.accept()
    runtime.camera_registry.set_connected(camera_id, True)
    await websocket.send_json(
        {
            "type": "ack",
            "camera_id": camera_id,
            "jpeg_quality": runtime.config_store.config.ingest.jpeg_quality,
            "fps_cap": runtime.config_store.config.ingest.client_fps_cap,
        }
    )

    last_hint = 0.0
    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"] is not None:
                payload = message["bytes"]
                if len(payload) <= runtime.config_store.config.ingest.max_frame_size:
                    ok = runtime.capture_hub.ingest_jpeg(camera_id, payload)
                    if not ok:
                        await websocket.send_json({"type": "warn", "reason": "decode_failed"})
            elif "text" in message and message["text"] is not None:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    data = {"type": "unknown"}
                if data.get("type") == "heartbeat":
                    runtime.capture_hub.heartbeat(camera_id)
                    runtime.camera_registry.heartbeat(camera_id)
            now = time.time()
            if now - last_hint >= 2.0:
                last_hint = now
                await websocket.send_json(
                    {
                        "type": "hint",
                        "jpeg_quality": runtime.config_store.config.ingest.jpeg_quality,
                        "fps_cap": runtime.config_store.config.ingest.client_fps_cap,
                        "recommended_fps_cap": runtime.calibration_service.session.recommended_fps_cap,
                    }
                )
    except WebSocketDisconnect:
        runtime.capture_hub.mark_disconnected(camera_id)
        runtime.camera_registry.set_connected(camera_id, False)
