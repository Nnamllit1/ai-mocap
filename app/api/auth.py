from __future__ import annotations

from fastapi import HTTPException, Request, WebSocket


def resolve_token_from_request(request: Request) -> str:
    token = request.query_params.get("token")
    if token:
        return token
    token = request.cookies.get("portal_token")
    if token:
        return token
    token = request.headers.get("x-access-token")
    if token:
        return token
    return ""


def require_http_token(request: Request, expected_token: str) -> str:
    token = resolve_token_from_request(request)
    if not expected_token or token == expected_token:
        return token
    raise HTTPException(status_code=401, detail="invalid token")


async def require_ws_token(websocket: WebSocket, expected_token: str) -> None:
    token = websocket.query_params.get("token", "")
    if expected_token and token != expected_token:
        await websocket.close(code=4401, reason="invalid token")
        raise RuntimeError("invalid token")


async def require_ws_token_flexible(
    websocket: WebSocket,
    *,
    master_token: str,
    token_validator,
) -> None:
    token = websocket.query_params.get("token", "")
    if master_token and token == master_token:
        return
    if token_validator(token):
        return
    await websocket.close(code=4401, reason="invalid token")
    raise RuntimeError("invalid token")
