from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.api.auth import resolve_token_from_request

router = APIRouter()


def _templates(request: Request) -> Jinja2Templates:
    return request.app.state.templates


def _require_or_redirect(request: Request):
    runtime = request.app.state.runtime
    expected = runtime.config_store.config.server.token
    token = resolve_token_from_request(request)
    if expected and token != expected:
        return RedirectResponse(url="/login", status_code=302)
    return None


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return _templates(request).TemplateResponse(
        request, "login.html", {"title": "Login", "error": ""}
    )


@router.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request):
    form = await request.form()
    token = str(form.get("token", ""))
    expected = request.app.state.runtime.config_store.config.server.token
    if expected and token != expected:
        return _templates(request).TemplateResponse(
            request, "login.html", {"title": "Login", "error": "Invalid token"}
        )
    response = RedirectResponse(url="/", status_code=302)
    response.set_cookie("portal_token", token, httponly=False, samesite="lax")
    return response


@router.get("/", response_class=HTMLResponse)
def dashboard_page(request: Request):
    guard = _require_or_redirect(request)
    if guard:
        return guard
    runtime = request.app.state.runtime
    token = resolve_token_from_request(request)
    return _templates(request).TemplateResponse(
        request,
        "dashboard.html",
        {
            "title": "Dashboard",
            "token": token,
            "status": runtime.session_manager.status(),
            "roster": runtime.camera_registry.list_records(),
            "calibration": runtime.config_store.config.calibration,
        },
    )


@router.get("/cameras", response_class=HTMLResponse)
def cameras_page(request: Request):
    guard = _require_or_redirect(request)
    if guard:
        return guard
    token = resolve_token_from_request(request)
    return _templates(request).TemplateResponse(
        request,
        "cameras.html",
        {
            "title": "Cameras",
            "token": token,
            "cameras": request.app.state.runtime.camera_registry.list_records(),
        },
    )


@router.get("/calibration", response_class=HTMLResponse)
def calibration_page(request: Request):
    guard = _require_or_redirect(request)
    if guard:
        return guard
    token = resolve_token_from_request(request)
    report = request.app.state.runtime.calibration_service.report()
    return _templates(request).TemplateResponse(
        request,
        "calibration.html",
        {"title": "Calibration", "token": token, "report": report},
    )


@router.get("/config", response_class=HTMLResponse)
def config_page(request: Request):
    guard = _require_or_redirect(request)
    if guard:
        return guard
    token = resolve_token_from_request(request)
    cfg = request.app.state.runtime.config_store.config.model_dump()
    return _templates(request).TemplateResponse(
        request,
        "config.html",
        {"title": "Config", "token": token, "config": cfg},
    )


@router.get("/preview", response_class=HTMLResponse)
def preview_page(request: Request):
    guard = _require_or_redirect(request)
    if guard:
        return guard
    token = resolve_token_from_request(request)
    return _templates(request).TemplateResponse(
        request,
        "preview.html",
        {
            "title": "Preview",
            "token": token,
            "cameras": request.app.state.runtime.camera_registry.list_records(),
        },
    )


@router.get("/recording", response_class=HTMLResponse)
def recording_page(request: Request):
    guard = _require_or_redirect(request)
    if guard:
        return guard
    token = resolve_token_from_request(request)
    return _templates(request).TemplateResponse(
        request,
        "recording.html",
        {
            "title": "Recording",
            "token": token,
            "cameras": request.app.state.runtime.camera_registry.list_records(),
        },
    )


@router.get("/guides", response_class=HTMLResponse)
def guides_page(request: Request):
    guard = _require_or_redirect(request)
    if guard:
        return guard
    token = resolve_token_from_request(request)
    return _templates(request).TemplateResponse(
        request,
        "guides.html",
        {
            "title": "Guides",
            "token": token,
        },
    )


@router.get("/join", response_class=HTMLResponse)
def join_page(request: Request):
    ticket = request.query_params.get("ticket", "")
    if not ticket:
        return RedirectResponse(url="/login", status_code=302)
    invite = request.app.state.runtime.join_invites.get(ticket)
    valid = request.app.state.runtime.join_invites.is_valid(ticket)
    return _templates(request).TemplateResponse(
        request,
        "join.html",
        {
            "title": "Join Camera",
            "ticket_id": ticket,
            "valid_ticket": bool(valid),
            "preset_label": invite.preset_label if invite else "",
        },
    )


@router.get("/join/{camera_id}", response_class=HTMLResponse)
def legacy_join_camera_page(request: Request, camera_id: str):
    token = resolve_token_from_request(request) or request.query_params.get("token", "")
    expected = request.app.state.runtime.config_store.config.server.token
    if expected and token != expected:
        return RedirectResponse(url="/login", status_code=302)
    invite = request.app.state.runtime.join_invites.create(
        issued_by="legacy_route",
        ttl_s=120,
        preset_label=camera_id,
    )
    return RedirectResponse(url=f"/join?ticket={quote(invite.ticket_id)}", status_code=302)


@router.get("/logout")
def logout():
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie("portal_token")
    return response


@router.get("/mobile-nav")
def mobile_nav(request: Request):
    return _templates(request).TemplateResponse(
        request,
        "partials/mobile_nav.html",
        {"title": "Nav"},
    )
