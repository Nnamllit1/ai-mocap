from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.rest import router as rest_router
from app.api.web import router as web_router
from app.api.ws import router as ws_router
from app.services.runtime import build_runtime


def create_app() -> FastAPI:
    app = FastAPI(title="Mocap Web Portal", version="0.1.0")
    runtime = build_runtime(Path("configs/default.yaml"))
    app.state.runtime = runtime
    app.state.templates = Jinja2Templates(directory="app/web/templates")

    app.mount("/static", StaticFiles(directory="app/web/static"), name="static")
    app.include_router(web_router)
    app.include_router(rest_router)
    app.include_router(ws_router)
    return app


app = create_app()
