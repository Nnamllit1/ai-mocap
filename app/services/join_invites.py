from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from app.services.state_io import load_json, save_json_atomic


@dataclass
class JoinInvite:
    ticket_id: str
    created_at: float
    expires_at: float
    used: bool
    max_uses: int
    uses: int
    issued_by: str
    preset_label: str | None = None


class JoinInviteService:
    def __init__(self, default_ttl_s: int = 120, state_path: str | Path | None = None):
        self.default_ttl_s = default_ttl_s
        self._invites: Dict[str, JoinInvite] = {}
        self._lock = threading.Lock()
        self._state_path = Path(state_path) if state_path else None
        self._load_state()

    @staticmethod
    def _to_payload(invite: JoinInvite) -> dict:
        return {
            "ticket_id": invite.ticket_id,
            "created_at": float(invite.created_at),
            "expires_at": float(invite.expires_at),
            "used": bool(invite.used),
            "max_uses": int(invite.max_uses),
            "uses": int(invite.uses),
            "issued_by": str(invite.issued_by),
            "preset_label": invite.preset_label,
        }

    @staticmethod
    def _from_payload(payload: dict) -> JoinInvite:
        return JoinInvite(
            ticket_id=str(payload["ticket_id"]),
            created_at=float(payload["created_at"]),
            expires_at=float(payload["expires_at"]),
            used=bool(payload.get("used", False)),
            max_uses=int(payload.get("max_uses", 1)),
            uses=int(payload.get("uses", 0)),
            issued_by=str(payload.get("issued_by", "unknown")),
            preset_label=payload.get("preset_label"),
        )

    def _prune_expired_locked(self, now: float) -> None:
        stale_ids = [
            ticket_id
            for ticket_id, invite in self._invites.items()
            if now >= float(invite.expires_at)
        ]
        for ticket_id in stale_ids:
            self._invites.pop(ticket_id, None)

    def _load_state(self) -> None:
        if self._state_path is None:
            return
        payload = load_json(self._state_path, default={"invites": []})
        invites = payload.get("invites", []) if isinstance(payload, dict) else []
        now = time.time()
        with self._lock:
            self._invites.clear()
            for entry in invites:
                try:
                    invite = self._from_payload(entry)
                except Exception:  # noqa: BLE001
                    continue
                if now >= float(invite.expires_at):
                    continue
                self._invites[invite.ticket_id] = invite
            self._save_state_locked()

    def _save_state_locked(self) -> None:
        if self._state_path is None:
            return
        now = time.time()
        self._prune_expired_locked(now)
        payload = {"invites": [self._to_payload(invite) for invite in self._invites.values()]}
        save_json_atomic(self._state_path, payload)

    def create(self, issued_by: str, ttl_s: int | None = None, preset_label: str | None = None) -> JoinInvite:
        now = time.time()
        ticket_id = f"join_{secrets.token_urlsafe(6)}"
        ttl = int(ttl_s or self.default_ttl_s)
        invite = JoinInvite(
            ticket_id=ticket_id,
            created_at=now,
            expires_at=now + ttl,
            used=False,
            max_uses=1,
            uses=0,
            issued_by=issued_by,
            preset_label=preset_label,
        )
        with self._lock:
            self._invites[ticket_id] = invite
            self._save_state_locked()
        return invite

    def get(self, ticket_id: str) -> Optional[JoinInvite]:
        with self._lock:
            invite = self._invites.get(ticket_id)
            if invite is None:
                return None
            return JoinInvite(**invite.__dict__)

    def is_valid(self, ticket_id: str) -> bool:
        invite = self.get(ticket_id)
        if invite is None:
            return False
        now = time.time()
        if invite.used:
            return False
        if invite.uses >= invite.max_uses:
            return False
        if now >= invite.expires_at:
            return False
        return True

    def consume(self, ticket_id: str) -> JoinInvite:
        with self._lock:
            invite = self._invites.get(ticket_id)
            if invite is None:
                raise ValueError("ticket_not_found")
            now = time.time()
            if invite.used or invite.uses >= invite.max_uses:
                raise ValueError("ticket_used")
            if now >= invite.expires_at:
                raise ValueError("ticket_expired")
            invite.uses += 1
            invite.used = True
            self._save_state_locked()
            return JoinInvite(**invite.__dict__)
