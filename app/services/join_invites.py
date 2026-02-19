from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional


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
    def __init__(self, default_ttl_s: int = 120):
        self.default_ttl_s = default_ttl_s
        self._invites: Dict[str, JoinInvite] = {}
        self._lock = threading.Lock()

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
            return JoinInvite(**invite.__dict__)
