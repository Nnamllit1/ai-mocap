from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def load_json(path: str | Path, default: T, validator: Callable[[Any], T] | None = None) -> T:
    p = Path(path)
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default
    if validator is None:
        return payload if payload is not None else default
    try:
        return validator(payload)
    except Exception:  # noqa: BLE001
        return default


def save_json_atomic(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f"{p.suffix}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(p))
