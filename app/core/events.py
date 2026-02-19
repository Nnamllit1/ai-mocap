from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np


@dataclass
class FrameEvent:
    camera_id: str
    timestamp: float
    frame: np.ndarray


@dataclass
class Pose3DEvent:
    timestamp: float
    joints: Dict[int, np.ndarray]


@dataclass
class SessionStatusEvent:
    running: bool
    active_cameras: int
    valid_joints: int
    message: str


class EventBus:
    def __init__(self):
        self._subs: Dict[str, List[Callable]] = {}

    def subscribe(self, event_name: str, callback: Callable) -> None:
        self._subs.setdefault(event_name, []).append(callback)

    def publish(self, event_name: str, payload) -> None:
        for callback in self._subs.get(event_name, []):
            callback(payload)
