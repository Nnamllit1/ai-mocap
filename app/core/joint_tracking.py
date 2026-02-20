from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from app.core.constants import TRACKING_BONE_EDGES


@dataclass
class JointCacheItem:
    xyz: np.ndarray
    confidence: float
    timestamp: float


class JointStateTracker:
    def __init__(
        self,
        hold_ms: int = 250,
        max_jump_m: float = 0.35,
        jump_reject_conf: float = 0.85,
        bone_length_guard_enabled: bool = True,
        bone_length_soft_rel_tol: float = 0.15,
        bone_length_hard_rel_tol: float = 0.35,
        bone_length_ema_alpha: float = 0.2,
        bone_length_learn_conf: float = 0.65,
    ):
        self.hold_ms = max(0, int(hold_ms))
        self.max_jump_m = max(0.0, float(max_jump_m))
        self.jump_reject_conf = max(0.0, min(1.0, float(jump_reject_conf)))
        self.bone_length_guard_enabled = bool(bone_length_guard_enabled)
        self.bone_length_soft_rel_tol = max(0.0, float(bone_length_soft_rel_tol))
        self.bone_length_hard_rel_tol = max(
            self.bone_length_soft_rel_tol, float(bone_length_hard_rel_tol)
        )
        self.bone_length_ema_alpha = max(
            0.0, min(1.0, float(bone_length_ema_alpha))
        )
        self.bone_length_learn_conf = max(
            0.0, min(1.0, float(bone_length_learn_conf))
        )
        self._cache: Dict[int, JointCacheItem] = {}
        self._bone_len_ema: Dict[tuple[int, int], float] = {}
        self._last_bone_guard_clamped_count: int = 0
        self._last_bone_guard_rejected_count: int = 0

    def reset(self) -> None:
        self._cache.clear()
        self._bone_len_ema.clear()
        self._last_bone_guard_clamped_count = 0
        self._last_bone_guard_rejected_count = 0

    @property
    def last_bone_guard_clamped_count(self) -> int:
        return int(self._last_bone_guard_clamped_count)

    @property
    def last_bone_guard_rejected_count(self) -> int:
        return int(self._last_bone_guard_rejected_count)

    def _get_fresh_cached_point(
        self, joint_idx: int, now: float
    ) -> tuple[np.ndarray, float, int] | None:
        prev = self._cache.get(joint_idx)
        if prev is None:
            return None
        age_ms = int(max(0.0, (now - float(prev.timestamp)) * 1000.0))
        if age_ms > self.hold_ms:
            return None
        return np.array(prev.xyz, dtype=float), float(prev.confidence), age_ms

    @staticmethod
    def _joint_conf(entry: dict) -> float:
        return float(entry.get("confidence", 0.0))

    @staticmethod
    def _joint_state(entry: dict) -> str:
        return str(entry.get("state", "measured"))

    @staticmethod
    def _joint_point(entry: dict) -> np.ndarray:
        return np.array(entry["xyz"], dtype=float)

    def _choose_moving_joint(self, a: int, b: int, out: dict[int, dict]) -> int:
        conf_a = self._joint_conf(out[a])
        conf_b = self._joint_conf(out[b])
        if conf_a < conf_b:
            return a
        if conf_b < conf_a:
            return b
        # Tie: use directed distal endpoint from edge ordering.
        return b

    def _clamp_edge_length(
        self,
        edge: tuple[int, int],
        out: dict[int, dict],
        move_idx: int,
        min_len: float,
        max_len: float,
    ) -> bool:
        a, b = edge
        anchor_idx = b if move_idx == a else a
        anchor = self._joint_point(out[anchor_idx])
        moving = self._joint_point(out[move_idx])
        vec = moving - anchor
        dist = float(np.linalg.norm(vec))
        if not np.isfinite(dist):
            return False

        target = min(max(dist, float(min_len)), float(max_len))
        if abs(target - dist) <= 1e-9:
            return False

        if dist > 1e-6:
            direction = vec / dist
        else:
            prev_move = self._cache.get(move_idx)
            prev_anchor = self._cache.get(anchor_idx)
            if prev_move is not None and prev_anchor is not None:
                prev_vec = np.array(prev_move.xyz, dtype=float) - np.array(
                    prev_anchor.xyz, dtype=float
                )
                prev_dist = float(np.linalg.norm(prev_vec))
                direction = prev_vec / prev_dist if prev_dist > 1e-6 else None
            else:
                direction = None
            if direction is None:
                direction = np.array([1.0, 0.0, 0.0], dtype=float)

        out[move_idx]["xyz"] = np.array(anchor + (direction * target), dtype=float)
        if self._joint_state(out[move_idx]) == "held":
            out[move_idx]["age_ms"] = 0
        return True

    def _apply_bone_guard(self, now: float, out: dict[int, dict]) -> tuple[int, int]:
        if not self.bone_length_guard_enabled:
            return 0, 0

        clamped_count = 0
        rejected_count = 0
        soft_tol = float(self.bone_length_soft_rel_tol)
        hard_tol = float(self.bone_length_hard_rel_tol)

        for edge in TRACKING_BONE_EDGES:
            a, b = edge
            if a not in out or b not in out:
                continue

            pa = self._joint_point(out[a])
            pb = self._joint_point(out[b])
            dist = float(np.linalg.norm(pb - pa))
            if not np.isfinite(dist) or dist <= 1e-9:
                continue

            base_len = float(self._bone_len_ema.get(edge, 0.0))
            if base_len > 1e-9:
                rel_dev = abs(dist - base_len) / max(base_len, 1e-6)
                min_len = base_len * (1.0 - soft_tol)
                max_len = base_len * (1.0 + soft_tol)
                move_idx = self._choose_moving_joint(a, b, out)
                conf_a = self._joint_conf(out[a])
                conf_b = self._joint_conf(out[b])
                state_a = self._joint_state(out[a])
                state_b = self._joint_state(out[b])

                if rel_dev > soft_tol and rel_dev <= hard_tol:
                    if self._clamp_edge_length(edge, out, move_idx, min_len, max_len):
                        clamped_count += 1
                elif rel_dev > hard_tol:
                    weak_evidence = (
                        min(conf_a, conf_b) < self.jump_reject_conf
                        or state_a == "single_view"
                        or state_b == "single_view"
                    )
                    if weak_evidence:
                        cached = self._get_fresh_cached_point(move_idx, now)
                        if cached is not None:
                            cached_xyz, cached_conf, age_ms = cached
                            out[move_idx] = {
                                "xyz": cached_xyz,
                                "confidence": cached_conf,
                                "state": "held",
                                "age_ms": age_ms,
                            }
                            rejected_count += 1
                        elif self._clamp_edge_length(
                            edge, out, move_idx, min_len, max_len
                        ):
                            clamped_count += 1
                    elif self._clamp_edge_length(edge, out, move_idx, min_len, max_len):
                        clamped_count += 1

            # Learn baseline from stable measured pairs only.
            conf_a = self._joint_conf(out[a])
            conf_b = self._joint_conf(out[b])
            if (
                self._joint_state(out[a]) == "measured"
                and self._joint_state(out[b]) == "measured"
                and conf_a >= self.bone_length_learn_conf
                and conf_b >= self.bone_length_learn_conf
            ):
                pa = self._joint_point(out[a])
                pb = self._joint_point(out[b])
                current_len = float(np.linalg.norm(pb - pa))
                if np.isfinite(current_len) and current_len > 1e-6:
                    prior = self._bone_len_ema.get(edge)
                    if prior is None:
                        self._bone_len_ema[edge] = current_len
                    else:
                        alpha = self.bone_length_ema_alpha
                        self._bone_len_ema[edge] = ((1.0 - alpha) * prior) + (
                            alpha * current_len
                        )

        return clamped_count, rejected_count

    def stabilize(
        self,
        timestamp: float,
        measured_pose3d: Dict[int, np.ndarray],
        measured_confidences: Dict[int, float],
        measured_states: Dict[int, str] | None = None,
    ) -> dict[int, dict]:
        out: dict[int, dict] = {}
        now = float(timestamp)

        self._last_bone_guard_clamped_count = 0
        self._last_bone_guard_rejected_count = 0
        measured_states = measured_states or {}
        for joint_idx, xyz in measured_pose3d.items():
            conf = float(measured_confidences.get(joint_idx, 0.0))
            point = np.array(xyz, dtype=float)
            prev = self._cache.get(joint_idx)
            if prev is not None:
                jump_m = float(np.linalg.norm(point - np.array(prev.xyz, dtype=float)))
                if jump_m > self.max_jump_m and conf < self.jump_reject_conf:
                    age_ms = int(max(0.0, (now - float(prev.timestamp)) * 1000.0))
                    if age_ms <= self.hold_ms:
                        out[joint_idx] = {
                            "xyz": np.array(prev.xyz, dtype=float),
                            "confidence": float(prev.confidence),
                            "state": "held",
                            "age_ms": age_ms,
                        }
                    continue

            state = str(measured_states.get(joint_idx, "measured"))
            out[joint_idx] = {
                "xyz": point,
                "confidence": conf,
                "state": state,
                "age_ms": 0,
            }

        clamped_count, rejected_count = self._apply_bone_guard(now, out)
        self._last_bone_guard_clamped_count = clamped_count
        self._last_bone_guard_rejected_count = rejected_count

        for joint_idx, entry in out.items():
            if self._joint_state(entry) == "held":
                continue
            self._cache[joint_idx] = JointCacheItem(
                xyz=np.array(entry["xyz"], dtype=float),
                confidence=float(entry["confidence"]),
                timestamp=now,
            )

        for joint_idx, item in self._cache.items():
            if joint_idx in out:
                continue
            age_ms = int(max(0.0, (now - float(item.timestamp)) * 1000.0))
            if age_ms > self.hold_ms:
                continue
            out[joint_idx] = {
                "xyz": np.array(item.xyz, dtype=float),
                "confidence": float(item.confidence),
                "state": "held",
                "age_ms": age_ms,
            }

        return out
