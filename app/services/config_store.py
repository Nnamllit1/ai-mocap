from __future__ import annotations

from pathlib import Path

import yaml

from app.models.config import AppConfig, ConfigUpdate


class ConfigStore:
    def __init__(self, path: Path):
        self.path = path
        self.config = self._load_or_create()

    def _load_or_create(self) -> AppConfig:
        if self.path.exists():
            payload = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
            return AppConfig.model_validate(payload)
        cfg = AppConfig()
        self.save(cfg)
        return cfg

    def save(self, cfg: AppConfig) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            yaml.safe_dump(cfg.model_dump(), sort_keys=False),
            encoding="utf-8",
        )
        self.config = cfg

    def update(self, update: ConfigUpdate) -> AppConfig:
        merged = self.config.model_copy(update=update.model_dump(exclude_none=True))
        self.save(merged)
        return merged
