from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal
from urllib.parse import unquote

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class SettingsError(RuntimeError):
    """Raised when the application configuration cannot be loaded safely."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelSettings(StrictModel):
    key: str = Field(min_length=1, max_length=120)
    label: str = Field(min_length=1, max_length=120)
    provider_model: str = Field(min_length=1, max_length=120)
    supports_images: bool
    supports_streaming: bool
    enabled: bool

    @field_validator("key", "label", "provider_model", mode="before")
    @classmethod
    def strip_nonempty_model_strings(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class LlmSettings(StrictModel):
    provider: Literal["openai"]
    default_model_key: str = Field(min_length=1, max_length=120)
    models: list[ModelSettings]

    @field_validator("default_model_key", mode="before")
    @classmethod
    def strip_default_key(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_models(self) -> "LlmSettings":
        keys = [model.key for model in self.models]
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("llm.models keys must be non-empty and unique")
        enabled = {model.key: model for model in self.models if model.enabled}
        if self.default_model_key not in enabled:
            raise ValueError("llm.default_model_key must name an enabled model")
        unsupported = [model.key for model in enabled.values() if not model.supports_streaming]
        if unsupported:
            raise ValueError(f"enabled models must support streaming: {', '.join(unsupported)}")
        return self

    @property
    def enabled_models(self) -> dict[str, ModelSettings]:
        return {model.key: model for model in self.models if model.enabled}


class CompactionSettings(StrictModel):
    enabled: bool = False
    compact_threshold: int = Field(default=200000, ge=1000)


class ResponsesSettings(StrictModel):
    instructions: str = Field(max_length=100000)
    max_output_tokens: int = Field(ge=1, le=100000)
    store: Literal[True]
    compaction: CompactionSettings


class MessageSettings(StrictModel):
    max_text_length: int = Field(ge=1, le=1_000_000)


class ImageSettings(StrictModel):
    enabled: bool
    max_files: int = Field(ge=1, le=16)
    max_file_size_mb: int = Field(ge=1, le=100)
    max_dimension_px: int = Field(ge=64, le=8192)
    max_decoded_pixels: int = Field(ge=1, le=200_000_000)
    allowed_types: list[Literal["image/jpeg", "image/png", "image/webp"]]
    convert_to: Literal["webp"]
    webp_quality: int = Field(ge=1, le=100)
    remove_exif: Literal[True]

    @field_validator("allowed_types")
    @classmethod
    def mime_types_are_unique(cls, value: list[str]) -> list[str]:
        if not value or len(value) != len(set(value)):
            raise ValueError("images.allowed_types must be non-empty and unique")
        return value

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


class DatabaseSettings(StrictModel):
    path: str = Field(min_length=1)
    wal_mode: bool = True


class StorageSettings(StrictModel):
    upload_directory: str = Field(min_length=1)
    temp_directory: str = Field(min_length=1)


class RetentionSettings(StrictModel):
    history_days: int = Field(ge=1, le=7)
    cleanup_interval_hours: int = Field(ge=1, le=24)
    delete_remote_responses: bool = True


class ServerSettings(StrictModel):
    host: Literal["127.0.0.1"]
    port: int = Field(ge=1, le=65535)
    max_request_size_mb: int = Field(ge=1, le=200)

    @property
    def max_request_bytes(self) -> int:
        return self.max_request_size_mb * 1024 * 1024


class UiSettings(StrictModel):
    title: str = Field(min_length=1, max_length=120)
    ai_icon: str = "/favicon.svg"

    @field_validator("ai_icon", mode="before")
    @classmethod
    def validate_ai_icon(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        value = value.strip()
        decoded = unquote(value)
        if (
            not value.startswith("/")
            or value.startswith("//")
            or "\\" in decoded
            or "?" in decoded
            or "#" in decoded
            or any(part in {"", ".", ".."} for part in decoded.split("/")[1:])
        ):
            raise ValueError("ui.ai_icon must be a root-relative static image path")
        if Path(decoded).suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg"}:
            raise ValueError("ui.ai_icon must reference a supported image file")
        return value


class Settings(StrictModel):
    llm: LlmSettings
    responses: ResponsesSettings
    messages: MessageSettings
    images: ImageSettings
    database: DatabaseSettings
    storage: StorageSettings
    retention: RetentionSettings
    server: ServerSettings
    ui: UiSettings
    project_root: Path = Field(default=PROJECT_ROOT, exclude=True)

    @property
    def database_path(self) -> Path:
        return self._project_path(self.database.path)

    @property
    def uploads_path(self) -> Path:
        return self._project_path(self.storage.upload_directory)

    @property
    def tmp_path(self) -> Path:
        return self._project_path(self.storage.temp_directory)

    def _project_path(self, value: str) -> Path:
        candidate = (self.project_root / value).resolve()
        try:
            candidate.relative_to(self.project_root.resolve())
        except ValueError as exc:
            raise SettingsError(f"configured path leaves the project root: {value}") from exc
        return candidate

    def public_dict(self, *, api_key_available: bool) -> dict[str, object]:
        return {
            "app_title": self.ui.title,
            "ai_icon_url": self.ui.ai_icon,
            "provider": self.llm.provider,
            "default_model_key": self.llm.default_model_key,
            "llm_configured": api_key_available,
            "models": [
                {
                    "key": model.key,
                    "label": model.label,
                    "supports_images": model.supports_images,
                    "supports_streaming": model.supports_streaming,
                }
                for model in self.llm.models
                if model.enabled
            ],
            "images_enabled": self.images.enabled,
            "max_images": self.images.max_files,
            "max_file_size_mb": self.images.max_file_size_mb,
            "max_request_size_mb": self.server.max_request_size_mb,
            "max_text_length": self.messages.max_text_length,
            "history_days": self.retention.history_days,
        }


def load_settings(path: str | Path | None = None) -> Settings:
    raw_path = path or os.getenv("SIMPLE_CHAT_CONFIG") or PROJECT_ROOT / "config.toml"
    config_path = Path(raw_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.is_file():
        raise SettingsError(
            f"configuration file not found: {config_path}. "
            "Copy config.example.toml to config.toml before starting."
        )

    try:
        with config_path.open("rb") as source:
            data = tomllib.load(source)
        settings = Settings.model_validate(data)
        settings.project_root = PROJECT_ROOT
        # Resolve all configured paths now so unsafe values fail at startup.
        _ = settings.database_path, settings.uploads_path, settings.tmp_path
        return settings
    except (OSError, tomllib.TOMLDecodeError, ValidationError, SettingsError) as exc:
        if isinstance(exc, SettingsError):
            raise
        raise SettingsError(f"invalid configuration in {config_path}: {exc}") from exc
