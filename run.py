from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import sqlite3
import sys
import tomllib
import uuid
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote

import openai
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
MAX_CONVERSATIONS = 5
logger = logging.getLogger("simple_chat")


class SettingsError(RuntimeError):
    pass


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelSettings(StrictModel):
    target: str = Field(min_length=1, max_length=120)
    label: str = Field(min_length=1, max_length=120)

    @field_validator("target", "label", mode="before")
    @classmethod
    def strip_value(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class ResponseSettings(StrictModel):
    instructions: str = Field(max_length=100000)
    max_output_tokens: int = Field(ge=1, le=100000)


class MessageSettings(StrictModel):
    max_text_length: int = Field(default=100000, ge=1, le=1_000_000)


class ImageSettings(StrictModel):
    enabled: bool = True
    max_files: int = Field(default=4, ge=1, le=16)
    max_file_size_mb: int = Field(default=10, ge=1, le=100)
    max_dimension_px: int = Field(default=2048, ge=64, le=8192)
    max_decoded_pixels: int = Field(default=40_000_000, ge=1, le=200_000_000)
    webp_quality: int = Field(default=90, ge=1, le=100)

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


class DatabaseSettings(StrictModel):
    path: str = Field(default="data/chat.db", min_length=1)


class StorageSettings(StrictModel):
    upload_directory: str = Field(default="data/uploads", min_length=1)
    temp_directory: str = Field(default="data/tmp", min_length=1)


class ServerSettings(StrictModel):
    host: Literal["127.0.0.1"] = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    max_request_size_mb: int = Field(default=45, ge=1, le=200)

    @property
    def max_request_bytes(self) -> int:
        return self.max_request_size_mb * 1024 * 1024


class UiSettings(StrictModel):
    title: str = Field(default="Simple Chat", min_length=1, max_length=120)
    ai_icon: str = "/ai-icon.svg"

    @field_validator("ai_icon", mode="before")
    @classmethod
    def validate_icon(cls, value: object) -> object:
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
            or Path(decoded).suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg"}
        ):
            raise ValueError("ui.ai_icon must be a safe image path under web/")
        return value


class Settings(StrictModel):
    model: ModelSettings
    responses: ResponseSettings
    messages: MessageSettings = MessageSettings()
    images: ImageSettings = ImageSettings()
    database: DatabaseSettings = DatabaseSettings()
    storage: StorageSettings = StorageSettings()
    server: ServerSettings = ServerSettings()
    ui: UiSettings = UiSettings()
    root: Path = Field(default=ROOT, exclude=True)

    def project_path(self, value: str) -> Path:
        candidate = (self.root / value).resolve()
        try:
            candidate.relative_to(self.root.resolve())
        except ValueError as exc:
            raise SettingsError(f"configured path leaves project root: {value}") from exc
        return candidate

    @property
    def database_path(self) -> Path:
        return self.project_path(self.database.path)

    @property
    def uploads_path(self) -> Path:
        return self.project_path(self.storage.upload_directory)

    @property
    def temp_path(self) -> Path:
        return self.project_path(self.storage.temp_directory)

    def public_dict(self, api_key_available: bool) -> dict[str, object]:
        return {
            "app_title": self.ui.title,
            "ai_icon_url": self.ui.ai_icon,
            "current_model_label": self.model.label,
            "llm_configured": api_key_available,
            "images_enabled": self.images.enabled,
            "max_images": self.images.max_files,
            "max_file_size_mb": self.images.max_file_size_mb,
            "max_request_size_mb": self.server.max_request_size_mb,
            "max_text_length": self.messages.max_text_length,
            "max_conversations": MAX_CONVERSATIONS,
        }


def load_settings(path: str | Path | None = None) -> Settings:
    raw_path = path or os.getenv("SIMPLE_CHAT_CONFIG") or ROOT / "config.toml"
    config_path = Path(raw_path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not config_path.is_file():
        raise SettingsError(f"設定ファイルがありません: {config_path}")
    try:
        with config_path.open("rb") as source:
            settings = Settings.model_validate(tomllib.load(source))
        _ = settings.database_path, settings.uploads_path, settings.temp_path
        return settings
    except (OSError, tomllib.TOMLDecodeError, ValidationError, SettingsError) as exc:
        if isinstance(exc, SettingsError):
            raise
        raise SettingsError(f"設定ファイルが不正です: {exc}") from exc


@dataclass(frozen=True, slots=True)
class StagedImage:
    id: str
    original_name: str
    stored_name: str
    width: int
    height: int
    byte_size: int
    path: Path


@dataclass(frozen=True, slots=True)
class DeletionSnapshot:
    response_ids: tuple[str, ...]
    attachment_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StreamEvent:
    type: str
    text: str = ""
    response_id: str | None = None
    input_tokens: int = 0


class LLMError(Exception):
    def __init__(self, code: str, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


class AppError(Exception):
    def __init__(self, code: str, message: str, status: int = 400, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.retryable = retryable

    def payload(self) -> dict[str, object]:
        return {"error": {"code": self.code, "message": self.message, "retryable": self.retryable}}


class OpenAIService:
    def __init__(self, api_key: str) -> None:
        self.client = AsyncOpenAI(api_key=api_key)

    async def stream(
        self,
        *,
        model: str,
        text: str,
        image_paths: list[Path],
        previous_response_id: str | None,
        instructions: str,
        max_output_tokens: int,
    ) -> AsyncIterator[StreamEvent]:
        content: list[dict[str, str]] = []
        if text:
            content.append({"type": "input_text", "text": text})
        for path in image_paths:
            encoded = await asyncio.to_thread(base64.b64encode, path.read_bytes())
            content.append({"type": "input_image", "image_url": f"data:image/webp;base64,{encoded.decode('ascii')}"})
        arguments: dict[str, Any] = {
            "model": model,
            "input": [{"role": "user", "content": content}],
            "instructions": instructions,
            "max_output_tokens": max_output_tokens,
            "store": True,
            "stream": True,
        }
        if previous_response_id:
            arguments["previous_response_id"] = previous_response_id
        try:
            stream = await self.client.responses.create(**arguments)
            try:
                completed = False
                async for event in stream:
                    event_type = str(getattr(event, "type", ""))
                    if event_type == "response.created":
                        yield StreamEvent("created", response_id=getattr(getattr(event, "response", None), "id", None))
                    elif event_type == "response.output_text.delta":
                        yield StreamEvent("delta", text=str(getattr(event, "delta", "")))
                    elif event_type == "response.completed":
                        response = getattr(event, "response", None)
                        response_id = getattr(response, "id", None)
                        if getattr(response, "status", "completed") != "completed" or not response_id:
                            raise LLMError("upstream_error", "回答が正常に完了しませんでした。", True)
                        usage = getattr(response, "usage", None)
                        input_tokens = max(0, int(getattr(usage, "input_tokens", 0) or 0))
                        completed = True
                        yield StreamEvent("completed", response_id=response_id, input_tokens=input_tokens)
                    elif event_type in {"response.failed", "response.incomplete", "error"}:
                        raise LLMError("upstream_error", "回答が完了しませんでした。", True)
                    elif event_type in {"response.refusal.delta", "response.refusal.done"}:
                        raise LLMError("upstream_error", "モデルが回答を生成できませんでした。")
                if not completed:
                    raise LLMError("upstream_error", "回答ストリームが途中で終了しました。", True)
            finally:
                await stream.close()
        except LLMError:
            raise
        except openai.RateLimitError as exc:
            raise LLMError("rate_limited", "しばらく待ってから再試行してください。", True) from exc
        except openai.APITimeoutError as exc:
            raise LLMError("upstream_timeout", "応答がタイムアウトしました。", True) from exc
        except openai.APIConnectionError as exc:
            raise LLMError("upstream_unavailable", "OpenAIへ接続できませんでした。", True) from exc
        except (openai.BadRequestError, openai.NotFoundError) as exc:
            if previous_response_id and _is_reference_error(exc):
                raise LLMError("context_expired", "OpenAI側の会話コンテキストが失効しました。") from exc
            raise LLMError("upstream_error", "OpenAIがリクエストを受理できませんでした。") from exc
        except openai.APIError as exc:
            raise LLMError("upstream_error", "OpenAIでエラーが発生しました。", True) from exc

    async def delete_response(self, response_id: str) -> None:
        try:
            await self.client.responses.delete(response_id, timeout=10.0)
        except openai.NotFoundError:
            pass


def _is_reference_error(exc: Exception) -> bool:
    if isinstance(exc, openai.NotFoundError):
        return True
    body = getattr(exc, "body", None)
    marker = json.dumps(body, ensure_ascii=False).lower() if body is not None else ""
    return "previous_response" in marker or "response_id" in marker


class ChatDatabase:
    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def connect(self) -> Any:
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=15000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    model_target TEXT NOT NULL,
                    model_label TEXT NOT NULL,
                    latest_response_id TEXT,
                    context_tokens INTEGER NOT NULL DEFAULT 0 CHECK(context_tokens >= 0),
                    context_status TEXT NOT NULL DEFAULT 'active'
                        CHECK(context_status IN ('active', 'expired')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    response_id TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_messages_conversation
                    ON messages(conversation_id, created_at, id);
                CREATE TABLE IF NOT EXISTS attachments (
                    id TEXT PRIMARY KEY,
                    message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                    original_name TEXT NOT NULL,
                    stored_name TEXT NOT NULL UNIQUE,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    byte_size INTEGER NOT NULL
                );
                """
            )

    def list_conversations(self) -> list[dict[str, object]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM conversations ORDER BY updated_at DESC, id ASC LIMIT ?",
                (MAX_CONVERSATIONS,),
            ).fetchall()
        return [self._conversation(row) for row in rows]

    def detail(self, conversation_id: str) -> dict[str, object]:
        with self.connect() as db:
            row = db.execute("SELECT * FROM conversations WHERE id=?", (conversation_id,)).fetchone()
            if row is None:
                raise AppError("not_found", "対象が見つかりません。", 404)
            messages = db.execute(
                "SELECT * FROM messages WHERE conversation_id=? "
                "ORDER BY created_at, CASE role WHEN 'user' THEN 0 ELSE 1 END, id",
                (conversation_id,),
            ).fetchall()
            payloads = []
            for message in messages:
                attachments = db.execute(
                    "SELECT * FROM attachments WHERE message_id=? ORDER BY id", (message["id"],)
                ).fetchall()
                payloads.append(self._message(message, attachments))
        return {"conversation": self._conversation(row), "messages": payloads}

    def generation_target(self, conversation_id: str | None, settings: Settings) -> dict[str, object]:
        if conversation_id is None:
            return {
                "id": None,
                "model_target": settings.model.target,
                "model_label": settings.model.label,
                "latest_response_id": None,
                "context_status": "active",
            }
        with self.connect() as db:
            row = db.execute("SELECT * FROM conversations WHERE id=?", (conversation_id,)).fetchone()
        if row is None:
            raise AppError("not_found", "対象が見つかりません。", 404)
        if row["context_status"] == "expired":
            raise AppError(
                "context_expired",
                "この会話はOpenAI側のコンテキストが失効したため継続できません。新しいチャットを開始してください。",
                409,
            )
        return dict(row)

    def save_turn(
        self,
        *,
        conversation_id: str | None,
        model_target: str,
        model_label: str,
        user_text: str,
        assistant_text: str,
        response_id: str,
        context_tokens: int,
        images: list[StagedImage],
        now: str,
    ) -> tuple[str, list[DeletionSnapshot]]:
        user_id = str(uuid.uuid4())
        assistant_id = str(uuid.uuid4())
        with self.connect() as db:
            if conversation_id is None:
                conversation_id = str(uuid.uuid4())
                db.execute(
                    "INSERT INTO conversations VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?)",
                    (
                        conversation_id,
                        derive_title(user_text),
                        model_target,
                        model_label,
                        response_id,
                        context_tokens,
                        now,
                        now,
                    ),
                )
            else:
                cursor = db.execute(
                    "UPDATE conversations SET latest_response_id=?, context_tokens=?, updated_at=? "
                    "WHERE id=? AND context_status='active'",
                    (response_id, context_tokens, now, conversation_id),
                )
                if cursor.rowcount != 1:
                    raise AppError("context_expired", "この会話は継続できません。", 409)
            db.execute(
                "INSERT INTO messages VALUES (?, ?, 'user', ?, NULL, ?)",
                (user_id, conversation_id, user_text, now),
            )
            db.execute(
                "INSERT INTO messages VALUES (?, ?, 'assistant', ?, ?, ?)",
                (assistant_id, conversation_id, assistant_text, response_id, now),
            )
            db.executemany(
                "INSERT INTO attachments VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (image.id, user_id, image.original_name, image.stored_name, image.width, image.height, image.byte_size)
                    for image in images
                ],
            )
            snapshots = self._remove_overflow(db)
        return conversation_id, snapshots

    def mark_expired(self, conversation_id: str) -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE conversations SET context_status='expired' WHERE id=?",
                (conversation_id,),
            )

    def delete(self, conversation_id: str) -> DeletionSnapshot:
        with self.connect() as db:
            row = db.execute("SELECT id FROM conversations WHERE id=?", (conversation_id,)).fetchone()
            if row is None:
                raise AppError("not_found", "対象が見つかりません。", 404)
            snapshot = self._snapshot(db, conversation_id)
            db.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
        return snapshot

    def prune(self) -> list[DeletionSnapshot]:
        with self.connect() as db:
            return self._remove_overflow(db)

    def attachment(self, attachment_id: str) -> dict[str, object]:
        with self.connect() as db:
            row = db.execute("SELECT * FROM attachments WHERE id=?", (attachment_id,)).fetchone()
        if row is None:
            raise AppError("not_found", "対象が見つかりません。", 404)
        return dict(row)

    def check(self) -> None:
        with self.connect() as db:
            db.execute("SELECT 1").fetchone()

    def _remove_overflow(self, db: sqlite3.Connection) -> list[DeletionSnapshot]:
        rows = db.execute(
            "SELECT id FROM conversations ORDER BY updated_at DESC, id ASC LIMIT -1 OFFSET ?",
            (MAX_CONVERSATIONS,),
        ).fetchall()
        snapshots = [self._snapshot(db, row["id"]) for row in rows]
        db.executemany("DELETE FROM conversations WHERE id=?", [(row["id"],) for row in rows])
        return snapshots

    @staticmethod
    def _snapshot(db: sqlite3.Connection, conversation_id: str) -> DeletionSnapshot:
        response_ids = db.execute(
            "SELECT response_id FROM messages WHERE conversation_id=? AND response_id IS NOT NULL",
            (conversation_id,),
        ).fetchall()
        names = db.execute(
            "SELECT a.stored_name FROM attachments a JOIN messages m ON m.id=a.message_id "
            "WHERE m.conversation_id=?",
            (conversation_id,),
        ).fetchall()
        return DeletionSnapshot(
            tuple(sorted({row[0] for row in response_ids if row[0]})),
            tuple(row[0] for row in names),
        )

    @staticmethod
    def _conversation(row: sqlite3.Row) -> dict[str, object]:
        return {
            "id": row["id"],
            "title": row["title"],
            "model_label": row["model_label"],
            "context_tokens": row["context_tokens"],
            "context_status": row["context_status"],
            "continuable": row["context_status"] == "active",
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _message(row: sqlite3.Row, attachments: list[sqlite3.Row]) -> dict[str, object]:
        return {
            "id": row["id"],
            "role": row["role"],
            "content": row["content"],
            "created_at": row["created_at"],
            "attachments": [
                {
                    "id": item["id"],
                    "original_name": item["original_name"],
                    "width": item["width"],
                    "height": item["height"],
                    "byte_size": item["byte_size"],
                    "content_url": f"/api/attachments/{item['id']}/content",
                }
                for item in attachments
            ],
        }


_MIME_BY_FORMAT = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
_ALLOWED_MIME = set(_MIME_BY_FORMAT.values())
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


class ImageService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.uploads = settings.uploads_path
        self.temp = settings.temp_path

    def initialize(self) -> None:
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.temp.mkdir(parents=True, exist_ok=True)
        Image.MAX_IMAGE_PIXELS = self.settings.images.max_decoded_pixels

    async def stage(self, uploads: list[UploadFile]) -> list[StagedImage]:
        if not uploads:
            return []
        if not self.settings.images.enabled or len(uploads) > self.settings.images.max_files:
            raise AppError("invalid_image", "画像の枚数が上限を超えています。")
        staged: list[StagedImage] = []
        try:
            for upload in uploads:
                staged.append(await self._stage_one(upload))
            return staged
        except BaseException:
            self.remove(staged)
            raise
        finally:
            for upload in uploads:
                await upload.close()

    async def _stage_one(self, upload: UploadFile) -> StagedImage:
        claimed = (upload.content_type or "").lower()
        if claimed not in _ALLOWED_MIME:
            raise AppError("invalid_image", "対応していない画像形式です。")
        identifier = str(uuid.uuid4())
        source_path = self.temp / f"{identifier}.upload"
        output_path = self.uploads / f"{identifier}.webp"
        size = 0
        try:
            with source_path.open("xb") as destination:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    if size > self.settings.images.max_file_bytes:
                        raise AppError("payload_too_large", "画像容量が上限を超えています。", 413)
                    destination.write(chunk)
            width, height = await asyncio.to_thread(self._normalize, source_path, output_path, claimed)
            return StagedImage(
                identifier,
                sanitize_filename(upload.filename),
                output_path.name,
                width,
                height,
                output_path.stat().st_size,
                output_path,
            )
        except AppError:
            output_path.unlink(missing_ok=True)
            raise
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            output_path.unlink(missing_ok=True)
            raise AppError("invalid_image", "画像を安全に読み取れませんでした。") from None
        finally:
            source_path.unlink(missing_ok=True)

    def _normalize(self, source_path: Path, output_path: Path, claimed: str) -> tuple[int, int]:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(source_path) as source:
                actual = _MIME_BY_FORMAT.get(source.format or "")
                if actual != claimed or actual not in _ALLOWED_MIME:
                    raise AppError("invalid_image", "画像の実形式が一致しません。")
                if getattr(source, "is_animated", False) or getattr(source, "n_frames", 1) > 1:
                    raise AppError("invalid_image", "アニメーション画像には対応していません。")
                if source.width * source.height > self.settings.images.max_decoded_pixels:
                    raise AppError("invalid_image", "画像の総画素数が上限を超えています。")
                source.load()
                normalized = ImageOps.exif_transpose(source)
                normalized.thumbnail(
                    (self.settings.images.max_dimension_px, self.settings.images.max_dimension_px),
                    Image.Resampling.LANCZOS,
                )
                output = normalized.convert("RGBA" if "A" in normalized.getbands() else "RGB")
                try:
                    output.save(output_path, "WEBP", quality=self.settings.images.webp_quality, method=6)
                    return output.size
                finally:
                    output.close()

    def remove(self, images: list[StagedImage]) -> None:
        for image in images:
            try:
                (self.uploads / Path(image.stored_name).name).unlink(missing_ok=True)
            except OSError:
                pass

    def delete_names(self, names: tuple[str, ...]) -> None:
        for name in names:
            if Path(name).name == name:
                (self.uploads / name).unlink(missing_ok=True)


def sanitize_filename(value: str | None) -> str:
    name = Path((value or "image").replace("\\", "/")).name
    return (_CONTROL_CHARS.sub("", name).strip() or "image")[:200]


def utc_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def derive_title(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized[:40] if normalized else "画像のチャット"


def require_uuid(value: str) -> str:
    try:
        parsed = str(uuid.UUID(value))
    except (ValueError, AttributeError):
        raise AppError("invalid_request", "IDの形式が正しくありません。") from None
    if parsed != value.lower():
        raise AppError("invalid_request", "IDの形式が正しくありません。")
    return parsed


SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; script-src 'self' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; style-src 'self' https://cdnjs.cloudflare.com; img-src 'self' blob: data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


class RequestTooLarge(Exception):
    pass


class RequestSizeLimitMiddleware:
    def __init__(self, app: Any, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Callable[..., Awaitable[Any]], send: Callable[..., Awaitable[Any]]) -> None:
        received = 0

        async def limited_receive() -> dict[str, Any]:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise RequestTooLarge()
            return message

        await self.app(scope, limited_receive if scope["type"] == "http" else receive, send)


def contains_exception(value: object, expected: type[BaseException]) -> bool:
    seen: set[int] = set()

    def visit(candidate: object) -> bool:
        if id(candidate) in seen:
            return False
        seen.add(id(candidate))
        if isinstance(candidate, expected):
            return True
        if isinstance(candidate, BaseException):
            return any(
                visit(item)
                for item in (
                    candidate.__cause__,
                    candidate.__context__,
                    candidate.args,
                    getattr(candidate, "_errors", None),
                )
                if item is not None
            )
        if isinstance(candidate, dict):
            return any(visit(item) for item in candidate.values())
        if isinstance(candidate, (list, tuple, set)):
            return any(visit(item) for item in candidate)
        return False

    return visit(value)


def create_app(settings: Settings | None = None, *, llm_service: Any | None = None) -> FastAPI:
    settings = settings or load_settings()
    database = ChatDatabase(settings.database_path)
    image_service = ImageService(settings)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    llm = llm_service or (OpenAIService(api_key) if api_key else None)
    locks: dict[str, asyncio.Lock] = {}

    async def cleanup(snapshot: DeletionSnapshot) -> None:
        await asyncio.to_thread(image_service.delete_names, snapshot.attachment_names)
        if llm is not None:
            for response_id in snapshot.response_ids:
                try:
                    await llm.delete_response(response_id)
                except Exception as exc:
                    logger.warning("OpenAI Response削除失敗: %s", type(exc).__name__)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await asyncio.to_thread(database.initialize)
        await asyncio.to_thread(image_service.initialize)
        for snapshot in await asyncio.to_thread(database.prune):
            await cleanup(snapshot)
        yield

    app = FastAPI(title=settings.ui.title, docs_url=None, redoc_url=None, lifespan=lifespan)
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=settings.server.max_request_bytes)

    allowed_hosts = {f"127.0.0.1:{settings.server.port}", f"localhost:{settings.server.port}", "testserver"}
    allowed_origins = {f"http://127.0.0.1:{settings.server.port}", f"http://localhost:{settings.server.port}"}

    @app.middleware("http")
    async def security(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        host = request.headers.get("host", "")
        origin = request.headers.get("origin")
        if host not in allowed_hosts:
            response: Response = JSONResponse(AppError("invalid_request", "許可されていないHostです。").payload(), status_code=400)
        elif request.method in {"POST", "DELETE"} and request.headers.get("x-simple-chat-request") != "1":
            response = JSONResponse(AppError("invalid_request", "必要なリクエストヘッダーがありません。").payload(), status_code=400)
        elif request.method in {"POST", "DELETE"} and origin is not None and origin not in allowed_origins:
            response = JSONResponse(AppError("invalid_request", "許可されていないOriginです。").payload(), status_code=400)
        else:
            response = await call_next(request)
        for key, value in SECURITY_HEADERS.items():
            response.headers[key] = value
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(AppError)
    async def app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(exc.payload(), status_code=exc.status)

    @app.exception_handler(RequestTooLarge)
    async def too_large(_: Request, __: RequestTooLarge) -> JSONResponse:
        exc = AppError("payload_too_large", "リクエスト全体の容量が上限を超えています。", 413)
        return JSONResponse(exc.payload(), status_code=413)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, error: RequestValidationError) -> JSONResponse:
        if contains_exception(error, RequestTooLarge):
            exc = AppError("payload_too_large", "リクエスト全体の容量が上限を超えています。", 413)
            return JSONResponse(exc.payload(), status_code=413)
        exc = AppError("invalid_request", "入力内容を確認してください。", 400)
        return JSONResponse(exc.payload(), status_code=400)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> Response:
        if contains_exception(exc, RequestTooLarge):
            error = AppError("payload_too_large", "リクエスト全体の容量が上限を超えています。", 413)
            return JSONResponse(error.payload(), status_code=413)
        if request.url.path.startswith("/api/"):
            error = AppError("not_found", "対象が見つかりません。", exc.status_code)
            return JSONResponse(error.payload(), status_code=exc.status_code)
        return JSONResponse({"detail": "Not found"}, status_code=exc.status_code)

    @app.get("/api/health")
    async def health() -> dict[str, object]:
        await asyncio.to_thread(database.check)
        return {"status": "ok", "llm_configured": llm is not None}

    @app.get("/api/state")
    async def state() -> dict[str, object]:
        return {
            "config": settings.public_dict(llm is not None),
            "conversations": await asyncio.to_thread(database.list_conversations),
        }

    @app.get("/api/conversations/{conversation_id}")
    async def conversation(conversation_id: str) -> dict[str, object]:
        return await asyncio.to_thread(database.detail, require_uuid(conversation_id))

    @app.delete("/api/conversations/{conversation_id}", status_code=204)
    async def delete_conversation(conversation_id: str) -> Response:
        parsed = require_uuid(conversation_id)
        lock = locks.setdefault(parsed, asyncio.Lock())
        if lock.locked():
            raise AppError("conversation_busy", "回答生成中の会話は削除できません。", 409)
        async with lock:
            snapshot = await asyncio.to_thread(database.delete, parsed)
            await cleanup(snapshot)
        return Response(status_code=204)

    @app.post("/api/messages")
    async def send_message(
        request: Request,
        conversation_id: str = Form(default=""),
        text: str = Form(default=""),
        images: list[UploadFile] = File(default=[]),
    ) -> StreamingResponse:
        if llm is None:
            raise AppError("llm_not_configured", "OpenAI APIキーが設定されていません。", 503)
        parsed = require_uuid(conversation_id) if conversation_id else None
        if not text.strip() and not images:
            raise AppError("invalid_request", "本文または画像を入力してください。")
        if len(text) > settings.messages.max_text_length:
            raise AppError("invalid_request", "本文が文字数上限を超えています。")
        lock_key = parsed or "__new_conversation__"
        lock = locks.setdefault(lock_key, asyncio.Lock())
        if lock.locked():
            raise AppError("conversation_busy", "回答を生成中です。", 409, True)
        await lock.acquire()
        staged: list[StagedImage] = []
        iterator: AsyncIterator[StreamEvent] | None = None
        try:
            target = await asyncio.to_thread(database.generation_target, parsed, settings)
            staged = await image_service.stage(images)
            iterator = llm.stream(
                model=str(target["model_target"]),
                text=text,
                image_paths=[image.path for image in staged],
                previous_response_id=target.get("latest_response_id"),
                instructions=settings.responses.instructions,
                max_output_tokens=settings.responses.max_output_tokens,
            ).__aiter__()
            first = await anext(iterator)
            if first.type != "created":
                raise LLMError("upstream_error", "回答を開始できませんでした。", True)
        except LLMError as exc:
            if exc.code == "context_expired" and parsed:
                await asyncio.to_thread(database.mark_expired, parsed)
            image_service.remove(staged)
            lock.release()
            if iterator is not None:
                await close_iterator(iterator)
            raise AppError(exc.code, exc.message, llm_error_status(exc.code), exc.retryable) from exc
        except BaseException:
            image_service.remove(staged)
            lock.release()
            if iterator is not None:
                await close_iterator(iterator)
            raise

        async def stream_body() -> AsyncIterator[bytes]:
            content = ""
            saved = False
            try:
                yield ndjson({"type": "start"})
                assert iterator is not None
                async for event in iterator:
                    if await request.is_disconnected():
                        raise asyncio.CancelledError()
                    if event.type == "delta":
                        content += event.text
                        yield ndjson({"type": "delta", "text": event.text})
                    elif event.type == "completed":
                        if not event.response_id:
                            raise LLMError("upstream_error", "回答IDを取得できませんでした。", True)
                        saved_id, snapshots = await asyncio.shield(
                            asyncio.to_thread(
                                database.save_turn,
                                conversation_id=parsed,
                                model_target=str(target["model_target"]),
                                model_label=str(target["model_label"]),
                                user_text=text,
                                assistant_text=content,
                                response_id=event.response_id,
                                context_tokens=event.input_tokens,
                                images=staged,
                                now=utc_text(),
                            )
                        )
                        saved = True
                        for snapshot in snapshots:
                            await cleanup(snapshot)
                        yield ndjson({"type": "completed", "conversation_id": saved_id})
                        return
                raise LLMError("upstream_error", "回答ストリームが途中で終了しました。", True)
            except LLMError as exc:
                if exc.code == "context_expired" and parsed:
                    await asyncio.shield(asyncio.to_thread(database.mark_expired, parsed))
                yield ndjson({"type": "error", "code": exc.code, "message": exc.message, "retryable": exc.retryable})
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Streaming failure: %s", type(exc).__name__)
                yield ndjson({"type": "error", "code": "internal_error", "message": "回答の保存に失敗しました。", "retryable": True})
            finally:
                if not saved:
                    image_service.remove(staged)
                lock.release()
                if iterator is not None:
                    await close_iterator(iterator)

        return StreamingResponse(stream_body(), media_type="application/x-ndjson; charset=utf-8")

    @app.get("/api/attachments/{attachment_id}/content")
    async def attachment(attachment_id: str) -> FileResponse:
        item = await asyncio.to_thread(database.attachment, require_uuid(attachment_id))
        path = settings.uploads_path / Path(str(item["stored_name"])).name
        if not path.is_file() or path.is_symlink():
            raise AppError("not_found", "対象が見つかりません。", 404)
        return FileResponse(path, media_type="image/webp", headers={"X-Content-Type-Options": "nosniff"})

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


def ndjson(value: dict[str, object]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def llm_error_status(code: str) -> int:
    return {"context_expired": 409, "rate_limited": 429, "upstream_unavailable": 503, "upstream_timeout": 504}.get(code, 502)


async def close_iterator(iterator: object) -> None:
    close = getattr(iterator, "aclose", None)
    if close is not None:
        try:
            await close()
        except Exception:
            pass


def main() -> int:
    if "--smoke" in sys.argv:
        return asyncio.run(smoke_openai(settings))
    try:
        settings = load_settings()
    except SettingsError as exc:
        print(f"設定エラー: {exc}", file=sys.stderr)
        return 2
    try:
        uvicorn.run(
            create_app(settings),
            host=settings.server.host,
            port=settings.server.port,
            workers=1,
            log_level="info",
        )
    except KeyboardInterrupt:
        return 130
    return 0


async def smoke_openai(settings: Settings) -> int:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("OPENAI_API_KEY が設定されていません。")
        return 2
    service = OpenAIService(api_key)
    response_id: str | None = None
    output = ""
    try:
        async for event in service.stream(
            model=settings.model.target,
            text="疎通確認です。「OK」とだけ回答してください。",
            image_paths=[],
            previous_response_id=None,
            instructions=settings.responses.instructions,
            max_output_tokens=min(settings.responses.max_output_tokens, 64),
        ):
            response_id = event.response_id or response_id
            output += event.text
    except LLMError as exc:
        print(f"実API疎通に失敗しました: {exc.code} / {exc.message}")
        return 1
    finally:
        if response_id:
            try:
                await service.delete_response(response_id)
            except Exception as exc:
                print(f"作成したResponseの後始末に失敗しました: {type(exc).__name__}")
    print(f"実API疎通に成功しました: model={settings.model.target}, output_chars={len(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
