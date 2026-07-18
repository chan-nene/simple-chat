from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, model_validator
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import CompactionSettings, ModelSettings, Settings, load_settings
from .database import Database
from .deletion import ConversationDeletionService
from .domain import LLMService, LLMServiceError
from .errors import AppError, invalid_request
from .image_service import ImageService
from .locks import ConversationLocks
from .models import Conversation
from .openai_service import OpenAIResponsesService
from .presenters import conversation_json, message_json
from .repository import ChatRepository
from .security import RequestTooLarge, install_security_middleware, request_too_large_response
from .time import Clock, utc_now, utc_text


logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateConversationRequest(ApiModel):
    model_key: str | None = None


class UpdateConversationRequest(ApiModel):
    title: str | None = None
    model_key: str | None = None

    @model_validator(mode="after")
    def has_valid_field(self) -> "UpdateConversationRequest":
        if not self.model_fields_set:
            raise ValueError("at least one field is required")
        if "title" in self.model_fields_set and self.title is None:
            raise ValueError("title cannot be null")
        if "model_key" in self.model_fields_set and self.model_key is None:
            raise ValueError("model_key cannot be null")
        return self


def create_app(
    settings: Settings | None = None,
    *,
    llm_service: LLMService | None = None,
    clock: Clock = utc_now,
) -> FastAPI:
    settings = settings or load_settings()
    database = Database(settings.database_path, wal=settings.database.wal_mode)
    repository = ChatRepository(database, settings)
    image_service = ImageService(settings)
    locks = ConversationLocks()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if llm_service is None and api_key:
        llm_service = OpenAIResponsesService(api_key)
    llm_configured = llm_service is not None
    deletion = ConversationDeletionService(
        repository, image_service, llm_service, locks, settings, clock
    )

    def present_conversation(conversation: Conversation) -> dict[str, object]:
        payload = conversation_json(conversation, settings)
        payload["is_generating"] = bool(payload["is_generating"]) or locks.is_generating(
            conversation.id
        )
        return payload

    cleanup_task: asyncio.Task[None] | None = None

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        nonlocal cleanup_task
        await asyncio.to_thread(database.initialize)
        await asyncio.to_thread(image_service.initialize)
        await asyncio.to_thread(database.recover_interrupted_streams, utc_text(clock()))
        await deletion.cleanup_expired()

        async def cleanup_loop() -> None:
            while True:
                await asyncio.sleep(settings.retention.cleanup_interval_hours * 3600)
                try:
                    await deletion.cleanup_expired()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.error("Retention cleanup failed (%s)", type(exc).__name__)

        cleanup_task = asyncio.create_task(cleanup_loop(), name="retention-cleanup")
        try:
            yield
        finally:
            if cleanup_task is not None:
                cleanup_task.cancel()
                try:
                    await cleanup_task
                except asyncio.CancelledError:
                    pass
            database.close()

    app = FastAPI(title=settings.ui.title, docs_url=None, redoc_url=None, lifespan=lifespan)
    install_security_middleware(
        app, port=settings.server.port, max_request_bytes=settings.server.max_request_bytes
    )

    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(exc.payload(), status_code=exc.status_code)

    app.add_exception_handler(RequestTooLarge, request_too_large_response)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        if exception_tree_contains(exc, RequestTooLarge):
            return request_too_large_response(request, RequestTooLarge())
        error = invalid_request()
        return JSONResponse(error.payload(), status_code=400)

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException) -> Response:
        if exception_tree_contains(exc, RequestTooLarge):
            return request_too_large_response(request, RequestTooLarge())
        if request.url.path.startswith("/api/"):
            error = AppError(
                "not_found" if exc.status_code == 404 else "invalid_request",
                "対象が見つかりません。" if exc.status_code == 404 else "入力内容を確認してください。",
                exc.status_code,
                False,
            )
            return JSONResponse(error.payload(), status_code=exc.status_code)
        return JSONResponse({"detail": "Not found"}, status_code=exc.status_code)

    @app.exception_handler(SQLAlchemyError)
    async def database_error_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
        logger.error(
            "Database error request=%s type=%s",
            getattr(request.state, "correlation_id", "unknown"),
            type(exc).__name__,
        )
        error = AppError("database_error", "データベース処理に失敗しました。", 500, True)
        return JSONResponse(error.payload(), status_code=500)

    @app.exception_handler(Exception)
    async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "Internal error request=%s type=%s",
            getattr(request.state, "correlation_id", "unknown"),
            type(exc).__name__,
        )
        error = AppError("internal_error", "内部エラーが発生しました。", 500, True)
        return JSONResponse(error.payload(), status_code=500)

    @app.get("/api/health")
    async def health() -> JSONResponse:
        try:
            await asyncio.to_thread(repository.check_database)
        except Exception:
            payload = {
                "status": "unhealthy",
                "database": "error",
                "llm_configured": llm_configured,
                "error": {
                    "code": "database_error",
                    "message": "データベースを利用できません。",
                    "retryable": True,
                },
            }
            return JSONResponse(payload, status_code=503)
        return JSONResponse(
            {
                "status": "ok" if llm_configured else "degraded",
                "database": "ok",
                "llm_configured": llm_configured,
            }
        )

    @app.get("/api/config/public")
    async def public_config() -> dict[str, object]:
        return settings.public_dict(api_key_available=llm_configured)

    @app.get("/api/conversations")
    async def list_conversations() -> list[dict[str, object]]:
        conversations = await asyncio.to_thread(repository.list_conversations, clock())
        return [present_conversation(item) for item in conversations]

    @app.post("/api/conversations", status_code=201)
    async def create_conversation(body: CreateConversationRequest) -> dict[str, object]:
        model = resolve_model(body.model_key or settings.llm.default_model_key, settings)
        conversation = await asyncio.to_thread(repository.create_conversation, model, clock())
        return present_conversation(conversation)

    @app.get("/api/conversations/{conversation_id}")
    async def get_conversation(conversation_id: str) -> dict[str, object]:
        parsed = require_uuid(conversation_id)
        conversation = await asyncio.to_thread(repository.get_conversation, parsed, clock())
        return present_conversation(conversation)

    @app.patch("/api/conversations/{conversation_id}")
    async def update_conversation(
        conversation_id: str, body: UpdateConversationRequest
    ) -> dict[str, object]:
        parsed = require_uuid(conversation_id)
        title: str | None = None
        if "title" in body.model_fields_set:
            assert body.title is not None
            title = body.title.strip()
            if not 1 <= len(title) <= 100:
                raise invalid_request("タイトルは1文字以上100文字以下で入力してください。")
        model = None
        if "model_key" in body.model_fields_set:
            assert body.model_key is not None
            model = resolve_model(body.model_key, settings)
        lock = locks.get(parsed)
        if lock.locked():
            raise AppError("conversation_busy", "回答生成中の会話は変更できません。", 409, True)
        await lock.acquire()
        try:
            if await asyncio.to_thread(repository.is_generating, parsed, clock()):
                raise AppError(
                    "conversation_busy", "回答生成中の会話は変更できません。", 409, True
                )
            conversation = await asyncio.to_thread(
                repository.update_conversation,
                parsed,
                title=title,
                model=model,
                now=clock(),
            )
        finally:
            lock.release()
        return present_conversation(conversation)

    @app.delete("/api/conversations/{conversation_id}", status_code=204)
    async def delete_conversation(conversation_id: str) -> Response:
        parsed = require_uuid(conversation_id)
        lock = locks.get(parsed)
        if lock.locked():
            raise AppError("conversation_busy", "回答生成中の会話は削除できません。", 409, True)
        await lock.acquire()
        try:
            if await asyncio.to_thread(repository.is_generating, parsed, clock()):
                raise AppError(
                    "conversation_busy", "回答生成中の会話は削除できません。", 409, True
                )
            await deletion.delete_active_locked(parsed)
        finally:
            lock.release()
        return Response(status_code=204)

    @app.get("/api/conversations/{conversation_id}/messages")
    async def list_messages(conversation_id: str) -> list[dict[str, object]]:
        parsed = require_uuid(conversation_id)
        messages = await asyncio.to_thread(repository.list_messages, parsed, clock())
        return [message_json(item) for item in messages]

    @app.post("/api/conversations/{conversation_id}/messages")
    async def send_message(
        request: Request,
        conversation_id: str,
        text: str = Form(default=""),
        images: list[UploadFile] = File(default=[]),
    ) -> StreamingResponse:
        parsed = require_uuid(conversation_id)
        if not llm_configured or llm_service is None:
            raise AppError(
                "llm_not_configured", "OpenAI APIキーが設定されていません。", 503, False
            )
        if len(text) > settings.messages.max_text_length:
            raise invalid_request("本文が文字数上限を超えています。")
        if not text.strip() and not images:
            raise invalid_request("本文または画像を入力してください。")
        if len(images) > settings.images.max_files:
            raise AppError("invalid_image", "画像の枚数が上限を超えています。", 400, False)

        lock = locks.get(parsed)
        if lock.locked():
            raise AppError("conversation_busy", "この会話では回答を生成中です。", 409, True)
        await lock.acquire()
        locks.mark_generating(parsed)
        staged = []
        try:
            conversation = await asyncio.to_thread(repository.get_conversation, parsed, clock())
            if any(message.status == "streaming" for message in conversation.messages):
                raise AppError("conversation_busy", "この会話では回答を生成中です。", 409, True)
            model = settings.llm.enabled_models.get(conversation.model_key)
            if model is None:
                raise AppError(
                    "model_unavailable", "保存されたモデルを利用できません。モデルを選び直してください。", 400
                )
            if images and not model.supports_images:
                raise AppError(
                    "image_not_supported",
                    "選択中のモデルは画像入力に対応していません。画像を外すか、画像対応モデルへ変更してください。",
                    400,
                )
            staged = await image_service.stage(images)
            conversation, turn = await asyncio.to_thread(
                repository.start_turn,
                parsed,
                text_content=text,
                attachments=staged,
                now=clock(),
            )
        except BaseException:
            image_service.remove_staged(staged)
            locks.finish_generation(parsed)
            raise

        stream_iterator = llm_service.stream_response(
            model_target=conversation.model_target,
            user_text=text,
            image_paths=[item.path for item in staged],
            previous_response_id=conversation.latest_response_id,
            instructions=conversation.instructions,
            max_output_tokens=settings.responses.max_output_tokens,
            compaction=(
                CompactionSettings(
                    enabled=True,
                    compact_threshold=conversation.compact_threshold,
                )
                if conversation.compaction_enabled
                else None
            ),
        ).__aiter__()
        initial_response_id: str | None = None
        try:
            first_event = await anext(stream_iterator)
            if first_event.type != "created" or not first_event.response_id:
                raise LLMServiceError(
                    "upstream_error", "OpenAIの応答を開始できませんでした。", retryable=True
                )
            initial_response_id = first_event.response_id
        except StopAsyncIteration:
            error = LLMServiceError(
                "upstream_error", "OpenAIの応答を開始できませんでした。", retryable=True
            )
            await asyncio.to_thread(
                repository.fail_turn,
                turn.assistant_message_id,
                content="",
                response_id=None,
                code=error.code,
                message=error.message,
                cancelled=False,
                reset_context=False,
                now=clock(),
            )
            try:
                await close_iterator(stream_iterator)
            finally:
                locks.finish_generation(parsed)
            raise AppError(error.code, error.message, llm_error_status(error.code), error.retryable)
        except LLMServiceError as error:
            await asyncio.to_thread(
                repository.fail_turn,
                turn.assistant_message_id,
                content="",
                response_id=None,
                code=error.code,
                message=error.message,
                cancelled=False,
                reset_context=error.code == "context_reference_lost",
                now=clock(),
            )
            try:
                await close_iterator(stream_iterator)
            finally:
                locks.finish_generation(parsed)
            raise AppError(error.code, error.message, llm_error_status(error.code), error.retryable)
        except asyncio.CancelledError:
            try:
                await asyncio.shield(
                    asyncio.to_thread(
                        repository.fail_turn,
                        turn.assistant_message_id,
                        content="",
                        response_id=initial_response_id,
                        code="cancelled",
                        message="回答生成を停止しました。",
                        cancelled=True,
                        reset_context=False,
                        now=clock(),
                    )
                )
            finally:
                locks.finish_generation(parsed)
                await close_iterator(stream_iterator)
            raise
        except Exception as error:
            logger.error("Upstream startup failure (%s)", type(error).__name__)
            await asyncio.to_thread(
                repository.fail_turn,
                turn.assistant_message_id,
                content="",
                response_id=initial_response_id,
                code="internal_error",
                message="回答の生成を開始できませんでした。",
                cancelled=False,
                reset_context=False,
                now=clock(),
            )
            try:
                await close_iterator(stream_iterator)
            finally:
                locks.finish_generation(parsed)
            raise AppError("internal_error", "回答の生成を開始できませんでした。", 500, True)

        async def ndjson_stream() -> AsyncIterator[bytes]:
            content = ""
            response_id: str | None = initial_response_id
            finalized = False
            try:
                yield ndjson(
                    {
                        "type": "start",
                        "user_message_id": turn.user_message_id,
                        "assistant_message_id": turn.assistant_message_id,
                        "context_epoch": turn.context_epoch,
                    }
                )
                async for event in stream_iterator:
                    if await request.is_disconnected():
                        raise asyncio.CancelledError()
                    if event.type == "created":
                        raise LLMServiceError(
                            "upstream_error", "開始イベントが重複しました。", retryable=True
                        )
                    elif event.type == "delta":
                        content += event.text
                        yield ndjson(
                            {
                                "type": "delta",
                                "assistant_message_id": turn.assistant_message_id,
                                "text": event.text,
                            }
                        )
                    elif event.type == "completed":
                        response_id = event.response_id or response_id
                        if not response_id:
                            raise LLMServiceError(
                                "upstream_error", "回答IDを取得できませんでした。", retryable=True
                            )
                        await asyncio.shield(
                            asyncio.to_thread(
                                repository.complete_turn,
                                turn.assistant_message_id,
                                content=content,
                                response_id=response_id,
                                now=clock(),
                            )
                        )
                        finalized = True
                        yield ndjson(
                            {
                                "type": "completed",
                                "assistant_message_id": turn.assistant_message_id,
                                "status": "completed",
                            }
                        )
                        return
                if not finalized:
                    raise LLMServiceError(
                        "upstream_error", "回答ストリームが完了しませんでした。", retryable=True
                    )
            except LLMServiceError as exc:
                await asyncio.shield(
                    asyncio.to_thread(
                        repository.fail_turn,
                        turn.assistant_message_id,
                        content=content,
                        response_id=response_id,
                        code=exc.code,
                        message=exc.message,
                        cancelled=False,
                        reset_context=exc.code == "context_reference_lost",
                        now=clock(),
                    )
                )
                finalized = True
                yield ndjson(
                    {
                        "type": "error",
                        "assistant_message_id": turn.assistant_message_id,
                        "code": exc.code,
                        "message": exc.message,
                        "retryable": exc.retryable,
                    }
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Streaming failure (%s)", type(exc).__name__)
                safe_message = "回答の生成中にエラーが発生しました。"
                await asyncio.shield(
                    asyncio.to_thread(
                        repository.fail_turn,
                        turn.assistant_message_id,
                        content=content,
                        response_id=response_id,
                        code="internal_error",
                        message=safe_message,
                        cancelled=False,
                        reset_context=False,
                        now=clock(),
                    )
                )
                finalized = True
                yield ndjson(
                    {
                        "type": "error",
                        "assistant_message_id": turn.assistant_message_id,
                        "code": "internal_error",
                        "message": safe_message,
                        "retryable": True,
                    }
                )
            finally:
                try:
                    if not finalized:
                        await asyncio.shield(
                            asyncio.to_thread(
                                repository.fail_turn,
                                turn.assistant_message_id,
                                content=content,
                                response_id=response_id,
                                code="cancelled",
                                message="回答生成を停止しました。",
                                cancelled=True,
                                reset_context=False,
                                now=clock(),
                            )
                        )
                finally:
                    locks.finish_generation(parsed)
                    await close_iterator(stream_iterator)

        return StreamingResponse(
            ndjson_stream(), media_type="application/x-ndjson; charset=utf-8"
        )

    @app.get("/api/attachments/{attachment_id}/content")
    async def attachment_content(attachment_id: str) -> FileResponse:
        parsed = require_uuid(attachment_id)
        attachment = await asyncio.to_thread(repository.get_attachment, parsed, clock())
        try:
            path = image_service._safe_child(settings.uploads_path, attachment.stored_name)
        except RuntimeError:
            raise AppError("not_found", "対象が見つかりません。", 404, False) from None
        if not path.is_file() or path.is_symlink():
            raise AppError("not_found", "対象が見つかりません。", 404, False)
        return FileResponse(
            path,
            media_type=attachment.stored_mime_type,
            headers={
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Cross-Origin-Resource-Policy": "same-origin",
            },
        )

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def resolve_model(model_key: str, settings: Settings) -> ModelSettings:
    model = settings.llm.enabled_models.get(model_key)
    if model is None:
        raise AppError("invalid_model", "指定されたモデルは利用できません。", 400, False)
    return model


def require_uuid(value: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError):
        raise invalid_request("IDの形式が正しくありません。") from None
    if str(parsed) != value.lower():
        raise invalid_request("IDの形式が正しくありません。")
    return str(parsed)


def ndjson(value: dict[str, object]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def llm_error_status(code: str) -> int:
    return {
        "context_reference_lost": 400,
        "rate_limited": 429,
        "upstream_unavailable": 503,
        "upstream_timeout": 504,
        "upstream_error": 502,
    }.get(code, 502)


async def close_iterator(iterator: object) -> None:
    close = getattr(iterator, "aclose", None)
    if close is not None:
        try:
            await close()
        except Exception as exc:
            logger.warning("Upstream iterator cleanup failed (%s)", type(exc).__name__)


def exception_tree_contains(value: object, expected: type[BaseException]) -> bool:
    seen: set[int] = set()

    def visit(candidate: object) -> bool:
        identifier = id(candidate)
        if identifier in seen:
            return False
        seen.add(identifier)
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
