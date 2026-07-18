from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any, AsyncIterator

import openai
from openai import AsyncOpenAI

from .config import CompactionSettings
from .domain import LLMServiceError, LLMStreamEvent, ResponseAlreadyGone


class OpenAIResponsesService:
    """The only module that knows OpenAI SDK event and request shapes."""

    def __init__(self, api_key: str) -> None:
        self.client = AsyncOpenAI(api_key=api_key)

    async def stream_response(
        self,
        *,
        model_target: str,
        user_text: str,
        image_paths: list[Path],
        previous_response_id: str | None,
        instructions: str,
        max_output_tokens: int,
        compaction: CompactionSettings | None,
    ) -> AsyncIterator[LLMStreamEvent]:
        content: list[dict[str, str]] = []
        if user_text:
            content.append({"type": "input_text", "text": user_text})
        for path in image_paths:
            encoded = await asyncio.to_thread(_image_data_url, path)
            content.append({"type": "input_image", "image_url": encoded})

        arguments: dict[str, Any] = {
            "model": model_target,
            "input": [{"role": "user", "content": content}],
            "instructions": instructions,
            "max_output_tokens": max_output_tokens,
            "store": True,
            "stream": True,
        }
        if previous_response_id is not None:
            arguments["previous_response_id"] = previous_response_id
        if isinstance(compaction, CompactionSettings) and compaction.enabled:
            # Kept in the provider adapter so SDK/API-specific compaction shape cannot leak outward.
            arguments["context_management"] = [
                {"type": "compaction", "compact_threshold": compaction.compact_threshold}
            ]

        try:
            stream = await self.client.responses.create(**arguments)
            try:
                saw_terminal = False
                unsupported_output = False
                async for event in stream:
                    event_type = str(getattr(event, "type", ""))
                    if event_type == "response.created":
                        response_id = getattr(getattr(event, "response", None), "id", None)
                        yield LLMStreamEvent("created", response_id=response_id)
                    elif event_type == "response.output_text.delta":
                        yield LLMStreamEvent("delta", text=str(getattr(event, "delta", "")))
                    elif event_type in {"response.refusal.delta", "response.refusal.done"}:
                        saw_terminal = True
                        raise LLMServiceError(
                            "upstream_error", "モデルが回答を生成できませんでした。", retryable=False
                        )
                    elif event_type in {"response.output_item.added", "response.output_item.done"}:
                        item_type = str(getattr(getattr(event, "item", None), "type", ""))
                        if item_type and item_type not in {"message", "reasoning"}:
                            unsupported_output = True
                    elif event_type in {"response.content_part.added", "response.content_part.done"}:
                        part_type = str(getattr(getattr(event, "part", None), "type", ""))
                        if part_type == "refusal":
                            raise LLMServiceError(
                                "upstream_error", "モデルが回答を生成できませんでした。", retryable=False
                            )
                        if part_type and part_type != "output_text":
                            unsupported_output = True
                    elif event_type in {"response.failed", "response.incomplete", "error"}:
                        saw_terminal = True
                        raise LLMServiceError(
                            "upstream_error",
                            "回答が完了しませんでした。しばらく待って再試行してください。",
                            retryable=True,
                        )
                    elif event_type == "response.completed":
                        response = getattr(event, "response", None)
                        response_id = getattr(response, "id", None)
                        status = getattr(response, "status", "completed")
                        if status != "completed" or not response_id or unsupported_output:
                            raise LLMServiceError(
                                "upstream_error", "回答が正常に完了しませんでした。", retryable=True
                            )
                        saw_terminal = True
                        yield LLMStreamEvent("completed", response_id=response_id)
                if not saw_terminal:
                    raise LLMServiceError(
                        "upstream_error", "回答ストリームが完了せず終了しました。", retryable=True
                    )
            finally:
                await stream.close()
        except LLMServiceError:
            raise
        except openai.RateLimitError as exc:
            raise LLMServiceError(
                "rate_limited", "しばらく待ってから再試行してください。", retryable=True
            ) from exc
        except openai.APITimeoutError as exc:
            raise LLMServiceError(
                "upstream_timeout", "応答がタイムアウトしました。再試行してください。", retryable=True
            ) from exc
        except openai.APIConnectionError as exc:
            raise LLMServiceError(
                "upstream_unavailable", "OpenAIへ接続できませんでした。", retryable=True
            ) from exc
        except (openai.BadRequestError, openai.NotFoundError) as exc:
            if previous_response_id and _is_reference_error(exc):
                raise LLMServiceError(
                    "context_reference_lost",
                    "AI側の会話コンテキストを継続できなかったため、次回から新しいコンテキストを開始します。",
                    retryable=False,
                ) from exc
            raise LLMServiceError(
                "upstream_error", "OpenAIがリクエストを受理できませんでした。", retryable=False
            ) from exc
        except openai.APIError as exc:
            raise LLMServiceError(
                "upstream_error", "OpenAIでエラーが発生しました。", retryable=True
            ) from exc

    async def delete_response(self, response_id: str) -> None:
        try:
            await self.client.responses.delete(response_id, timeout=10.0)
        except openai.NotFoundError as exc:
            raise ResponseAlreadyGone() from exc


def _image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/webp;base64,{encoded}"


def _is_reference_error(exc: Exception) -> bool:
    if isinstance(exc, openai.NotFoundError):
        return True
    body = getattr(exc, "body", None)
    pieces: list[str] = []
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            pieces.extend(str(error.get(key, "")) for key in ("param", "code", "type"))
    marker = " ".join(pieces).lower()
    return "previous_response" in marker or "response_id" in marker
