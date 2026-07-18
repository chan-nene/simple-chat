from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Protocol

from .config import CompactionSettings


@dataclass(frozen=True, slots=True)
class StagedAttachment:
    id: str
    original_name: str
    stored_name: str
    original_mime_type: str
    stored_mime_type: str
    width: int
    height: int
    source_byte_size: int
    byte_size: int
    sha256: str
    path: Path


@dataclass(frozen=True, slots=True)
class TurnIds:
    user_message_id: str
    assistant_message_id: str
    context_epoch: int


@dataclass(frozen=True, slots=True)
class LLMStreamEvent:
    type: str
    text: str = ""
    response_id: str | None = None


class LLMServiceError(Exception):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


class ResponseAlreadyGone(Exception):
    """Remote response is already deleted or expired."""


class LLMService(Protocol):
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
    ) -> AsyncIterator[LLMStreamEvent]: ...

    async def delete_response(self, response_id: str) -> None: ...
