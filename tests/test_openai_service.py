from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from app.config import CompactionSettings
from app.domain import LLMServiceError, ResponseAlreadyGone
from app.openai_service import OpenAIResponsesService


class FakeStream:
    def __init__(self, events: list[object]) -> None:
        self.events = events
        self.closed = False

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for event in self.events:
            yield event

    async def close(self) -> None:
        self.closed = True


class FakeResponses:
    def __init__(self, stream: FakeStream | None = None, error: Exception | None = None) -> None:
        self.stream = stream
        self.error = error
        self.create_kwargs: dict[str, Any] | None = None
        self.deleted: list[tuple[str, float]] = []
        self.delete_error: Exception | None = None

    async def create(self, **kwargs: Any) -> FakeStream:
        self.create_kwargs = kwargs
        if self.error:
            raise self.error
        assert self.stream is not None
        return self.stream

    async def delete(self, response_id: str, *, timeout: float) -> None:
        self.deleted.append((response_id, timeout))
        if self.delete_error:
            raise self.delete_error


def event(event_type: str, **values: object) -> SimpleNamespace:
    return SimpleNamespace(type=event_type, **values)


def completed_events(response_id: str = "resp_new") -> list[object]:
    response = SimpleNamespace(id=response_id, status="completed")
    return [
        event("response.created", response=SimpleNamespace(id=response_id)),
        event("response.output_item.added", item=SimpleNamespace(type="reasoning")),
        event("response.output_item.added", item=SimpleNamespace(type="message")),
        event("response.content_part.added", part=SimpleNamespace(type="output_text")),
        event("response.output_text.delta", delta="こんにちは"),
        event("response.completed", response=response),
    ]


async def collect(service: OpenAIResponsesService, **overrides: object) -> list[object]:
    arguments: dict[str, object] = {
        "model_target": "gpt-test",
        "user_text": "current only",
        "image_paths": [],
        "previous_response_id": None,
        "instructions": "instructions",
        "max_output_tokens": 123,
        "compaction": None,
    }
    arguments.update(overrides)
    return [item async for item in service.stream_response(**arguments)]


def make_service(responses: FakeResponses) -> OpenAIResponsesService:
    service = OpenAIResponsesService("test-key")
    service.client = SimpleNamespace(responses=responses)
    return service


@pytest.mark.asyncio
async def test_request_contains_only_current_input_images_and_response_chain(tmp_path: Path) -> None:
    image = tmp_path / "normalized.webp"
    image.write_bytes(b"normalized image")
    stream = FakeStream(completed_events())
    responses = FakeResponses(stream)
    service = make_service(responses)
    events = await collect(
        service,
        image_paths=[image],
        previous_response_id="resp_previous",
        compaction=CompactionSettings(enabled=True, compact_threshold=5000),
    )

    assert [item.type for item in events] == ["created", "delta", "completed"]
    assert stream.closed is True
    request = responses.create_kwargs
    assert request is not None
    assert request["model"] == "gpt-test"
    assert request["instructions"] == "instructions"
    assert request["max_output_tokens"] == 123
    assert request["store"] is True
    assert request["stream"] is True
    assert request["previous_response_id"] == "resp_previous"
    assert request["context_management"] == [
        {"type": "compaction", "compact_threshold": 5000}
    ]
    content = request["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "current only"}
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"] == (
        "data:image/webp;base64," + base64.b64encode(b"normalized image").decode("ascii")
    )
    assert "older message" not in str(request)


@pytest.mark.asyncio
async def test_first_turn_omits_previous_and_disabled_compaction() -> None:
    responses = FakeResponses(FakeStream(completed_events()))
    service = make_service(responses)
    await collect(service)
    assert "previous_response_id" not in responses.create_kwargs
    assert "context_management" not in responses.create_kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_event",
    [
        event("response.incomplete", response=SimpleNamespace(id="resp_bad")),
        event("response.failed", response=SimpleNamespace(id="resp_bad")),
        event("response.refusal.delta", delta="no"),
        event("response.content_part.added", part=SimpleNamespace(type="refusal")),
    ],
)
async def test_incomplete_failed_and_refusal_are_errors_and_close_stream(
    bad_event: object,
) -> None:
    stream = FakeStream(
        [event("response.created", response=SimpleNamespace(id="resp_bad")), bad_event]
    )
    service = make_service(FakeResponses(stream))
    with pytest.raises(LLMServiceError, match="回答"):
        await collect(service)
    assert stream.closed is True


@pytest.mark.asyncio
async def test_unsupported_output_cannot_be_reported_as_completed() -> None:
    stream = FakeStream(
        [
            event("response.created", response=SimpleNamespace(id="resp_tool")),
            event("response.output_item.added", item=SimpleNamespace(type="function_call")),
            event(
                "response.completed",
                response=SimpleNamespace(id="resp_tool", status="completed"),
            ),
        ]
    )
    service = make_service(FakeResponses(stream))
    with pytest.raises(LLMServiceError) as captured:
        await collect(service)
    assert captured.value.code == "upstream_error"
    assert stream.closed is True


def api_exception(error_type: type[openai.APIStatusError], status: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(status, request=request)
    return error_type("upstream secret", response=response, body={"error": {}})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (api_exception(openai.RateLimitError, 429), "rate_limited"),
        (openai.APITimeoutError(request=httpx.Request("POST", "https://api.openai.com")), "upstream_timeout"),
        (openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com")), "upstream_unavailable"),
    ],
)
async def test_sdk_errors_are_normalized(error: Exception, expected_code: str) -> None:
    service = make_service(FakeResponses(error=error))
    with pytest.raises(LLMServiceError) as captured:
        await collect(service)
    assert captured.value.code == expected_code
    assert "secret" not in captured.value.message


@pytest.mark.asyncio
async def test_missing_previous_response_is_normalized_as_reference_loss() -> None:
    error = api_exception(openai.NotFoundError, 404)
    service = make_service(FakeResponses(error=error))
    with pytest.raises(LLMServiceError) as captured:
        await collect(service, previous_response_id="resp_missing")
    assert captured.value.code == "context_reference_lost"


@pytest.mark.asyncio
async def test_delete_uses_bounded_timeout_and_404_is_already_gone() -> None:
    responses = FakeResponses()
    service = make_service(responses)
    await service.delete_response("resp_delete")
    assert responses.deleted == [("resp_delete", 10.0)]
    responses.delete_error = api_exception(openai.NotFoundError, 404)
    with pytest.raises(ResponseAlreadyGone):
        await service.delete_response("resp_gone")
