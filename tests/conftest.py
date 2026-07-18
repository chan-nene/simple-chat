from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, load_settings
from app.domain import LLMServiceError, LLMStreamEvent, ResponseAlreadyGone
from app.main import create_app


@dataclass
class MutableClock:
    value: datetime

    def __call__(self) -> datetime:
        return self.value


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.deleted: list[str] = []
        self.plans: list[list[LLMStreamEvent | Exception]] = []

    def queue_success(self, text: str, response_id: str) -> None:
        midpoint = max(1, len(text) // 2)
        self.plans.append(
            [
                LLMStreamEvent("created", response_id=response_id),
                LLMStreamEvent("delta", text=text[:midpoint]),
                LLMStreamEvent("delta", text=text[midpoint:]),
                LLMStreamEvent("completed", response_id=response_id),
            ]
        )

    def queue_error(self, error: LLMServiceError) -> None:
        self.plans.append([error])

    def queue_stream_error(self, error: LLMServiceError, response_id: str = "resp_failed") -> None:
        self.plans.append([LLMStreamEvent("created", response_id=response_id), error])

    async def stream_response(self, **kwargs: Any) -> AsyncIterator[LLMStreamEvent]:
        self.calls.append(kwargs)
        plan = self.plans.pop(0) if self.plans else []
        for item in plan:
            if isinstance(item, Exception):
                raise item
            yield item

    async def delete_response(self, response_id: str) -> None:
        self.deleted.append(response_id)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    result = load_settings("config.example.toml").model_copy(deep=True)
    result.project_root = tmp_path
    result.database.path = "data/chat.db"
    result.storage.upload_directory = "data/uploads"
    result.storage.temp_directory = "data/tmp"
    return result


@pytest.fixture
def clock() -> MutableClock:
    return MutableClock(datetime(2026, 7, 18, 6, 0, tzinfo=timezone.utc))


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def client(
    settings: Settings, clock: MutableClock, fake_llm: FakeLLM
) -> Iterator[TestClient]:
    app = create_app(settings, llm_service=fake_llm, clock=clock)
    with TestClient(app, base_url="http://127.0.0.1:8000") as test_client:
        test_client.headers.update({"X-Simple-Chat-Request": "1"})
        yield test_client


@pytest.fixture
def unconfigured_client(
    settings: Settings, clock: MutableClock, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = create_app(settings, llm_service=None, clock=clock)
    with TestClient(app, base_url="http://127.0.0.1:8000") as test_client:
        test_client.headers.update({"X-Simple-Chat-Request": "1"})
        yield test_client


def create_conversation(client: TestClient, model_key: str | None = None) -> dict[str, Any]:
    payload = {} if model_key is None else {"model_key": model_key}
    response = client.post("/api/conversations", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def stream_events(client: TestClient, conversation_id: str, text: str) -> list[dict[str, Any]]:
    response = client.post(
        f"/api/conversations/{conversation_id}/messages", data={"text": text}
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/x-ndjson; charset=utf-8"
    return [json.loads(line) for line in response.text.splitlines()]
