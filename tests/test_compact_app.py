from __future__ import annotations

import io
import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import run


class FakeLLM:
    def __init__(self) -> None:
        self.plans: list[list[run.StreamEvent | Exception]] = []
        self.calls: list[dict[str, Any]] = []
        self.deleted: list[str] = []

    def success(self, text: str, response_id: str, input_tokens: int = 42) -> None:
        self.plans.append(
            [
                run.StreamEvent("created", response_id=response_id),
                run.StreamEvent("delta", text=text),
                run.StreamEvent("completed", response_id=response_id, input_tokens=input_tokens),
            ]
        )

    async def stream(self, **kwargs: Any) -> AsyncIterator[run.StreamEvent]:
        self.calls.append(kwargs)
        for item in self.plans.pop(0):
            if isinstance(item, Exception):
                raise item
            yield item

    async def delete_response(self, response_id: str) -> None:
        self.deleted.append(response_id)


@pytest.fixture
def fake() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def settings(tmp_path: Path) -> run.Settings:
    value = run.load_settings("config.example.toml").model_copy(deep=True)
    value.root = tmp_path
    return value


@pytest.fixture
def client(settings: run.Settings, fake: FakeLLM) -> Iterator[TestClient]:
    with TestClient(run.create_app(settings, llm_service=fake)) as value:
        yield value


def mutation() -> dict[str, str]:
    return {"X-Simple-Chat-Request": "1"}


def events(response: Any) -> list[dict[str, Any]]:
    return [json.loads(line) for line in response.text.splitlines()]


def send(client: TestClient, fake: FakeLLM, text: str, response_id: str) -> str:
    fake.success(f"回答:{text}", response_id)
    response = client.post("/api/messages", data={"text": text}, headers=mutation())
    assert response.status_code == 200
    values = events(response)
    assert [item["type"] for item in values] == ["start", "delta", "completed"]
    return values[-1]["conversation_id"]


def test_completed_turn_is_saved_and_model_is_read_only(
    client: TestClient, fake: FakeLLM
) -> None:
    conversation_id = send(client, fake, "最初の質問", "resp_1")
    detail = client.get(f"/api/conversations/{conversation_id}").json()

    assert detail["conversation"]["model_label"] == "GPT-5.6 Luna"
    assert detail["conversation"]["context_tokens"] == 42
    assert detail["conversation"]["continuable"] is True
    assert [message["role"] for message in detail["messages"]] == ["user", "assistant"]
    assert fake.calls[0]["previous_response_id"] is None

    fake.success("続きの回答", "resp_2", input_tokens=128)
    response = client.post(
        "/api/messages",
        data={"conversation_id": conversation_id, "text": "続き"},
        headers=mutation(),
    )
    assert response.status_code == 200
    assert fake.calls[1]["previous_response_id"] == "resp_1"
    updated = client.get(f"/api/conversations/{conversation_id}").json()
    assert updated["conversation"]["context_tokens"] == 128


def test_only_five_recent_conversations_are_kept(
    client: TestClient, fake: FakeLLM
) -> None:
    ids = [send(client, fake, f"質問{i}", f"resp_{i}") for i in range(6)]
    state = client.get("/api/state").json()

    assert len(state["conversations"]) == 5
    assert ids[0] not in {item["id"] for item in state["conversations"]}
    assert "resp_0" in fake.deleted


def test_failed_turn_is_not_saved_and_reference_loss_expires_conversation(
    client: TestClient, fake: FakeLLM
) -> None:
    conversation_id = send(client, fake, "成功", "resp_ok")
    fake.plans.append(
        [
            run.StreamEvent("created", response_id="resp_failed"),
            run.StreamEvent("delta", text="途中"),
            run.LLMError("upstream_error", "失敗しました。", True),
        ]
    )
    failed = client.post(
        "/api/messages",
        data={"conversation_id": conversation_id, "text": "失敗する質問"},
        headers=mutation(),
    )
    assert events(failed)[-1]["type"] == "error"
    assert len(client.get(f"/api/conversations/{conversation_id}").json()["messages"]) == 2

    fake.plans.append([run.LLMError("context_expired", "失効しました。")])
    expired = client.post(
        "/api/messages",
        data={"conversation_id": conversation_id, "text": "もう一度"},
        headers=mutation(),
    )
    assert expired.status_code == 409
    detail = client.get(f"/api/conversations/{conversation_id}").json()
    assert detail["conversation"]["continuable"] is False
    assert len(detail["messages"]) == 2


def test_image_is_normalized_and_removed_with_conversation(
    client: TestClient, fake: FakeLLM, settings: run.Settings
) -> None:
    source = io.BytesIO()
    Image.new("RGB", (20, 10), "red").save(source, "PNG")
    fake.success("画像回答", "resp_image")
    response = client.post(
        "/api/messages",
        data={"text": "画像"},
        files={"images": ("sample.png", source.getvalue(), "image/png")},
        headers=mutation(),
    )
    conversation_id = events(response)[-1]["conversation_id"]
    detail = client.get(f"/api/conversations/{conversation_id}").json()
    attachment = detail["messages"][0]["attachments"][0]
    assert client.get(attachment["content_url"]).headers["content-type"] == "image/webp"
    assert list(settings.uploads_path.glob("*.webp"))

    deleted = client.delete(f"/api/conversations/{conversation_id}", headers=mutation())
    assert deleted.status_code == 204
    assert not list(settings.uploads_path.glob("*.webp"))


def test_failed_new_chat_creates_no_history(client: TestClient, fake: FakeLLM) -> None:
    fake.plans.append(
        [
            run.StreamEvent("created", response_id="resp_failed"),
            run.LLMError("upstream_error", "失敗しました。", True),
        ]
    )
    response = client.post("/api/messages", data={"text": "失敗"}, headers=mutation())

    assert events(response)[-1]["type"] == "error"
    assert client.get("/api/state").json()["conversations"] == []


def test_invalid_image_is_not_saved(client: TestClient, fake: FakeLLM) -> None:
    send(client, fake, "タイトル", "resp_title")
    fake.success("使用されない回答", "resp_unused")
    invalid = client.post(
        "/api/messages",
        data={"text": "画像"},
        files={"images": ("fake.png", b"not an image", "image/png")},
        headers=mutation(),
    )
    assert invalid.status_code == 400
    assert len(client.get("/api/state").json()["conversations"]) == 1


def test_request_size_limit_returns_413(settings: run.Settings, fake: FakeLLM) -> None:
    settings.server.max_request_size_mb = 1
    with TestClient(run.create_app(settings, llm_service=fake), raise_server_exceptions=False) as limited:
        response = limited.post(
            "/api/messages",
            data={"text": "画像"},
            files={"images": ("large.png", b"x" * (1024 * 1024 + 1), "image/png")},
            headers=mutation(),
        )
    assert response.status_code == 413
