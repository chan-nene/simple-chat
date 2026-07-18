from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.domain import LLMServiceError, LLMStreamEvent
from app.database import Database
from app.main import create_app
from app.repository import ChatRepository
from tests.conftest import FakeLLM, MutableClock, create_conversation, stream_events


class BlockingLLM:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls: list[dict[str, Any]] = []

    async def stream_response(self, **kwargs: Any) -> AsyncIterator[LLMStreamEvent]:
        self.calls.append(kwargs)
        response_id = f"resp_{uuid.uuid4().hex}"
        yield LLMStreamEvent("created", response_id=response_id)
        if kwargs["user_text"] == "block":
            self.started.set()
            while not self.release.is_set():
                await asyncio.sleep(0.01)
        yield LLMStreamEvent("delta", text=f"answer:{kwargs['user_text']}")
        yield LLMStreamEvent("completed", response_id=response_id)

    async def delete_response(self, response_id: str) -> None:
        return None


def test_same_conversation_is_locked_while_other_conversation_can_generate(
    settings: object, clock: MutableClock
) -> None:
    llm = BlockingLLM()
    app = create_app(settings, llm_service=llm, clock=clock)
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        client.headers.update({"X-Simple-Chat-Request": "1"})
        first = create_conversation(client)
        second = create_conversation(client)
        result: dict[str, object] = {}

        def run_blocking_request() -> None:
            result["response"] = client.post(
                f"/api/conversations/{first['id']}/messages", data={"text": "block"}
            )

        thread = threading.Thread(target=run_blocking_request, daemon=True)
        thread.start()
        assert llm.started.wait(timeout=5)

        duplicate = client.post(
            f"/api/conversations/{first['id']}/messages", data={"text": "duplicate"}
        )
        update = client.patch(
            f"/api/conversations/{first['id']}", json={"model_key": "gpt-5.6-sol"}
        )
        remove = client.delete(f"/api/conversations/{first['id']}")
        assert [duplicate.status_code, update.status_code, remove.status_code] == [409, 409, 409]
        assert all(
            response.json()["error"]["code"] == "conversation_busy"
            for response in (duplicate, update, remove)
        )

        parallel = client.post(
            f"/api/conversations/{second['id']}/messages", data={"text": "other"}
        )
        assert parallel.status_code == 200
        assert json.loads(parallel.text.splitlines()[-1])["type"] == "completed"

        llm.release.set()
        thread.join(timeout=8)
        assert not thread.is_alive()
        assert result["response"].status_code == 200


def test_database_streaming_state_alone_blocks_mutations(
    settings: object, clock: MutableClock, fake_llm: FakeLLM
) -> None:
    with TestClient(
        create_app(settings, llm_service=fake_llm, clock=clock),
        base_url="http://127.0.0.1:8000",
    ) as client:
        client.headers.update({"X-Simple-Chat-Request": "1"})
        conversation = create_conversation(client)
        turn_id = str(uuid.uuid4())
        timestamp = conversation["updated_at"]
        with sqlite3.connect(settings.database_path) as connection:
            for role, status in (("user", "completed"), ("assistant", "streaming")):
                connection.execute(
                    """
                    INSERT INTO messages (
                        id, conversation_id, turn_id, role, content, status, response_id,
                        context_epoch, included_in_context, error_code, error_message,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, '', ?, NULL, 1, 0, NULL, NULL, ?, ?)
                    """,
                    (str(uuid.uuid4()), conversation["id"], turn_id, role, status, timestamp, timestamp),
                )
            connection.commit()

        detail = client.get(f"/api/conversations/{conversation['id']}")
        assert detail.json()["is_generating"] is True
        update = client.patch(
            f"/api/conversations/{conversation['id']}", json={"title": "blocked"}
        )
        remove = client.delete(f"/api/conversations/{conversation['id']}")
        send = client.post(
            f"/api/conversations/{conversation['id']}/messages", data={"text": "blocked"}
        )
        assert [update.status_code, remove.status_code, send.status_code] == [409, 409, 409]
        assert fake_llm.calls == []


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (LLMServiceError("rate_limited", "later", retryable=True), 429),
        (LLMServiceError("upstream_timeout", "timeout", retryable=True), 504),
        (LLMServiceError("upstream_unavailable", "unavailable", retryable=True), 503),
        (LLMServiceError("upstream_error", "bad upstream", retryable=False), 502),
    ],
)
def test_upstream_errors_before_start_use_http_status_and_persist_failed_turn(
    client: TestClient, fake_llm: FakeLLM, error: LLMServiceError, status: int
) -> None:
    conversation = create_conversation(client)
    fake_llm.queue_error(error)
    response = client.post(
        f"/api/conversations/{conversation['id']}/messages", data={"text": "attempt"}
    )
    assert response.status_code == status
    assert response.json()["error"] == {
        "code": error.code,
        "message": error.message,
        "retryable": error.retryable,
    }
    messages = client.get(f"/api/conversations/{conversation['id']}/messages").json()
    assert [message["status"] for message in messages] == ["completed", "failed"]
    assert all(message["included_in_context"] is False for message in messages)


def test_reference_loss_after_start_has_one_ndjson_terminal_and_resets_epoch(
    client: TestClient, fake_llm: FakeLLM
) -> None:
    conversation = create_conversation(client)
    fake_llm.queue_success("first", "resp_first")
    stream_events(client, conversation["id"], "one")
    fake_llm.queue_stream_error(
        LLMServiceError("context_reference_lost", "lost", retryable=False),
        response_id="resp_failed",
    )
    events = stream_events(client, conversation["id"], "two")
    assert [event["type"] for event in events] == ["start", "error"]
    assert events[-1]["code"] == "context_reference_lost"
    assert client.get(f"/api/conversations/{conversation['id']}").json()["context_epoch"] == 2
    messages = client.get(f"/api/conversations/{conversation['id']}/messages").json()
    assert messages[-1]["status"] == "failed"
    assert messages[-1]["error_code"] == "context_reference_lost"


def test_multibyte_deltas_and_missing_terminal_are_normalized(
    client: TestClient, fake_llm: FakeLLM
) -> None:
    conversation = create_conversation(client)
    fake_llm.plans.append(
        [
            LLMStreamEvent("created", response_id="resp_jp"),
            LLMStreamEvent("delta", text="こん"),
            LLMStreamEvent("delta", text="にちは🌙"),
            LLMStreamEvent("completed", response_id="resp_jp"),
        ]
    )
    events = stream_events(client, conversation["id"], "日本語")
    assert "".join(event.get("text", "") for event in events) == "こんにちは🌙"
    fake_llm.plans.append([LLMStreamEvent("created", response_id="resp_open")])
    incomplete = stream_events(client, conversation["id"], "incomplete")
    assert [event["type"] for event in incomplete] == ["start", "error"]
    assert incomplete[-1]["code"] == "upstream_error"


def test_completed_cancelled_race_obeys_committed_state(
    settings: object, clock: MutableClock
) -> None:
    database = Database(settings.database_path)
    database.initialize()
    repository = ChatRepository(database, settings)
    model = settings.llm.enabled_models[settings.llm.default_model_key]
    conversation = repository.create_conversation(model, clock())

    _, completed_turn = repository.start_turn(
        conversation.id, text_content="complete first", attachments=[], now=clock()
    )
    repository.complete_turn(
        completed_turn.assistant_message_id,
        content="done",
        response_id="resp_done",
        now=clock(),
    )
    repository.fail_turn(
        completed_turn.assistant_message_id,
        content="partial",
        response_id="resp_done",
        code="cancelled",
        message="stopped",
        cancelled=True,
        reset_context=False,
        now=clock(),
    )
    messages = repository.list_messages(conversation.id, clock())
    completed = next(item for item in messages if item.id == completed_turn.assistant_message_id)
    assert completed.status == "completed"
    assert repository.get_conversation(conversation.id, clock()).latest_response_id == "resp_done"

    _, cancelled_turn = repository.start_turn(
        conversation.id, text_content="cancel first", attachments=[], now=clock()
    )
    repository.fail_turn(
        cancelled_turn.assistant_message_id,
        content="partial",
        response_id="resp_partial",
        code="cancelled",
        message="stopped",
        cancelled=True,
        reset_context=False,
        now=clock(),
    )
    with pytest.raises(RuntimeError, match="streaming assistant"):
        repository.complete_turn(
            cancelled_turn.assistant_message_id,
            content="too late",
            response_id="resp_too_late",
            now=clock(),
        )
    messages = repository.list_messages(conversation.id, clock())
    cancelled = next(item for item in messages if item.id == cancelled_turn.assistant_message_id)
    assert cancelled.status == "cancelled"
    assert repository.get_conversation(conversation.id, clock()).latest_response_id == "resp_done"
    database.close()
