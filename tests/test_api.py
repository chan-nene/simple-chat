from __future__ import annotations

import io
import hashlib
import sqlite3
import uuid
from datetime import timedelta
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.domain import LLMServiceError
from app.main import create_app
from tests.conftest import FakeLLM, MutableClock, create_conversation, stream_events


def test_health_and_public_config(client: TestClient) -> None:
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "database": "ok", "llm_configured": True}

    public = client.get("/api/config/public").json()
    assert public["app_title"] == "Simple Chat"
    assert public["ai_icon_url"] == "/favicon.svg"
    assert public["history_days"] == 7
    assert {model["key"] for model in public["models"]} == {
        "gpt-5.6-luna",
        "gpt-5.6-terra",
        "gpt-5.6-sol",
    }
    serialized = str(public).lower()
    assert "provider_model" not in serialized
    assert "instructions" not in serialized
    assert "api_key" not in serialized


def test_unconfigured_mode_reads_but_rejects_send(unconfigured_client: TestClient) -> None:
    health = unconfigured_client.get("/api/health").json()
    assert health["status"] == "degraded"
    conversation = create_conversation(unconfigured_client)
    response = unconfigured_client.post(
        f"/api/conversations/{conversation['id']}/messages", data={"text": "hello"}
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "llm_not_configured"


def test_conversation_crud_and_internal_fields_are_hidden(client: TestClient) -> None:
    conversation = create_conversation(client, "gpt-5.6-terra")
    assert conversation["title"] == "新しいチャット"
    assert conversation["model_key"] == "gpt-5.6-terra"
    assert conversation["context_epoch"] == 1
    assert "latest_response_id" not in conversation
    assert "model_target" not in conversation

    updated = client.patch(
        f"/api/conversations/{conversation['id']}", json={"title": "  設計相談  "}
    )
    assert updated.status_code == 200
    assert updated.json()["title"] == "設計相談"
    assert client.get("/api/conversations").json()[0]["id"] == conversation["id"]

    deleted = client.delete(f"/api/conversations/{conversation['id']}")
    assert deleted.status_code == 204
    assert client.get(f"/api/conversations/{conversation['id']}").status_code == 404


def test_invalid_uuid_model_body_and_security_contract(client: TestClient) -> None:
    assert client.get("/api/conversations/not-a-uuid").status_code == 400
    bad_model = client.post("/api/conversations", json={"model_key": "missing"})
    assert bad_model.status_code == 400
    assert bad_model.json()["error"]["code"] == "invalid_model"
    unknown = client.post("/api/conversations", json={"unexpected": True})
    assert unknown.status_code == 400
    assert unknown.json()["error"]["code"] == "invalid_request"

    without_marker = client.post(
        "/api/conversations", json={}, headers={"X-Simple-Chat-Request": "0"}
    )
    assert without_marker.status_code == 400
    hostile_origin = client.post(
        "/api/conversations",
        json={},
        headers={"Origin": "https://example.com", "X-Simple-Chat-Request": "1"},
    )
    assert hostile_origin.status_code == 400
    hostile_host = client.get("/api/health", headers={"Host": "example.com"})
    assert hostile_host.status_code == 400


def test_stream_success_uses_only_current_input_and_previous_response_id(
    client: TestClient, fake_llm: FakeLLM, settings: object
) -> None:
    conversation = create_conversation(client)
    fake_llm.queue_success("最初の回答", "resp_1")
    first = stream_events(client, conversation["id"], "最初の質問")
    assert [event["type"] for event in first] == ["start", "delta", "delta", "completed"]
    assert all("response_id" not in event for event in first)
    assert fake_llm.calls[0]["previous_response_id"] is None
    assert fake_llm.calls[0]["user_text"] == "最初の質問"

    fake_llm.queue_success("次の回答", "resp_2")
    second = stream_events(client, conversation["id"], "次の質問")
    assert second[-1]["type"] == "completed"
    assert fake_llm.calls[1]["previous_response_id"] == "resp_1"
    assert fake_llm.calls[1]["user_text"] == "次の質問"
    assert "最初の質問" not in fake_llm.calls[1]["user_text"]

    messages = client.get(f"/api/conversations/{conversation['id']}/messages").json()
    assert [message["role"] for message in messages] == ["user", "assistant", "user", "assistant"]
    assert all(message["included_in_context"] for message in messages)
    assert all("response_id" not in message for message in messages)
    assert messages[0]["content"] == "最初の質問"
    assert client.get(f"/api/conversations/{conversation['id']}").json()["title"] == "最初の質問"
    with sqlite3.connect(settings.database_path) as connection:
        rows = connection.execute(
            """
            SELECT turn_id, role, status, response_id, included_in_context
            FROM messages WHERE conversation_id = ? ORDER BY created_at, id
            """,
            (conversation["id"],),
        ).fetchall()
        latest_response_id = connection.execute(
            "SELECT latest_response_id FROM conversations WHERE id = ?",
            (conversation["id"],),
        ).fetchone()[0]
    assert rows[0][0] == rows[1][0] and rows[2][0] == rows[3][0]
    assert rows[0][0] != rows[2][0]
    assert [(row[1], row[2], row[4]) for row in rows] == [
        ("user", "completed", 1),
        ("assistant", "completed", 1),
        ("user", "completed", 1),
        ("assistant", "completed", 1),
    ]
    assert rows[-1][3] == latest_response_id == "resp_2"


def test_general_failure_preserves_last_good_chain(
    client: TestClient, fake_llm: FakeLLM
) -> None:
    conversation = create_conversation(client)
    fake_llm.queue_success("ok", "resp_good")
    stream_events(client, conversation["id"], "good")
    fake_llm.queue_stream_error(LLMServiceError("rate_limited", "later", retryable=True))
    failed = stream_events(client, conversation["id"], "failed")
    assert failed[-1] == {
        "type": "error",
        "assistant_message_id": failed[-1]["assistant_message_id"],
        "code": "rate_limited",
        "message": "later",
        "retryable": True,
    }
    fake_llm.queue_success("recovered", "resp_next")
    stream_events(client, conversation["id"], "next")
    assert fake_llm.calls[2]["previous_response_id"] == "resp_good"
    messages = client.get(f"/api/conversations/{conversation['id']}/messages").json()
    assert messages[2]["included_in_context"] is False
    assert messages[3]["status"] == "failed"


def test_context_reference_loss_resets_epoch_without_retry(
    client: TestClient, fake_llm: FakeLLM
) -> None:
    conversation = create_conversation(client)
    fake_llm.queue_success("ok", "resp_old")
    stream_events(client, conversation["id"], "one")
    fake_llm.queue_error(
        LLMServiceError("context_reference_lost", "context lost", retryable=False)
    )
    failed = client.post(
        f"/api/conversations/{conversation['id']}/messages", data={"text": "two"}
    )
    assert failed.status_code == 400
    assert failed.json()["error"]["code"] == "context_reference_lost"
    assert len(fake_llm.calls) == 2
    current = client.get(f"/api/conversations/{conversation['id']}").json()
    assert current["context_epoch"] == 2

    fake_llm.queue_success("fresh", "resp_fresh")
    stream_events(client, conversation["id"], "three")
    assert fake_llm.calls[2]["previous_response_id"] is None


def test_model_change_with_history_resets_context_but_noop_does_not(
    client: TestClient, fake_llm: FakeLLM
) -> None:
    conversation = create_conversation(client)
    unchanged = client.patch(
        f"/api/conversations/{conversation['id']}", json={"model_key": "gpt-5.6-luna"}
    ).json()
    assert unchanged["context_epoch"] == 1
    fake_llm.queue_success("ok", "resp_1")
    stream_events(client, conversation["id"], "hello")
    changed = client.patch(
        f"/api/conversations/{conversation['id']}", json={"model_key": "gpt-5.6-sol"}
    ).json()
    assert changed["context_epoch"] == 2
    fake_llm.queue_success("new", "resp_2")
    stream_events(client, conversation["id"], "after")
    assert fake_llm.calls[-1]["previous_response_id"] is None
    assert fake_llm.calls[-1]["model_target"] == "gpt-5.6-sol"


def test_image_is_normalized_and_sent_by_local_path(
    client: TestClient, fake_llm: FakeLLM, settings: object
) -> None:
    conversation = create_conversation(client)
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), "#ff0000").save(buffer, format="PNG")
    fake_llm.queue_success("見えました", "resp_image")
    response = client.post(
        f"/api/conversations/{conversation['id']}/messages",
        data={"text": ""},
        files={"images": ("sample.png", buffer.getvalue(), "image/png")},
    )
    assert response.status_code == 200
    messages = client.get(f"/api/conversations/{conversation['id']}/messages").json()
    attachment = messages[0]["attachments"][0]
    assert attachment["stored_mime_type"] == "image/webp"
    assert attachment["width"] == 32
    assert attachment["height"] == 24
    assert "stored_name" not in attachment
    content = client.get(attachment["content_url"])
    assert content.status_code == 200
    assert content.headers["content-type"] == "image/webp"
    stored_path = fake_llm.calls[0]["image_paths"][0]
    assert stored_path.suffix == ".webp"
    with sqlite3.connect(settings.database_path) as connection:
        saved_hash, source_byte_size, byte_size = connection.execute(
            "SELECT sha256, source_byte_size, byte_size FROM attachments"
        ).fetchone()
    assert saved_hash == hashlib.sha256(stored_path.read_bytes()).hexdigest()
    assert source_byte_size == len(buffer.getvalue())
    assert byte_size == stored_path.stat().st_size


def test_mime_mismatch_and_empty_submission_are_rejected_before_llm(
    client: TestClient, fake_llm: FakeLLM
) -> None:
    conversation = create_conversation(client)
    empty = client.post(
        f"/api/conversations/{conversation['id']}/messages", data={"text": "   "}
    )
    assert empty.status_code == 400
    data = io.BytesIO()
    Image.new("RGB", (8, 8)).save(data, format="PNG")
    mismatch = client.post(
        f"/api/conversations/{conversation['id']}/messages",
        data={"text": "image"},
        files={"images": ("wrong.jpg", data.getvalue(), "image/jpeg")},
    )
    assert mismatch.status_code == 400
    assert mismatch.json()["error"]["code"] == "invalid_image"
    assert fake_llm.calls == []


def test_expired_conversation_is_logically_hidden_and_removed_on_restart(
    settings: object, clock: MutableClock, fake_llm: FakeLLM
) -> None:
    from app.main import create_app

    app = create_app(settings, llm_service=fake_llm, clock=clock)
    with TestClient(app, base_url="http://127.0.0.1:8000") as first:
        first.headers.update({"X-Simple-Chat-Request": "1"})
        conversation = create_conversation(first)
    clock.value += timedelta(days=7, milliseconds=1)
    app2 = create_app(settings, llm_service=fake_llm, clock=clock)
    with TestClient(app2, base_url="http://127.0.0.1:8000") as second:
        second.headers.update({"X-Simple-Chat-Request": "1"})
        assert second.get(f"/api/conversations/{conversation['id']}").status_code == 404
        assert second.get("/api/conversations").json() == []


def test_security_headers_are_present(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert "default-src 'self'" in response.headers["content-security-policy"]


def test_model_target_is_snapshotted_and_new_conversation_uses_new_target(
    client: TestClient, fake_llm: FakeLLM, settings: object
) -> None:
    old = create_conversation(client, "gpt-5.6-terra")
    settings.llm.enabled_models["gpt-5.6-terra"].provider_model = "gpt-5.6-terra-new-target"
    fake_llm.queue_success("old target", "resp_old_target")
    stream_events(client, old["id"], "existing")
    assert fake_llm.calls[-1]["model_target"] == "gpt-5.6-terra"

    new = create_conversation(client, "gpt-5.6-terra")
    fake_llm.queue_success("new target", "resp_new_target")
    stream_events(client, new["id"], "new")
    assert fake_llm.calls[-1]["model_target"] == "gpt-5.6-terra-new-target"


def test_instructions_and_compaction_are_conversation_snapshots(
    settings: object, clock: MutableClock
) -> None:
    settings.responses.instructions = "saved instructions"
    settings.responses.compaction.enabled = True
    settings.responses.compaction.compact_threshold = 5000
    fake = FakeLLM()
    with TestClient(
        create_app(settings, llm_service=fake, clock=clock),
        base_url="http://127.0.0.1:8000",
    ) as test_client:
        test_client.headers.update({"X-Simple-Chat-Request": "1"})
        conversation = create_conversation(test_client)
        settings.responses.instructions = "new instructions"
        settings.responses.compaction.compact_threshold = 9000
        fake.queue_success("ok", "resp_snapshot")
        stream_events(test_client, conversation["id"], "snapshot")

    call = fake.calls[-1]
    assert call["instructions"] == "saved instructions"
    assert call["compaction"].enabled is True
    assert call["compaction"].compact_threshold == 5000


def test_removed_model_keeps_history_blocks_send_and_never_falls_back(
    settings: object, clock: MutableClock
) -> None:
    first_llm = FakeLLM()
    app = create_app(settings, llm_service=first_llm, clock=clock)
    with TestClient(app, base_url="http://127.0.0.1:8000") as first:
        first.headers.update({"X-Simple-Chat-Request": "1"})
        conversation = create_conversation(first, "gpt-5.6-terra")
        first_llm.queue_success("saved", "resp_saved")
        stream_events(first, conversation["id"], "history")

    settings.llm.models = [model for model in settings.llm.models if model.key != "gpt-5.6-terra"]
    second_llm = FakeLLM()
    app = create_app(settings, llm_service=second_llm, clock=clock)
    with TestClient(app, base_url="http://127.0.0.1:8000") as second:
        second.headers.update({"X-Simple-Chat-Request": "1"})
        detail = second.get(f"/api/conversations/{conversation['id']}").json()
        assert detail["model_key"] == "gpt-5.6-terra"
        assert detail["model_label"] == "gpt-5.6-terra"
        assert detail["model_available"] is False
        assert len(second.get(f"/api/conversations/{conversation['id']}/messages").json()) == 2
        blocked = second.post(
            f"/api/conversations/{conversation['id']}/messages", data={"text": "do not fallback"}
        )
        assert blocked.status_code == 400
        assert blocked.json()["error"]["code"] == "model_unavailable"
        assert second_llm.calls == []
        changed = second.patch(
            f"/api/conversations/{conversation['id']}", json={"model_key": "gpt-5.6-luna"}
        ).json()
        assert changed["context_epoch"] == 2


def test_title_and_patch_validation_are_atomic(client: TestClient, fake_llm: FakeLLM) -> None:
    conversation = create_conversation(client)
    original_updated = conversation["updated_at"]
    empty = client.patch(f"/api/conversations/{conversation['id']}", json={})
    blank = client.patch(f"/api/conversations/{conversation['id']}", json={"title": "   "})
    long = client.patch(f"/api/conversations/{conversation['id']}", json={"title": "x" * 101})
    atomic = client.patch(
        f"/api/conversations/{conversation['id']}",
        json={"title": "must not apply", "model_key": "missing"},
    )
    assert [empty.status_code, blank.status_code, long.status_code, atomic.status_code] == [
        400,
        400,
        400,
        400,
    ]
    unchanged = client.get(f"/api/conversations/{conversation['id']}").json()
    assert unchanged["title"] == "新しいチャット"
    assert unchanged["updated_at"] == original_updated

    custom = client.patch(
        f"/api/conversations/{conversation['id']}", json={"title": "手動タイトル"}
    ).json()
    fake_llm.queue_success("ok", "resp_title")
    stream_events(client, conversation["id"], "自動タイトル候補")
    assert client.get(f"/api/conversations/{conversation['id']}").json()["title"] == custom["title"]


def test_empty_conversation_model_change_and_same_model_are_timestamp_noops(
    client: TestClient,
) -> None:
    conversation = create_conversation(client)
    changed = client.patch(
        f"/api/conversations/{conversation['id']}", json={"model_key": "gpt-5.6-sol"}
    ).json()
    assert changed["context_epoch"] == 1
    same = client.patch(
        f"/api/conversations/{conversation['id']}", json={"model_key": "gpt-5.6-sol"}
    ).json()
    assert same["context_epoch"] == 1
    assert same["updated_at"] == changed["updated_at"]
