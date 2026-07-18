from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.database import Database
from app.deletion import ConversationDeletionService
from app.domain import ResponseAlreadyGone
from app.image_service import ImageService
from app.locks import ConversationLocks
from app.main import create_app
from app.repository import ChatRepository
from app.time import utc_text
from tests.conftest import FakeLLM, MutableClock, create_conversation, stream_events
from tests.test_images_security import image_bytes


class DeleteFailureLLM(FakeLLM):
    async def delete_response(self, response_id: str) -> None:
        self.deleted.append(response_id)
        raise TimeoutError("secret upstream detail")


class AlreadyGoneLLM(FakeLLM):
    async def delete_response(self, response_id: str) -> None:
        self.deleted.append(response_id)
        raise ResponseAlreadyGone()


def make_client(settings: object, clock: MutableClock, fake: FakeLLM) -> TestClient:
    client = TestClient(
        create_app(settings, llm_service=fake, clock=clock),
        base_url="http://127.0.0.1:8000",
    )
    client.headers.update({"X-Simple-Chat-Request": "1"})
    return client


def test_manual_delete_deduplicates_response_ids_and_removes_attachment(
    client: TestClient, fake_llm: FakeLLM
) -> None:
    conversation = create_conversation(client)
    fake_llm.queue_success("with image", "resp_same")
    response = client.post(
        f"/api/conversations/{conversation['id']}/messages",
        data={"text": "image"},
        files={"images": ("image.png", image_bytes(), "image/png")},
    )
    assert response.status_code == 200
    stored_path = fake_llm.calls[0]["image_paths"][0]
    assert stored_path.is_file()

    deleted = client.delete(f"/api/conversations/{conversation['id']}")
    assert deleted.status_code == 204
    assert fake_llm.deleted == ["resp_same"]
    assert not stored_path.exists()
    assert client.get(f"/api/conversations/{conversation['id']}").status_code == 404


@pytest.mark.parametrize("fake_class", [DeleteFailureLLM, AlreadyGoneLLM])
def test_remote_delete_failure_never_blocks_local_delete(
    settings: object,
    clock: MutableClock,
    fake_class: type[FakeLLM],
) -> None:
    fake = fake_class()
    with make_client(settings, clock, fake) as client:
        conversation = create_conversation(client)
        fake.queue_success("ok", "resp_remote")
        stream_events(client, conversation["id"], "hello")
        response = client.delete(f"/api/conversations/{conversation['id']}")
        assert response.status_code == 204
        assert client.get(f"/api/conversations/{conversation['id']}").status_code == 404
        assert fake.deleted == ["resp_remote"]


def test_retention_boundary_is_active_but_past_boundary_is_hidden(
    settings: object, clock: MutableClock, fake_llm: FakeLLM
) -> None:
    with make_client(settings, clock, fake_llm) as client:
        conversation = create_conversation(client)
        updated = datetime.fromisoformat(conversation["updated_at"].replace("Z", "+00:00"))
        clock.value = updated + timedelta(days=7)
        assert client.get(f"/api/conversations/{conversation['id']}").status_code == 200
        clock.value += timedelta(milliseconds=1)
        assert client.get(f"/api/conversations/{conversation['id']}").status_code == 404
        assert client.get("/api/conversations").json() == []


def test_expired_conversation_cannot_be_resurrected_by_any_mutation(
    settings: object, clock: MutableClock, fake_llm: FakeLLM
) -> None:
    with make_client(settings, clock, fake_llm) as client:
        conversation = create_conversation(client)
        original_updated = conversation["updated_at"]
        clock.value += timedelta(days=7, milliseconds=1)
        patch = client.patch(
            f"/api/conversations/{conversation['id']}", json={"title": "resurrect"}
        )
        remove = client.delete(f"/api/conversations/{conversation['id']}")
        send = client.post(
            f"/api/conversations/{conversation['id']}/messages", data={"text": "resurrect"}
        )
        assert (patch.status_code, remove.status_code, send.status_code) == (404, 404, 404)
        with sqlite3.connect(settings.database_path) as connection:
            stored_updated = connection.execute(
                "SELECT updated_at FROM conversations WHERE id = ?", (conversation["id"],)
            ).fetchone()[0]
        assert stored_updated == original_updated


@pytest.mark.asyncio
async def test_auto_cleanup_skips_locked_conversation_then_deletes_next_time(
    settings: object,
) -> None:
    old_time = datetime(2026, 6, 1, tzinfo=timezone.utc)
    current_time = old_time + timedelta(days=8)
    clock = MutableClock(current_time)
    database = Database(settings.database_path)
    database.initialize()
    image_service = ImageService(settings)
    image_service.initialize()
    repository = ChatRepository(database, settings)
    model = settings.llm.enabled_models[settings.llm.default_model_key]
    conversation = repository.create_conversation(model, old_time)
    locks = ConversationLocks()
    service = ConversationDeletionService(
        repository, image_service, None, locks, settings, clock
    )
    lock = locks.get(conversation.id)
    await lock.acquire()
    try:
        assert await service.cleanup_expired() == 0
        with database.session() as session:
            assert session.get(__import__("app.models", fromlist=["Conversation"]).Conversation, conversation.id)
    finally:
        lock.release()
    assert await service.cleanup_expired() == 1
    with database.session() as session:
        assert session.get(__import__("app.models", fromlist=["Conversation"]).Conversation, conversation.id) is None
    database.close()


@pytest.mark.asyncio
async def test_auto_cleanup_rechecks_expiry_after_acquiring_lock(settings: object) -> None:
    old_time = datetime(2026, 6, 1, tzinfo=timezone.utc)
    current_time = old_time + timedelta(days=8)
    clock = MutableClock(current_time)
    database = Database(settings.database_path)
    database.initialize()
    image_service = ImageService(settings)
    image_service.initialize()
    repository = ChatRepository(database, settings)
    model = settings.llm.enabled_models[settings.llm.default_model_key]
    conversation = repository.create_conversation(model, old_time)
    original_list = repository.expired_conversation_ids

    def list_then_refresh(now: object) -> list[str]:
        identifiers = original_list(now)
        # Simulate an update transaction that started while the conversation was active
        # and committed after the cleanup candidate query returned.
        with sqlite3.connect(settings.database_path) as connection:
            connection.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                ("refreshed", utc_text(current_time), conversation.id),
            )
            connection.commit()
        return identifiers

    repository.expired_conversation_ids = list_then_refresh  # type: ignore[method-assign]
    service = ConversationDeletionService(
        repository, image_service, None, ConversationLocks(), settings, clock
    )
    assert await service.cleanup_expired() == 0
    assert repository.get_conversation(conversation.id, current_time).title == "refreshed"
    database.close()


def test_startup_cleanup_removes_expired_conversation_even_without_api_key(
    settings: object, clock: MutableClock
) -> None:
    database = Database(settings.database_path)
    database.initialize()
    repository = ChatRepository(database, settings)
    model = settings.llm.enabled_models[settings.llm.default_model_key]
    old = clock.value - timedelta(days=8)
    conversation = repository.create_conversation(model, old)
    _, turn = repository.start_turn(
        conversation.id, text_content="old", attachments=[], now=old
    )
    repository.complete_turn(
        turn.assistant_message_id, content="old", response_id="resp_old", now=old
    )
    database.close()

    with TestClient(
        create_app(settings, llm_service=None, clock=clock),
        base_url="http://127.0.0.1:8000",
    ) as client:
        assert client.get("/api/conversations").json() == []
    with sqlite3.connect(settings.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 0


def test_orphan_cleanup_obeys_age_name_and_reference_boundaries(settings: object) -> None:
    service = ImageService(settings)
    service.initialize()
    threshold = datetime(2026, 7, 18, tzinfo=timezone.utc).timestamp()
    old_upload = service.uploads_dir / f"{uuid.uuid4()}.webp"
    exact_upload = service.uploads_dir / f"{uuid.uuid4()}.webp"
    referenced_upload = service.uploads_dir / f"{uuid.uuid4()}.webp"
    old_temp = service.tmp_dir / f"{uuid.uuid4()}.upload"
    invalid_name = service.uploads_dir / "do-not-delete.webp"
    outside = settings.project_root / f"{uuid.uuid4()}.webp"
    for path in (old_upload, exact_upload, referenced_upload, old_temp, invalid_name, outside):
        path.write_bytes(b"test")
    for path in (old_upload, referenced_upload, old_temp, invalid_name, outside):
        os.utime(path, (threshold - 1, threshold - 1))
    os.utime(exact_upload, (threshold, threshold))

    removed = service.cleanup_orphans({referenced_upload.name}, threshold)
    assert removed == 2
    assert not old_upload.exists()
    assert not old_temp.exists()
    assert exact_upload.exists()
    assert referenced_upload.exists()
    assert invalid_name.exists()
    assert outside.exists()


def test_symlinked_storage_component_is_rejected(settings: object) -> None:
    data = settings.project_root / "data"
    target = data / "real-uploads"
    target.mkdir(parents=True)
    declared = data / "linked-uploads"
    try:
        declared.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symbolic links are not available in this Windows environment")
    settings.storage.upload_directory = "data/linked-uploads"
    service = ImageService(settings)
    with pytest.raises(RuntimeError, match="symbolic link"):
        service.initialize()
