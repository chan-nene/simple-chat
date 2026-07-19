from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT, SettingsError, load_settings
from app.database import Database, DatabaseError
from app.repository import ChatRepository
from app.time import utc_text


CANONICAL = (PROJECT_ROOT / "config.example.toml").read_text(encoding="utf-8")


def write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "test-config.toml"
    path.write_text(content, encoding="utf-8")
    return path


def test_canonical_toml_and_paths_are_project_root_relative(tmp_path: Path) -> None:
    config = write_config(tmp_path, CANONICAL)
    settings = load_settings(config)
    assert settings.llm.default_model_key == "gpt-5.6-luna"
    assert settings.responses.store is True
    assert settings.database_path == (PROJECT_ROOT / "data/chat.db").resolve()
    assert settings.uploads_path == (PROJECT_ROOT / "data/uploads").resolve()


@pytest.mark.parametrize(
    "mutated",
    [
        CANONICAL + "\n[unexpected]\nvalue = true\n",
        CANONICAL.replace('key = "gpt-5.6-terra"', 'key = "gpt-5.6-luna"', 1),
        CANONICAL.replace('provider_model = "gpt-5.6-terra"', 'provider_model = "   "', 1),
        CANONICAL.replace('default_model_key = "gpt-5.6-luna"', 'default_model_key = "missing"'),
        CANONICAL.replace("store = true", "store = false"),
        CANONICAL.replace("history_days = 7", "history_days = 8"),
        CANONICAL.replace('host = "127.0.0.1"', 'host = "0.0.0.0"'),
        CANONICAL.replace("supports_streaming = true", "supports_streaming = false", 1),
        CANONICAL.replace("remove_exif = true", "remove_exif = false"),
        CANONICAL.replace('path = "data/chat.db"', 'path = "../outside.db"'),
        CANONICAL.replace('ai_icon = "/favicon.svg"', 'ai_icon = "https://example.com/icon.png"'),
        CANONICAL.replace('ai_icon = "/favicon.svg"', 'ai_icon = "/../outside.png"'),
        CANONICAL.replace('ai_icon = "/favicon.svg"', 'ai_icon = "/icon.txt"'),
    ],
)
def test_invalid_configuration_is_rejected(tmp_path: Path, mutated: str) -> None:
    with pytest.raises(SettingsError):
        load_settings(write_config(tmp_path, mutated))


def test_missing_config_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="Copy config.example.toml"):
        load_settings(tmp_path / "missing.toml")


def test_unknown_database_schema_is_rejected(settings: object) -> None:
    database = Database(settings.database_path)
    database.initialize()
    database.close()
    with sqlite3.connect(settings.database_path) as connection:
        connection.execute("UPDATE app_meta SET value = '99' WHERE key = 'schema_version'")
        connection.commit()
    with pytest.raises(DatabaseError, match="unsupported database schema"):
        Database(settings.database_path).initialize()


def test_database_without_schema_metadata_is_rejected(settings: object) -> None:
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(settings.database_path) as connection:
        connection.execute("CREATE TABLE mystery (id INTEGER PRIMARY KEY)")
        connection.commit()
    with pytest.raises(DatabaseError, match="without schema metadata"):
        Database(settings.database_path).initialize()


def test_schema_uses_required_attachment_columns_and_turn_scope(settings: object) -> None:
    database = Database(settings.database_path)
    database.initialize()
    database.close()
    with sqlite3.connect(settings.database_path) as connection:
        attachment_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(attachments)")
        }
        messages_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'messages'"
        ).fetchone()[0]
        conversation_indexes = {
            row[1] for row in connection.execute("PRAGMA index_list(conversations)")
        }
        message_indexes = {
            row[1] for row in connection.execute("PRAGMA index_list(messages)")
        }
    assert {"source_byte_size", "byte_size"} <= attachment_columns
    assert "UNIQUE (conversation_id, turn_id, role)" in messages_sql
    assert "ix_conversations_updated_id" in conversation_indexes
    assert "ix_messages_conversation_created_id" in message_indexes


def test_startup_recovers_streaming_turn_without_changing_last_good_response(
    settings: object,
) -> None:
    now = datetime(2026, 7, 18, 6, 0, tzinfo=timezone.utc)
    database = Database(settings.database_path)
    database.initialize()
    repository = ChatRepository(database, settings)
    model = settings.llm.enabled_models[settings.llm.default_model_key]
    conversation = repository.create_conversation(model, now)
    _, first = repository.start_turn(
        conversation.id, text_content="first", attachments=[], now=now
    )
    repository.complete_turn(
        first.assistant_message_id, content="ok", response_id="resp_good", now=now
    )
    _, interrupted = repository.start_turn(
        conversation.id, text_content="second", attachments=[], now=now
    )

    assert database.recover_interrupted_streams(utc_text(now)) == 1
    messages = repository.list_messages(conversation.id, now)
    recovered = next(message for message in messages if message.id == interrupted.assistant_message_id)
    paired = next(message for message in messages if message.turn_id == recovered.turn_id and message.role == "user")
    assert recovered.status == "failed"
    assert recovered.error_code == "interrupted"
    assert recovered.included_in_context is False
    assert paired.included_in_context is False
    assert repository.get_conversation(conversation.id, now).latest_response_id == "resp_good"
    database.close()
