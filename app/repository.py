from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import delete, func, select, text
from sqlalchemy.orm import selectinload

from .config import ModelSettings, Settings
from .database import Database
from .domain import StagedAttachment, TurnIds
from .errors import not_found
from .models import Attachment, Conversation, Message
from .time import retention_cutoff, utc_text


NEW_CHAT_TITLE = "新しいチャット"


@dataclass(frozen=True, slots=True)
class DeletionSnapshot:
    response_ids: tuple[str, ...]
    attachment_paths: tuple[Path, ...]


class ChatRepository:
    def __init__(self, database: Database, settings: Settings) -> None:
        self.database = database
        self.settings = settings

    def _cutoff(self, now: datetime) -> str:
        return retention_cutoff(now, self.settings.retention.history_days)

    def create_conversation(self, model: ModelSettings, now: datetime) -> Conversation:
        timestamp = utc_text(now)
        conversation = Conversation(
            id=str(uuid.uuid4()),
            title=NEW_CHAT_TITLE,
            latest_response_id=None,
            provider=self.settings.llm.provider,
            model_key=model.key,
            model_target=model.provider_model,
            instructions=self.settings.responses.instructions,
            compaction_enabled=self.settings.responses.compaction.enabled,
            compact_threshold=self.settings.responses.compaction.compact_threshold,
            context_epoch=1,
            created_at=timestamp,
            updated_at=timestamp,
            messages=[],
        )
        with self.database.session() as session:
            session.add(conversation)
        return conversation

    def list_conversations(self, now: datetime) -> list[Conversation]:
        with self.database.session() as session:
            return list(
                session.scalars(
                    select(Conversation)
                    .where(Conversation.updated_at >= self._cutoff(now))
                    .options(selectinload(Conversation.messages))
                    .order_by(Conversation.updated_at.desc(), Conversation.id.asc())
                )
            )

    def get_conversation(self, conversation_id: str, now: datetime) -> Conversation:
        with self.database.session() as session:
            conversation = session.scalar(
                select(Conversation)
                .where(
                    Conversation.id == conversation_id,
                    Conversation.updated_at >= self._cutoff(now),
                )
                .options(selectinload(Conversation.messages))
            )
            if conversation is None:
                raise not_found()
            return conversation

    def update_conversation(
        self,
        conversation_id: str,
        *,
        title: str | None,
        model: ModelSettings | None,
        now: datetime,
    ) -> Conversation:
        with self.database.session() as session:
            conversation = session.scalar(
                select(Conversation)
                .where(
                    Conversation.id == conversation_id,
                    Conversation.updated_at >= self._cutoff(now),
                )
                .options(selectinload(Conversation.messages))
            )
            if conversation is None:
                raise not_found()

            changed = False
            if title is not None and title != conversation.title:
                conversation.title = title
                changed = True
            if model is not None and model.key != conversation.model_key:
                if conversation.messages:
                    conversation.latest_response_id = None
                    conversation.context_epoch += 1
                conversation.provider = self.settings.llm.provider
                conversation.model_key = model.key
                conversation.model_target = model.provider_model
                changed = True
            if changed:
                conversation.updated_at = monotonic_timestamp(now, conversation.updated_at)
            return conversation

    def list_messages(self, conversation_id: str, now: datetime) -> list[Message]:
        # Looking up the conversation first enforces logical expiry consistently.
        self.get_conversation(conversation_id, now)
        with self.database.session() as session:
            return list(
                session.scalars(
                    select(Message)
                    .where(Message.conversation_id == conversation_id)
                    .options(selectinload(Message.attachments))
                    .order_by(Message.created_at.asc(), Message.id.asc())
                )
            )

    def is_generating(self, conversation_id: str, now: datetime) -> bool:
        self.get_conversation(conversation_id, now)
        with self.database.session() as session:
            return bool(
                session.scalar(
                    select(func.count(Message.id)).where(
                        Message.conversation_id == conversation_id,
                        Message.status == "streaming",
                    )
                )
            )

    def start_turn(
        self,
        conversation_id: str,
        *,
        text_content: str,
        attachments: list[StagedAttachment],
        now: datetime,
    ) -> tuple[Conversation, TurnIds]:
        with self.database.session() as session:
            conversation = session.scalar(
                select(Conversation).where(
                    Conversation.id == conversation_id,
                    Conversation.updated_at >= self._cutoff(now),
                )
            )
            if conversation is None:
                raise not_found()

            user_timestamp = monotonic_timestamp(now, conversation.updated_at)
            assistant_timestamp = utc_text(parse_timestamp(user_timestamp) + timedelta(milliseconds=1))

            has_messages = bool(
                session.scalar(
                    select(func.count(Message.id)).where(Message.conversation_id == conversation_id)
                )
            )
            if not has_messages and conversation.title == NEW_CHAT_TITLE:
                conversation.title = derive_title(text_content)

            turn_id = str(uuid.uuid4())
            user_id = str(uuid.uuid4())
            assistant_id = str(uuid.uuid4())
            user = Message(
                id=user_id,
                conversation_id=conversation_id,
                turn_id=turn_id,
                role="user",
                content=text_content,
                status="completed",
                response_id=None,
                context_epoch=conversation.context_epoch,
                included_in_context=False,
                error_code=None,
                error_message=None,
                created_at=user_timestamp,
                updated_at=user_timestamp,
            )
            assistant = Message(
                id=assistant_id,
                conversation_id=conversation_id,
                turn_id=turn_id,
                role="assistant",
                content="",
                status="streaming",
                response_id=None,
                context_epoch=conversation.context_epoch,
                included_in_context=False,
                error_code=None,
                error_message=None,
                created_at=assistant_timestamp,
                updated_at=assistant_timestamp,
            )
            session.add_all([user, assistant])
            for staged in attachments:
                session.add(
                    Attachment(
                        id=staged.id,
                        message_id=user_id,
                        original_name=staged.original_name,
                        stored_name=staged.stored_name,
                        original_mime_type=staged.original_mime_type,
                        stored_mime_type=staged.stored_mime_type,
                        width=staged.width,
                        height=staged.height,
                        source_byte_size=staged.source_byte_size,
                        byte_size=staged.byte_size,
                        sha256=staged.sha256,
                        created_at=user_timestamp,
                    )
                )
            conversation.updated_at = assistant_timestamp
            return conversation, TurnIds(user_id, assistant_id, conversation.context_epoch)

    def complete_turn(
        self,
        assistant_message_id: str,
        *,
        content: str,
        response_id: str,
        now: datetime,
    ) -> None:
        with self.database.session() as session:
            assistant = session.get(Message, assistant_message_id)
            if assistant is None or assistant.status != "streaming":
                raise RuntimeError("streaming assistant message is missing")
            user = session.scalar(
                select(Message).where(Message.turn_id == assistant.turn_id, Message.role == "user")
            )
            conversation = session.get(Conversation, assistant.conversation_id)
            if user is None or conversation is None:
                raise RuntimeError("turn consistency invariant failed")

            timestamp = monotonic_timestamp(now, conversation.updated_at)

            assistant.content = content
            assistant.status = "completed"
            assistant.response_id = response_id
            assistant.included_in_context = True
            assistant.error_code = None
            assistant.error_message = None
            assistant.updated_at = timestamp
            user.included_in_context = True
            user.updated_at = timestamp
            conversation.latest_response_id = response_id
            conversation.updated_at = timestamp

    def fail_turn(
        self,
        assistant_message_id: str,
        *,
        content: str,
        response_id: str | None,
        code: str,
        message: str,
        cancelled: bool,
        reset_context: bool,
        now: datetime,
    ) -> None:
        with self.database.session() as session:
            assistant = session.get(Message, assistant_message_id)
            if assistant is None or assistant.status != "streaming":
                return
            user = session.scalar(
                select(Message).where(Message.turn_id == assistant.turn_id, Message.role == "user")
            )
            conversation = session.get(Conversation, assistant.conversation_id)
            if user is None or conversation is None:
                raise RuntimeError("turn consistency invariant failed")

            timestamp = monotonic_timestamp(now, conversation.updated_at)

            assistant.content = content
            assistant.status = "cancelled" if cancelled else "failed"
            assistant.response_id = response_id
            assistant.included_in_context = False
            assistant.error_code = code
            assistant.error_message = message
            assistant.updated_at = timestamp
            user.included_in_context = False
            user.updated_at = timestamp
            if reset_context:
                conversation.latest_response_id = None
                conversation.context_epoch += 1
            conversation.updated_at = timestamp

    def deletion_snapshot(self, conversation_id: str, now: datetime) -> DeletionSnapshot:
        self.get_conversation(conversation_id, now)
        with self.database.session() as session:
            conversation = session.get(Conversation, conversation_id)
            assert conversation is not None
            ids = set(
                session.scalars(
                    select(Message.response_id).where(
                        Message.conversation_id == conversation_id,
                        Message.response_id.is_not(None),
                    )
                )
            )
            if conversation.latest_response_id:
                ids.add(conversation.latest_response_id)
            names = list(
                session.scalars(
                    select(Attachment.stored_name)
                    .join(Message, Attachment.message_id == Message.id)
                    .where(Message.conversation_id == conversation_id)
                )
            )
            paths = tuple(self.settings.uploads_path / name for name in names)
            return DeletionSnapshot(tuple(sorted(value for value in ids if value)), paths)

    def delete_conversation_row(self, conversation_id: str) -> None:
        with self.database.session() as session:
            session.execute(delete(Conversation).where(Conversation.id == conversation_id))

    def expired_conversation_ids(self, now: datetime) -> list[str]:
        with self.database.session() as session:
            return list(
                session.scalars(
                    select(Conversation.id)
                    .where(Conversation.updated_at < self._cutoff(now))
                    .order_by(Conversation.updated_at.asc(), Conversation.id.asc())
                )
            )

    def get_expired_deletion_snapshot(
        self, conversation_id: str, now: datetime
    ) -> DeletionSnapshot | None:
        with self.database.session() as session:
            conversation = session.scalar(
                select(Conversation).where(
                    Conversation.id == conversation_id,
                    Conversation.updated_at < self._cutoff(now),
                )
            )
            if conversation is None:
                return None
            ids = set(
                session.scalars(
                    select(Message.response_id).where(
                        Message.conversation_id == conversation_id,
                        Message.response_id.is_not(None),
                    )
                )
            )
            if conversation.latest_response_id:
                ids.add(conversation.latest_response_id)
            names = list(
                session.scalars(
                    select(Attachment.stored_name)
                    .join(Message, Attachment.message_id == Message.id)
                    .where(Message.conversation_id == conversation_id)
                )
            )
            return DeletionSnapshot(
                tuple(sorted(value for value in ids if value)),
                tuple(self.settings.uploads_path / name for name in names),
            )

    def get_attachment(self, attachment_id: str, now: datetime) -> Attachment:
        with self.database.session() as session:
            attachment = session.scalar(
                select(Attachment)
                .join(Message, Attachment.message_id == Message.id)
                .join(Conversation, Message.conversation_id == Conversation.id)
                .where(
                    Attachment.id == attachment_id,
                    Conversation.updated_at >= self._cutoff(now),
                )
            )
            if attachment is None:
                raise not_found()
            return attachment

    def referenced_attachment_names(self) -> set[str]:
        with self.database.session() as session:
            return set(session.scalars(select(Attachment.stored_name)))

    def check_database(self) -> None:
        with self.database.session() as session:
            session.execute(text("SELECT 1"))


def derive_title(content: str) -> str:
    normalized = re.sub(r"\s+", " ", content.replace("\r", " ").replace("\n", " ")).strip()
    return normalized[:40] if normalized else NEW_CHAT_TITLE


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def monotonic_timestamp(now: datetime, floor: str) -> str:
    minimum = parse_timestamp(floor) + timedelta(milliseconds=1)
    return utc_text(max(now, minimum))
