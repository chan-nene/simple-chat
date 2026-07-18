from __future__ import annotations

from .config import Settings
from .models import Attachment, Conversation, Message


def conversation_json(conversation: Conversation, settings: Settings) -> dict[str, object]:
    configured = settings.llm.enabled_models.get(conversation.model_key)
    messages = list(conversation.messages)
    return {
        "id": conversation.id,
        "title": conversation.title,
        "provider": conversation.provider,
        "model_key": conversation.model_key,
        "model_label": configured.label if configured else conversation.model_key,
        "model_available": configured is not None,
        "context_epoch": conversation.context_epoch,
        "has_messages": bool(messages),
        "is_generating": any(message.status == "streaming" for message in messages),
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
    }


def message_json(message: Message) -> dict[str, object]:
    return {
        "id": message.id,
        "role": message.role,
        "content": message.content,
        "status": message.status,
        "context_epoch": message.context_epoch,
        "included_in_context": message.included_in_context,
        "error_code": message.error_code,
        "error_message": message.error_message,
        "created_at": message.created_at,
        "updated_at": message.updated_at,
        "attachments": [attachment_json(value) for value in message.attachments],
    }


def attachment_json(attachment: Attachment) -> dict[str, object]:
    return {
        "id": attachment.id,
        "original_name": attachment.original_name,
        "stored_mime_type": attachment.stored_mime_type,
        "width": attachment.width,
        "height": attachment.height,
        "byte_size": attachment.byte_size,
        "content_url": f"/api/attachments/{attachment.id}/content",
    }
