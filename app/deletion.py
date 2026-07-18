from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

from .config import Settings
from .domain import LLMService, ResponseAlreadyGone
from .image_service import ImageService
from .locks import ConversationLocks
from .repository import ChatRepository, DeletionSnapshot
from .time import Clock


logger = logging.getLogger(__name__)


class ConversationDeletionService:
    def __init__(
        self,
        repository: ChatRepository,
        image_service: ImageService,
        llm_service: LLMService | None,
        locks: ConversationLocks,
        settings: Settings,
        clock: Clock,
    ) -> None:
        self.repository = repository
        self.image_service = image_service
        self.llm_service = llm_service
        self.locks = locks
        self.settings = settings
        self.clock = clock

    async def delete_active(self, conversation_id: str) -> None:
        lock = self.locks.get(conversation_id)
        await lock.acquire()
        try:
            await self.delete_active_locked(conversation_id)
        finally:
            lock.release()

    async def delete_active_locked(self, conversation_id: str) -> None:
        """Delete an active conversation while its conversation lock is already held."""
        snapshot = await asyncio.to_thread(
            self.repository.deletion_snapshot, conversation_id, self.clock()
        )
        await self._delete(conversation_id, snapshot)

    async def cleanup_expired(self) -> int:
        removed = 0
        conversation_ids = await asyncio.to_thread(
            self.repository.expired_conversation_ids, self.clock()
        )
        for conversation_id in conversation_ids:
            lock = self.locks.get(conversation_id)
            if lock.locked():
                continue
            await lock.acquire()
            try:
                snapshot = await asyncio.to_thread(
                    self.repository.get_expired_deletion_snapshot,
                    conversation_id,
                    self.clock(),
                )
                if snapshot is None:
                    continue
                await self._delete(conversation_id, snapshot)
                removed += 1
            finally:
                lock.release()

        older_than = (self.clock() - timedelta(hours=24)).timestamp()
        referenced = await asyncio.to_thread(self.repository.referenced_attachment_names)
        await asyncio.to_thread(
            self.image_service.cleanup_orphans, referenced, older_than
        )
        return removed

    async def _delete(self, conversation_id: str, snapshot: DeletionSnapshot) -> None:
        if self.settings.retention.delete_remote_responses:
            if self.llm_service is None:
                if snapshot.response_ids:
                    logger.warning("Skipping remote response deletion: OpenAI API key is not configured")
            else:
                for response_id in snapshot.response_ids:
                    try:
                        await self.llm_service.delete_response(response_id)
                    except ResponseAlreadyGone:
                        pass
                    except Exception as exc:
                        logger.warning(
                            "Remote response deletion failed (%s)", type(exc).__name__
                        )
        await asyncio.to_thread(self.repository.delete_conversation_row, conversation_id)
        for path in snapshot.attachment_paths:
            try:
                await asyncio.to_thread(self.image_service.delete_path, path)
            except (OSError, RuntimeError) as exc:
                logger.warning("Attachment cleanup failed (%s)", type(exc).__name__)
