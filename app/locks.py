from __future__ import annotations

import asyncio


class ConversationLocks:
    """Single-process locks; the supported deployment is exactly one worker."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._generating: set[str] = set()

    def get(self, conversation_id: str) -> asyncio.Lock:
        return self._locks.setdefault(conversation_id, asyncio.Lock())

    def is_locked(self, conversation_id: str) -> bool:
        lock = self._locks.get(conversation_id)
        return bool(lock and lock.locked())

    def mark_generating(self, conversation_id: str) -> None:
        if not self.get(conversation_id).locked():
            raise RuntimeError("generation marker requires the conversation lock")
        self._generating.add(conversation_id)

    def is_generating(self, conversation_id: str) -> bool:
        return conversation_id in self._generating

    def finish_generation(self, conversation_id: str) -> None:
        self._generating.discard(conversation_id)
        lock = self.get(conversation_id)
        if lock.locked():
            lock.release()
