from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from app.config import load_settings
from app.domain import LLMStreamEvent
from app.main import create_app


class BrowserDemoLLM:
    async def stream_response(self, **kwargs: object) -> AsyncIterator[LLMStreamEvent]:
        response_id = f"resp_browser_{uuid.uuid4().hex}"
        text = str(kwargs.get("user_text", ""))
        answer = (
            "## 観測結果\n\n"
            f"受け取ったメッセージは **{text or '画像入力'}** です。\n\n"
            "```python\n"
            "def context_chain(previous_response_id):\n"
            "    return previous_response_id or 'new epoch'\n"
            "```\n\n"
            "履歴本文を再送せず、Response ID で文脈を継続します。"
        )
        delay = 0.025
        chunk_size = 7
        if text == "__slow__":
            answer = "停止検証用の長い応答です。" * 120
            delay = 0.08
            chunk_size = 5
        elif text == "__long__":
            answer = "\n\n".join(f"観測行 {index}: Response chain remains stable." for index in range(120))
            delay = 0.02
            chunk_size = 40
        yield LLMStreamEvent("created", response_id=response_id)
        for offset in range(0, len(answer), chunk_size):
            await asyncio.sleep(delay)
            yield LLMStreamEvent("delta", text=answer[offset : offset + chunk_size])
        yield LLMStreamEvent("completed", response_id=response_id)

    async def delete_response(self, response_id: str) -> None:
        return None


settings = load_settings("config.example.toml").model_copy(deep=True)
settings.project_root = Path(os.environ.get("SIMPLE_CHAT_BROWSER_ROOT", "C:/tmp/simple-chat-browser"))
app = create_app(settings, llm_service=BrowserDemoLLM())
