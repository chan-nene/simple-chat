from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_settings  # noqa: E402
from app.domain import LLMServiceError, ResponseAlreadyGone  # noqa: E402
from app.openai_service import OpenAIResponsesService  # noqa: E402


async def run_smoke() -> int:
    if os.getenv("SIMPLE_CHAT_RUN_REAL_API") != "1":
        print("実API疎通は無効です。SIMPLE_CHAT_RUN_REAL_API=1 を明示してください。")
        return 2

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("OPENAI_API_KEY が設定されていません。")
        return 2

    settings = load_settings()
    model = settings.llm.enabled_models[settings.llm.default_model_key]
    service = OpenAIResponsesService(api_key)
    response_id: str | None = None
    completed = False
    output_length = 0

    try:
        async for event in service.stream_response(
            model_target=model.provider_model,
            user_text="疎通確認です。「OK」とだけ回答してください。",
            image_paths=[],
            previous_response_id=None,
            instructions=settings.responses.instructions,
            max_output_tokens=min(settings.responses.max_output_tokens, 64),
            compaction=(
                settings.responses.compaction
                if settings.responses.compaction.enabled
                else None
            ),
        ):
            if event.type == "created":
                response_id = event.response_id
            elif event.type == "delta":
                output_length += len(event.text)
            elif event.type == "completed":
                response_id = event.response_id or response_id
                completed = True
    except LLMServiceError as exc:
        print(f"実API疎通に失敗しました: {exc.code} / {exc.message}")
        return 1
    except Exception as exc:
        print(f"実API疎通に失敗しました: {type(exc).__name__}")
        return 1
    finally:
        if response_id:
            try:
                await service.delete_response(response_id)
            except ResponseAlreadyGone:
                pass
            except Exception as exc:
                print(f"作成したResponseの後始末に失敗しました: {type(exc).__name__}")

    if not completed:
        print("実API疎通に失敗しました: 正常完了イベントがありません。")
        return 1
    print(f"実API疎通に成功しました: model_key={model.key}, output_chars={output_length}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_smoke()))
