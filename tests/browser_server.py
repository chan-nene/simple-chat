from __future__ import annotations

import argparse
import itertools
from collections.abc import AsyncIterator
from pathlib import Path

import uvicorn

import run


class BrowserLLM:
    def __init__(self) -> None:
        self.ids = itertools.count(1)

    async def stream(self, **kwargs: object) -> AsyncIterator[run.StreamEvent]:
        response_id = f"resp_browser_{next(self.ids)}"
        yield run.StreamEvent("created", response_id=response_id)
        text = f"受信しました: {kwargs.get('text', '')}\n\n```python\nprint('ok')\n```"
        for piece in (text[:12], text[12:]):
            yield run.StreamEvent("delta", text=piece)
        yield run.StreamEvent("completed", response_id=response_id)

    async def delete_response(self, response_id: str) -> None:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    settings = run.load_settings("config.example.toml").model_copy(deep=True)
    settings.root = args.data.resolve()
    settings.server.port = args.port
    uvicorn.run(
        run.create_app(settings, llm_service=BrowserLLM()),
        host="127.0.0.1",
        port=args.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
