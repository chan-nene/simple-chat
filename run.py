from __future__ import annotations

import sys

import uvicorn

from app.config import SettingsError, load_settings
from app.main import create_app


def main() -> int:
    try:
        settings = load_settings()
    except SettingsError as exc:
        print(f"設定エラー: {exc}", file=sys.stderr)
        return 2
    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.server.host,
        port=settings.server.port,
        workers=1,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
