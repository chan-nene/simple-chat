from __future__ import annotations

from types import SimpleNamespace

import run


def test_main_suppresses_ctrl_c_traceback(monkeypatch: object) -> None:
    settings = SimpleNamespace(server=SimpleNamespace(host="127.0.0.1", port=8000))
    app = object()
    monkeypatch.setattr(run, "load_settings", lambda: settings)
    monkeypatch.setattr(run, "create_app", lambda value: app)

    def interrupt(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(run.uvicorn, "run", interrupt)

    assert run.main() == 130


def test_main_returns_success_after_server_stops(monkeypatch: object) -> None:
    settings = SimpleNamespace(server=SimpleNamespace(host="127.0.0.1", port=8000))
    monkeypatch.setattr(run, "load_settings", lambda: settings)
    monkeypatch.setattr(run, "create_app", lambda value: object())
    monkeypatch.setattr(run.uvicorn, "run", lambda *args, **kwargs: None)

    assert run.main() == 0
