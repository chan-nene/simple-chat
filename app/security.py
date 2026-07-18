from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Request
from starlette.responses import JSONResponse, Response

from .errors import AppError


SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self' blob: data:; connect-src 'self'; object-src 'none'; "
        "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


class RequestTooLarge(Exception):
    pass


class RequestSizeLimitMiddleware:
    def __init__(self, app: Any, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Callable[..., Awaitable[Any]], send: Callable[..., Awaitable[Any]]) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        received = 0

        async def limited_receive() -> dict[str, Any]:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise RequestTooLarge()
            return message

        await self.app(scope, limited_receive, send)


def install_security_middleware(app: Any, *, port: int, max_request_bytes: int) -> None:
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    allowed_origins = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}

    @app.middleware("http")
    async def security(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        request.state.correlation_id = str(uuid.uuid4())
        host = request.headers.get("host", "")
        if host not in allowed_hosts:
            response = JSONResponse(
                AppError("invalid_request", "許可されていないHostです。", 400, False).payload(),
                status_code=400,
            )
        elif request.method in {"POST", "PATCH", "DELETE"} and request.headers.get(
            "x-simple-chat-request"
        ) != "1":
            response = JSONResponse(
                AppError("invalid_request", "必要なリクエストヘッダーがありません。", 400).payload(),
                status_code=400,
            )
        elif request.method in {"POST", "PATCH", "DELETE"} and (
            (origin := request.headers.get("origin")) is not None and origin not in allowed_origins
        ):
            response = JSONResponse(
                AppError("invalid_request", "許可されていないOriginです。", 400).payload(),
                status_code=400,
            )
        else:
            response = await call_next(request)

        for key, value in SECURITY_HEADERS.items():
            response.headers[key] = value
        if request.url.path.startswith("/vendor/"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        else:
            response.headers["Cache-Control"] = "no-store"
        response.headers["X-Request-ID"] = request.state.correlation_id
        return response

    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=max_request_bytes)


def request_too_large_response(_: Request, __: RequestTooLarge) -> JSONResponse:
    error = AppError(
        "payload_too_large", "リクエスト全体の容量が上限を超えています。", 413, False
    )
    return JSONResponse(error.payload(), status_code=413)
