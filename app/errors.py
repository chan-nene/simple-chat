from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AppError(Exception):
    code: str
    message: str
    status_code: int = 400
    retryable: bool = False

    def __str__(self) -> str:
        return self.message

    def payload(self) -> dict[str, object]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "retryable": self.retryable,
            }
        }


INVALID_REQUEST = "入力内容を確認してください。"
NOT_FOUND = "対象が見つかりません。"


def invalid_request(message: str = INVALID_REQUEST) -> AppError:
    return AppError("invalid_request", message, 400, False)


def not_found() -> AppError:
    return AppError("not_found", NOT_FOUND, 404, False)

