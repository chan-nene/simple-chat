from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable


Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def retention_cutoff(now: datetime, days: int) -> str:
    return utc_text(now - timedelta(days=days))

