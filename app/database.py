from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import Engine, create_engine, event, inspect, select
from sqlalchemy.orm import Session, sessionmaker

from .models import AppMeta, Base, Message


SCHEMA_VERSION = 1


class DatabaseError(RuntimeError):
    pass


class Database:
    def __init__(self, path: Path, *, wal: bool = True) -> None:
        self.path = path
        self.wal = wal
        self.engine: Engine | None = None
        self.session_factory: sessionmaker[Session] | None = None

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            f"sqlite:///{self.path.as_posix()}",
            connect_args={"check_same_thread": False, "timeout": 15},
            pool_pre_ping=True,
        )

        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragmas(dbapi_connection: object, _: object) -> None:
            cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=15000")
            if self.wal:
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()

        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)
        table_names = set(inspect(self.engine).get_table_names())
        if table_names and "app_meta" not in table_names:
            raise DatabaseError("database exists without schema metadata; refusing to guess its version")

        Base.metadata.create_all(self.engine)
        with self.session() as session:
            version = session.get(AppMeta, "schema_version")
            if version is None:
                session.add(AppMeta(key="schema_version", value=str(SCHEMA_VERSION)))
            elif int(version.value) != SCHEMA_VERSION:
                raise DatabaseError(
                    f"unsupported database schema {version.value}; expected {SCHEMA_VERSION}"
                )

    @contextmanager
    def session(self) -> Iterator[Session]:
        if self.session_factory is None:
            raise DatabaseError("database is not initialized")
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def recover_interrupted_streams(self, now: str) -> int:
        recovered = 0
        with self.session() as session:
            interrupted = list(
                session.scalars(select(Message).where(Message.status == "streaming"))
            )
            for assistant in interrupted:
                assistant.status = "failed"
                assistant.included_in_context = False
                assistant.error_code = "interrupted"
                assistant.error_message = "前回のアプリ終了により回答生成が中断されました。"
                assistant.updated_at = now
                user = session.scalar(
                    select(Message).where(
                        Message.turn_id == assistant.turn_id,
                        Message.role == "user",
                    )
                )
                if user is not None:
                    user.included_in_context = False
                    user.updated_at = now
                recovered += 1
        return recovered

    def close(self) -> None:
        if self.engine is not None:
            self.engine.dispose()
            self.engine = None
            self.session_factory = None
