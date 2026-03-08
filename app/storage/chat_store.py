from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteChatStore:
    def __init__(self, database_path: str) -> None:
        self._path = Path(database_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._bootstrap()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self._path))
        connection.row_factory = sqlite3.Row
        return connection

    def _bootstrap(self) -> None:
        with self._lock:
            with self._connect() as connection:
                cursor = connection.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        chat_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        archived INTEGER NOT NULL DEFAULT 0,
                        defaults_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        message_id TEXT PRIMARY KEY,
                        chat_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        mode TEXT,
                        run_id TEXT,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL
                    );
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_files (
                        file_id TEXT PRIMARY KEY,
                        chat_id TEXT NOT NULL,
                        message_id TEXT NOT NULL,
                        uri TEXT NOT NULL,
                        source_type TEXT NOT NULL DEFAULT 'mixed',
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL
                    );
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_runs (
                        run_id TEXT PRIMARY KEY,
                        chat_id TEXT NOT NULL,
                        message_id TEXT NOT NULL,
                        state TEXT NOT NULL DEFAULT 'running',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    """
                )
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_chat_id ON chat_messages(chat_id, created_at);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_files_chat_id ON chat_files(chat_id, created_at);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_runs_chat_id ON chat_runs(chat_id, updated_at);")
                connection.commit()

    def create_session(self, *, title: str | None = None, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
        chat_id = str(uuid4())
        now = _utc_now_iso()
        resolved_title = (title or "").strip() or f"Chat {now[:19].replace('T', ' ')}"
        payload = {
            "chat_id": chat_id,
            "title": resolved_title,
            "archived": False,
            "defaults": defaults or {},
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO chat_sessions(chat_id, title, archived, defaults_json, created_at, updated_at)
                    VALUES (?, ?, 0, ?, ?, ?)
                    """,
                    (chat_id, resolved_title, json.dumps(payload["defaults"]), now, now),
                )
                connection.commit()
        return payload

    def list_sessions(self, *, include_archived: bool = False, limit: int = 200) -> list[dict[str, Any]]:
        predicate = "" if include_archived else "WHERE archived = 0"
        with self._lock:
            with self._connect() as connection:
                rows = connection.execute(
                    f"""
                    SELECT chat_id, title, archived, defaults_json, created_at, updated_at
                    FROM chat_sessions
                    {predicate}
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (max(limit, 1),),
                ).fetchall()
        return [self._row_to_session(row) for row in rows]

    def get_session(self, chat_id: str) -> dict[str, Any] | None:
        with self._lock:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT chat_id, title, archived, defaults_json, created_at, updated_at
                    FROM chat_sessions
                    WHERE chat_id = ?
                    """,
                    (chat_id,),
                ).fetchone()
        return self._row_to_session(row) if row else None

    def update_session(
        self,
        *,
        chat_id: str,
        title: str | None = None,
        archived: bool | None = None,
        defaults: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        current = self.get_session(chat_id)
        if current is None:
            return None
        next_title = title.strip() if isinstance(title, str) and title.strip() else current["title"]
        next_archived = bool(archived) if archived is not None else bool(current["archived"])
        next_defaults = defaults if defaults is not None else dict(current["defaults"])
        now = _utc_now_iso()
        with self._lock:
            with self._connect() as connection:
                connection.execute(
                    """
                    UPDATE chat_sessions
                    SET title = ?, archived = ?, defaults_json = ?, updated_at = ?
                    WHERE chat_id = ?
                    """,
                    (next_title, 1 if next_archived else 0, json.dumps(next_defaults), now, chat_id),
                )
                connection.commit()
        return self.get_session(chat_id)

    def touch_session(self, chat_id: str) -> None:
        now = _utc_now_iso()
        with self._lock:
            with self._connect() as connection:
                connection.execute(
                    "UPDATE chat_sessions SET updated_at = ? WHERE chat_id = ?",
                    (now, chat_id),
                )
                connection.commit()

    def add_message(
        self,
        *,
        chat_id: str,
        role: str,
        content: str,
        mode: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        message_id = str(uuid4())
        now = _utc_now_iso()
        payload = {
            "message_id": message_id,
            "chat_id": chat_id,
            "role": role,
            "content": content,
            "mode": mode or "",
            "run_id": run_id or "",
            "metadata": metadata or {},
            "created_at": now,
        }
        with self._lock:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO chat_messages(message_id, chat_id, role, content, mode, run_id, metadata_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message_id,
                        chat_id,
                        role,
                        content,
                        payload["mode"],
                        payload["run_id"],
                        json.dumps(payload["metadata"]),
                        now,
                    ),
                )
                connection.execute(
                    "UPDATE chat_sessions SET updated_at = ? WHERE chat_id = ?",
                    (now, chat_id),
                )
                connection.commit()
        return payload

    def list_messages(self, *, chat_id: str, limit: int = 500) -> list[dict[str, Any]]:
        with self._lock:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT message_id, chat_id, role, content, mode, run_id, metadata_json, created_at
                    FROM chat_messages
                    WHERE chat_id = ?
                    ORDER BY created_at ASC
                    LIMIT ?
                    """,
                    (chat_id, max(limit, 1)),
                ).fetchall()
        return [self._row_to_message(row) for row in rows]

    def add_files(
        self,
        *,
        chat_id: str,
        message_id: str,
        sources: list[str],
        source_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not sources:
            return []
        now = _utc_now_iso()
        rows: list[dict[str, Any]] = []
        with self._lock:
            with self._connect() as connection:
                for source in sources:
                    file_id = str(uuid4())
                    payload = {
                        "file_id": file_id,
                        "chat_id": chat_id,
                        "message_id": message_id,
                        "uri": source,
                        "source_type": source_type,
                        "metadata": metadata or {},
                        "created_at": now,
                    }
                    connection.execute(
                        """
                        INSERT INTO chat_files(file_id, chat_id, message_id, uri, source_type, metadata_json, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            file_id,
                            chat_id,
                            message_id,
                            source,
                            source_type,
                            json.dumps(payload["metadata"]),
                            now,
                        ),
                    )
                    rows.append(payload)
                connection.commit()
        return rows

    def list_files(self, *, chat_id: str, limit: int = 500) -> list[dict[str, Any]]:
        with self._lock:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT file_id, chat_id, message_id, uri, source_type, metadata_json, created_at
                    FROM chat_files
                    WHERE chat_id = ?
                    ORDER BY created_at ASC
                    LIMIT ?
                    """,
                    (chat_id, max(limit, 1)),
                ).fetchall()
        return [self._row_to_file(row) for row in rows]

    def upsert_run(self, *, chat_id: str, message_id: str, run_id: str, state: str) -> dict[str, Any]:
        now = _utc_now_iso()
        with self._lock:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO chat_runs(run_id, chat_id, message_id, state, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        chat_id=excluded.chat_id,
                        message_id=excluded.message_id,
                        state=excluded.state,
                        updated_at=excluded.updated_at
                    """,
                    (run_id, chat_id, message_id, state, now, now),
                )
                connection.commit()
        return {"run_id": run_id, "chat_id": chat_id, "message_id": message_id, "state": state, "updated_at": now}

    def has_run(self, *, chat_id: str, run_id: str) -> bool:
        with self._lock:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT run_id FROM chat_runs WHERE chat_id = ? AND run_id = ? LIMIT 1",
                    (chat_id, run_id),
                ).fetchone()
        return row is not None

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "chat_id": str(row["chat_id"]),
            "title": str(row["title"]),
            "archived": bool(int(row["archived"])),
            "defaults": json.loads(str(row["defaults_json"] or "{}")),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "message_id": str(row["message_id"]),
            "chat_id": str(row["chat_id"]),
            "role": str(row["role"]),
            "content": str(row["content"]),
            "mode": str(row["mode"] or ""),
            "run_id": str(row["run_id"] or ""),
            "metadata": json.loads(str(row["metadata_json"] or "{}")),
            "created_at": str(row["created_at"]),
        }

    @staticmethod
    def _row_to_file(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "file_id": str(row["file_id"]),
            "chat_id": str(row["chat_id"]),
            "message_id": str(row["message_id"]),
            "uri": str(row["uri"]),
            "source_type": str(row["source_type"]),
            "metadata": json.loads(str(row["metadata_json"] or "{}")),
            "created_at": str(row["created_at"]),
        }
