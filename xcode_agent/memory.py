"""Small SQLite FTS5 memory store with transactional index maintenance."""
from datetime import datetime, timezone
from pathlib import Path
import re
import sqlite3
from threading import RLock
from uuid import uuid4

from pydantic import BaseModel, Field


class MemoryItem(BaseModel):
    id: str
    namespace: str
    content: str
    created_at: datetime
    updated_at: datetime
    score: float | None = Field(default=None, description="FTS5 BM25 score; lower ranks first")


class MemoryStoreError(RuntimeError):
    """The memory database is unavailable or lacks FTS5 support."""


class MemoryStore:
    MAX_CONTENT_CHARS = 100_000
    MAX_QUERY_CHARS = 1_000
    MAX_TOP_K = 100

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._lock = RLock()
        self._closed = False
        try:
            self._conn = sqlite3.connect(self.path, timeout=5.0, check_same_thread=False)
        except sqlite3.Error as exc:
            raise MemoryStoreError("Could not open SQLite memory store") from exc
        self._conn.row_factory = sqlite3.Row
        try:
            with self._lock, self._conn:
                self._conn.executescript("""
                    CREATE TABLE IF NOT EXISTS memories (
                        rowid INTEGER PRIMARY KEY,
                        memory_id TEXT NOT NULL,
                        namespace TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(namespace, memory_id)
                    );
                    CREATE INDEX IF NOT EXISTS memories_namespace_idx ON memories(namespace);
                    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                        content, content='memories', content_rowid='rowid', tokenize='unicode61'
                    );
                    CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                        INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
                    END;
                    CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                        INSERT INTO memories_fts(memories_fts, rowid, content)
                        VALUES ('delete', old.rowid, old.content);
                    END;
                    CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF content ON memories BEGIN
                        INSERT INTO memories_fts(memories_fts, rowid, content)
                        VALUES ('delete', old.rowid, old.content);
                        INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
                    END;
                """)
        except sqlite3.Error as exc:
            self._conn.close()
            raise MemoryStoreError("Could not initialize SQLite FTS5 memory store") from exc

    @staticmethod
    def _namespace(value: str) -> str:
        if not isinstance(value, str) or not value.strip() or len(value) > 128:
            raise ValueError("namespace must be a nonempty string of at most 128 characters")
        return value

    @staticmethod
    def _item_id(value: str) -> str:
        if not isinstance(value, str) or not value.strip() or len(value) > 128:
            raise ValueError("item_id must be a nonempty string of at most 128 characters")
        return value

    def _ensure_open(self) -> None:
        if self._closed:
            raise MemoryStoreError("Memory store is closed")

    @staticmethod
    def _item(row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(id=row["memory_id"], namespace=row["namespace"], content=row["content"],
                          created_at=datetime.fromisoformat(row["created_at"]),
                          updated_at=datetime.fromisoformat(row["updated_at"]),
                          score=row["score"] if "score" in row.keys() else None)

    def save(self, namespace: str, content: str, *, item_id: str | None = None) -> MemoryItem:
        """Insert or update one item within a namespace; trigger updates FTS atomically."""
        namespace = self._namespace(namespace)
        if not isinstance(content, str) or not content.strip() or len(content) > self.MAX_CONTENT_CHARS:
            raise ValueError("content must be nonempty and within the size limit")
        item_id = self._item_id(item_id if item_id is not None else uuid4().hex)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._ensure_open()
            with self._conn:
                self._conn.execute("""
                    INSERT INTO memories(memory_id, namespace, content, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(namespace, memory_id) DO UPDATE SET
                        content=excluded.content, updated_at=excluded.updated_at
                """, (item_id, namespace, content, now, now))
                row = self._conn.execute("""
                    SELECT memory_id, namespace, content, created_at, updated_at
                    FROM memories WHERE namespace=? AND memory_id=?
                """, (namespace, item_id)).fetchone()
        return self._item(row)

    def search(self, namespace: str, query: str, top_k: int = 5) -> list[MemoryItem]:
        """Literal token OR search, ordered by SQLite FTS5 BM25 (ascending)."""
        namespace = self._namespace(namespace)
        if not isinstance(query, str) or len(query) > self.MAX_QUERY_CHARS:
            raise ValueError("query must be a string within the size limit")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= self.MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {self.MAX_TOP_K}")
        terms = list(dict.fromkeys(re.findall(r"[^\W_]+", query, flags=re.UNICODE)))[:16]
        expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
        with self._lock:
            self._ensure_open()
            if not terms:
                return []
            rows = self._conn.execute("""
                SELECT m.memory_id, m.namespace, m.content, m.created_at, m.updated_at,
                       bm25(memories_fts) AS score
                FROM memories_fts JOIN memories AS m ON m.rowid = memories_fts.rowid
                WHERE memories_fts MATCH ? AND m.namespace = ?
                ORDER BY score ASC, m.rowid ASC LIMIT ?
            """, (expression, namespace, top_k)).fetchall()
        return [self._item(row) for row in rows]

    def delete(self, namespace: str, item_id: str) -> bool:
        """Delete one namespaced item; trigger removes its FTS posting atomically."""
        namespace, item_id = self._namespace(namespace), self._item_id(item_id)
        with self._lock:
            self._ensure_open()
            with self._conn:
                cursor = self._conn.execute("DELETE FROM memories WHERE namespace=? AND memory_id=?",
                                            (namespace, item_id))
                return cursor.rowcount > 0

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._conn.close()
                self._closed = True

    def __enter__(self) -> "MemoryStore":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
