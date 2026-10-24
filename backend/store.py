"""SQLite storage for documents, chunks (with their embedding vectors) and asked questions."""
from __future__ import annotations

import json
import math
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    filename TEXT NOT NULL,
    category TEXT NOT NULL,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    chunks INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'ready',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    is_seed INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding TEXT,
    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
"""

# Columns added after the first version. Existing databases are upgraded in place.
MIGRATIONS = [
    ("chunks", "embedding_model", "TEXT"),
    ("questions", "client_id", "TEXT"),
    ("questions", "confidence", "REAL"),
    ("questions", "mode", "TEXT"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._vector_cache: dict[int, tuple[str, list[float], float]] = {}
        self._cache_lock = threading.Lock()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init(self, seeds: list[dict[str, str]], chunker) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            for table, column, kind in MIGRATIONS:
                existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
                if column not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_questions_client ON questions(client_id)")
            # Remove chunks orphaned by deletes made before foreign keys were switched on.
            conn.execute("DELETE FROM chunks WHERE document_id NOT IN (SELECT id FROM documents)")
            if conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0:
                for seed in seeds:
                    self._insert_document(
                        conn,
                        name=seed["name"],
                        filename=seed["filename"],
                        category=seed["category"],
                        size_bytes=len(seed["content"].encode("utf-8")),
                        chunks=chunker(seed["content"]),
                        is_seed=True,
                    )

    # ---------- documents ----------
    def _insert_document(self, conn, *, name, filename, category, size_bytes, chunks, is_seed=False, vectors=None, model=None) -> str:
        document_id = uuid.uuid4().hex
        now = utc_now()
        conn.execute(
            "INSERT INTO documents (id, name, filename, category, size_bytes, chunks, status, created_at, updated_at, is_seed) "
            "VALUES (?, ?, ?, ?, ?, ?, 'ready', ?, ?, ?)",
            (document_id, name, filename, category, size_bytes, len(chunks), now, now, int(is_seed)),
        )
        conn.executemany(
            "INSERT INTO chunks (document_id, chunk_index, content, embedding, embedding_model) VALUES (?, ?, ?, ?, ?)",
            [
                (document_id, i, text, json.dumps(vectors[i]) if vectors else None, model if vectors else None)
                for i, text in enumerate(chunks)
            ],
        )
        return document_id

    def add_document(self, **kwargs) -> dict[str, Any]:
        with self.connect() as conn:
            document_id = self._insert_document(conn, **kwargs)
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        return serialize_document(row)

    def list_documents(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY updated_at DESC, rowid DESC").fetchall()
        return [serialize_document(row) for row in rows]

    def get_document(self, document_id: str) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()

    def delete_document(self, document_id: str) -> bool:
        with self.connect() as conn:
            chunk_ids = [r[0] for r in conn.execute("SELECT id FROM chunks WHERE document_id = ?", (document_id,))]
            deleted = conn.execute("DELETE FROM documents WHERE id = ?", (document_id,)).rowcount > 0
        with self._cache_lock:
            for chunk_id in chunk_ids:
                self._vector_cache.pop(chunk_id, None)
        return deleted

    def prune_uploads(self, keep: int) -> int:
        """Keep at most `keep` documents by removing the oldest non-seed uploads."""
        with self.connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            extra = total - keep
            if extra <= 0:
                return 0
            ids = [r[0] for r in conn.execute(
                "SELECT id FROM documents WHERE is_seed = 0 ORDER BY created_at ASC, rowid ASC LIMIT ?", (extra,)
            )]
        for document_id in ids:
            self.delete_document(document_id)
        return len(ids)

    # ---------- chunks & vectors ----------
    def all_chunks(self) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT chunks.id, chunks.document_id, chunks.chunk_index, chunks.content, chunks.embedding, "
                "chunks.embedding_model, documents.name AS document_name, documents.filename AS filename "
                "FROM chunks JOIN documents ON documents.id = chunks.document_id ORDER BY chunks.id"
            ).fetchall()

    def save_vectors(self, items: list[tuple[int, list[float]]], model: str) -> None:
        with self.connect() as conn:
            conn.executemany(
                "UPDATE chunks SET embedding = ?, embedding_model = ? WHERE id = ?",
                [(json.dumps(vector), model, chunk_id) for chunk_id, vector in items],
            )
        with self._cache_lock:
            for chunk_id, vector in items:
                self._vector_cache[chunk_id] = (model, vector, _norm(vector))

    def vector_for(self, row: sqlite3.Row, model: str) -> tuple[list[float], float] | None:
        """The chunk's vector for `model`, or None if it was never embedded with that model."""
        with self._cache_lock:
            cached = self._vector_cache.get(row["id"])
        if cached and cached[0] == model:
            return cached[1], cached[2]
        # Vectors stored before embedding_model was tracked have NULL model; they can't be trusted.
        if not row["embedding"] or row["embedding_model"] != model:
            return None
        vector = json.loads(row["embedding"])
        entry = (model, vector, _norm(vector))
        with self._cache_lock:
            self._vector_cache[row["id"]] = entry
        return vector, entry[2]

    # ---------- questions ----------
    def add_question(self, question: str, answer: str, confidence: float, mode: str, client_id: str | None) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO questions (question, answer, created_at, client_id, confidence, mode) VALUES (?, ?, ?, ?, ?, ?)",
                (question, answer, utc_now(), client_id, confidence, mode),
            )

    def history(self, client_id: str | None, include_unowned: bool, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as conn:
            if client_id:
                extra = " OR client_id IS NULL" if include_unowned else ""
                rows = conn.execute(
                    f"SELECT id, question, answer, created_at FROM questions WHERE client_id = ?{extra} ORDER BY id DESC LIMIT ?",
                    (client_id, limit),
                ).fetchall()
            elif include_unowned:
                rows = conn.execute(
                    "SELECT id, question, answer, created_at FROM questions ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = []
        return [dict(row) for row in rows]

    def stats(self) -> dict[str, Any]:
        with self.connect() as conn:
            documents = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            questions = conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
            recent = [r[0] for r in conn.execute(
                "SELECT confidence FROM questions WHERE confidence IS NOT NULL ORDER BY id DESC LIMIT 200"
            )]
            members = conn.execute(
                "SELECT COUNT(DISTINCT client_id) FROM questions WHERE client_id IS NOT NULL AND created_at >= datetime('now', '-30 days')"
            ).fetchone()[0]
        hit_rate = round(100 * sum(1 for c in recent if c >= 0.5) / len(recent), 1) if recent else None
        avg = round(sum(recent) / len(recent), 3) if recent else None
        return {
            "documents": documents,
            "chunks": chunks,
            "questions": questions,
            # Share of recent questions whose best sources scored at least 0.5.
            "retrieval_hit_rate": hit_rate,
            "average_confidence": avg,
            "active_members": members or None,
        }


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vector))


def serialize_document(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "filename": row["filename"],
        "category": row["category"],
        "size_bytes": row["size_bytes"],
        "chunks": row["chunks"],
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "is_seed": bool(row["is_seed"]),
    }
