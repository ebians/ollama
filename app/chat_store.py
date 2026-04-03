import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config import CHAT_DB_PATH

_db_path = Path(CHAT_DB_PATH)
_db_path.parent.mkdir(parents=True, exist_ok=True)


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_db():
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT 'rag',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
    """)
    conn.close()


_init_db()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_session(title: str = "", mode: str = "rag") -> str:
    """新しいチャットセッションを作成する"""
    session_id = uuid.uuid4().hex[:12]
    now = _now()
    conn = _get_conn()
    conn.execute(
        "INSERT INTO sessions (id, title, mode, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (session_id, title or "新しいチャット", mode, now, now),
    )
    conn.commit()
    conn.close()
    return session_id


def add_message(session_id: str, role: str, content: str, sources: list[dict] | None = None):
    """セッションにメッセージを追加する"""
    now = _now()
    conn = _get_conn()
    conn.execute(
        "INSERT INTO messages (session_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)",
        (session_id, role, content, json.dumps(sources, ensure_ascii=False) if sources else None, now),
    )
    # タイトル更新（最初のユーザーメッセージをタイトルに）
    row = conn.execute(
        "SELECT content FROM messages WHERE session_id = ? AND role = 'user' ORDER BY id LIMIT 1",
        (session_id,),
    ).fetchone()
    if row:
        title = row["content"][:50]
        conn.execute("UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?", (title, now, session_id))
    else:
        conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
    conn.commit()
    conn.close()


def get_sessions(limit: int = 50) -> list[dict]:
    """セッション一覧を返す（新しい順）"""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, title, mode, created_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_messages(session_id: str) -> list[dict]:
    """セッションのメッセージを返す"""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT role, content, sources, created_at FROM messages WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        msg = {"role": r["role"], "content": r["content"], "created_at": r["created_at"]}
        if r["sources"]:
            msg["sources"] = json.loads(r["sources"])
        result.append(msg)
    return result


def delete_session(session_id: str):
    """セッションを削除する"""
    conn = _get_conn()
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()
