"""利用ログ: 質問・回答履歴を管理者が閲覧できるようにする"""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.config import CHAT_DB_PATH

_db_path = Path(CHAT_DB_PATH)


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_log_table():
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS usage_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            username TEXT NOT NULL DEFAULT '',
            question TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT 'rag',
            model TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_usage_logs_created ON usage_logs(created_at);
    """)
    conn.close()


_init_log_table()


def add_usage_log(user_id: str, username: str, question: str, mode: str, model: str):
    """利用ログを記録する"""
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_conn()
    conn.execute(
        "INSERT INTO usage_logs (user_id, username, question, mode, model, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, username, question, mode, model, now),
    )
    conn.commit()
    conn.close()


def get_usage_logs(limit: int = 100, offset: int = 0) -> list[dict]:
    """利用ログを新しい順に返す"""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, user_id, username, question, mode, model, created_at FROM usage_logs ORDER BY id DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_usage_stats() -> dict:
    """利用統計を返す"""
    conn = _get_conn()
    total = conn.execute("SELECT COUNT(*) as cnt FROM usage_logs").fetchone()["cnt"]
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_count = conn.execute(
        "SELECT COUNT(*) as cnt FROM usage_logs WHERE created_at >= ?", (today,)
    ).fetchone()["cnt"]
    conn.close()
    return {"total_questions": total, "today_questions": today_count}
