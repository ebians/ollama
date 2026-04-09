"""暗黙知の形式知化ワークフロー: DB管理モジュール

ステージ: draft → summary → review → published
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config import CHAT_DB_PATH

_db_path = Path(CHAT_DB_PATH)


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_db():
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS knowledge_workflows (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            stage TEXT NOT NULL DEFAULT 'draft',
            category TEXT NOT NULL DEFAULT '',
            interviewee TEXT NOT NULL DEFAULT '',
            interviewer_id TEXT NOT NULL DEFAULT '',
            interview_data TEXT NOT NULL DEFAULT '[]',
            summary TEXT NOT NULL DEFAULT '',
            faq TEXT NOT NULL DEFAULT '[]',
            review_notes TEXT NOT NULL DEFAULT '',
            reviewer_id TEXT NOT NULL DEFAULT '',
            doc_id TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
    """)
    conn.close()


_init_db()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["interview_data"] = json.loads(d.get("interview_data") or "[]")
    d["faq"] = json.loads(d.get("faq") or "[]")
    return d


# ---------- CRUD ----------

def create_workflow(
    title: str,
    category: str = "",
    interviewee: str = "",
    interviewer_id: str = "",
) -> str:
    """新規ワークフローを作成（draft ステージ）"""
    wf_id = uuid.uuid4().hex[:12]
    now = _now()
    conn = _get_conn()
    conn.execute(
        "INSERT INTO knowledge_workflows "
        "(id, title, stage, category, interviewee, interviewer_id, created_at, updated_at) "
        "VALUES (?, ?, 'draft', ?, ?, ?, ?, ?)",
        (wf_id, title, category, interviewee, interviewer_id, now, now),
    )
    conn.commit()
    conn.close()
    return wf_id


def get_workflow(wf_id: str) -> dict | None:
    """ワークフロー詳細を取得"""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM knowledge_workflows WHERE id = ?", (wf_id,)
    ).fetchone()
    conn.close()
    return _row_to_dict(row) if row else None


def list_workflows(stage: str = "", user_id: str = "") -> list[dict]:
    """ワークフロー一覧（新しい順、stage/user_id指定可）"""
    conn = _get_conn()
    conditions: list[str] = []
    params: list[str] = []
    if stage:
        conditions.append("stage = ?")
        params.append(stage)
    if user_id:
        conditions.append("interviewer_id = ?")
        params.append(user_id)
    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
    rows = conn.execute(
        f"SELECT * FROM knowledge_workflows{where} ORDER BY updated_at DESC",
        params,
    ).fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def update_interview_data(wf_id: str, interview_data: list[dict]) -> bool:
    """インタビューデータ（Q&Aペアのリスト）を保存"""
    conn = _get_conn()
    now = _now()
    cursor = conn.execute(
        "UPDATE knowledge_workflows SET interview_data = ?, updated_at = ? "
        "WHERE id = ? AND stage = 'draft'",
        (json.dumps(interview_data, ensure_ascii=False), now, wf_id),
    )
    conn.commit()
    conn.close()
    return cursor.rowcount > 0


def save_summary(wf_id: str, summary: str, faq: list[dict]) -> bool:
    """LLM生成の要約とFAQを保存し、stageをsummaryに進める"""
    conn = _get_conn()
    now = _now()
    cursor = conn.execute(
        "UPDATE knowledge_workflows SET summary = ?, faq = ?, stage = 'summary', updated_at = ? "
        "WHERE id = ? AND stage IN ('draft', 'summary')",
        (summary, json.dumps(faq, ensure_ascii=False), now, wf_id),
    )
    conn.commit()
    conn.close()
    return cursor.rowcount > 0


def submit_for_review(wf_id: str) -> bool:
    """レビュー依頼に進める"""
    conn = _get_conn()
    now = _now()
    cursor = conn.execute(
        "UPDATE knowledge_workflows SET stage = 'review', updated_at = ? "
        "WHERE id = ? AND stage = 'summary'",
        (now, wf_id),
    )
    conn.commit()
    conn.close()
    return cursor.rowcount > 0


def approve_workflow(wf_id: str, reviewer_id: str, review_notes: str = "") -> bool:
    """レビュー承認 → published"""
    conn = _get_conn()
    now = _now()
    cursor = conn.execute(
        "UPDATE knowledge_workflows SET stage = 'published', reviewer_id = ?, "
        "review_notes = ?, updated_at = ? WHERE id = ? AND stage = 'review'",
        (reviewer_id, review_notes, now, wf_id),
    )
    conn.commit()
    conn.close()
    return cursor.rowcount > 0


def reject_workflow(wf_id: str, reviewer_id: str, review_notes: str = "") -> bool:
    """レビュー差し戻し → summary"""
    conn = _get_conn()
    now = _now()
    cursor = conn.execute(
        "UPDATE knowledge_workflows SET stage = 'summary', reviewer_id = ?, "
        "review_notes = ?, updated_at = ? WHERE id = ? AND stage = 'review'",
        (reviewer_id, review_notes, now, wf_id),
    )
    conn.commit()
    conn.close()
    return cursor.rowcount > 0


def set_doc_id(wf_id: str, doc_id: str) -> bool:
    """RAGに公開した際のdoc_idを記録"""
    conn = _get_conn()
    now = _now()
    cursor = conn.execute(
        "UPDATE knowledge_workflows SET doc_id = ?, updated_at = ? WHERE id = ?",
        (doc_id, now, wf_id),
    )
    conn.commit()
    conn.close()
    return cursor.rowcount > 0


def delete_workflow(wf_id: str) -> bool:
    """ワークフローを削除"""
    conn = _get_conn()
    cursor = conn.execute("DELETE FROM knowledge_workflows WHERE id = ?", (wf_id,))
    conn.commit()
    conn.close()
    return cursor.rowcount > 0


def get_workflow_stats() -> dict:
    """ステージ別の件数統計"""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT stage, COUNT(*) as cnt FROM knowledge_workflows GROUP BY stage"
    ).fetchall()
    conn.close()
    stats = {r["stage"]: r["cnt"] for r in rows}
    stats["total"] = sum(stats.values())
    return stats
