"""PDFページ画像をSQLiteに保存・取得するモジュール"""

import sqlite3
from app.config import CHAT_DB_PATH

_DB_PATH = CHAT_DB_PATH


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS page_images ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  source TEXT NOT NULL,"
        "  page INTEGER NOT NULL,"
        "  image_b64 TEXT NOT NULL,"
        "  width INTEGER DEFAULT 0,"
        "  height INTEGER DEFAULT 0,"
        "  UNIQUE(source, page)"
        ")"
    )
    return conn


def save_page_image(source: str, page: int, image_b64: str, width: int = 0, height: int = 0):
    """ページ画像をBase64で保存する（既存は上書き）"""
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO page_images (source, page, image_b64, width, height) "
        "VALUES (?, ?, ?, ?, ?)",
        (source, page, image_b64, width, height),
    )
    conn.commit()
    conn.close()


def get_page_images(source: str, pages: list[int]) -> list[dict]:
    """指定ページの画像を取得する"""
    if not pages:
        return []
    conn = _get_conn()
    placeholders = ",".join("?" for _ in pages)
    rows = conn.execute(
        f"SELECT page, image_b64, width, height FROM page_images "
        f"WHERE source = ? AND page IN ({placeholders}) ORDER BY page",
        [source] + pages,
    ).fetchall()
    conn.close()
    return [{"page": r[0], "image_b64": r[1], "width": r[2], "height": r[3]} for r in rows]


def get_all_page_images(source: str) -> list[dict]:
    """指定ドキュメントの全ページ画像を取得する"""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT page, image_b64, width, height FROM page_images "
        "WHERE source = ? ORDER BY page",
        (source,),
    ).fetchall()
    conn.close()
    return [{"page": r[0], "image_b64": r[1], "width": r[2], "height": r[3]} for r in rows]


def delete_page_images(source: str) -> int:
    """指定ドキュメントの全ページ画像を削除する"""
    conn = _get_conn()
    cursor = conn.execute("DELETE FROM page_images WHERE source = ?", (source,))
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted


def count_page_images(source: str) -> int:
    """指定ドキュメントのページ画像数を返す"""
    conn = _get_conn()
    row = conn.execute(
        "SELECT COUNT(*) FROM page_images WHERE source = ?", (source,)
    ).fetchone()
    conn.close()
    return row[0] if row else 0
