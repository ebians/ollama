"""ユーザー管理モジュール（SQLite + bcrypt）"""
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import bcrypt

from app.config import CHAT_DB_PATH

_db_path = Path(CHAT_DB_PATH)


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _init_users_db():
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            display_name TEXT NOT NULL,
            is_admin INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS auth_tokens (
            token TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            created_at TEXT NOT NULL
        );
    """)
    conn.close()


_init_users_db()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _check_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def create_user(username: str, password: str, display_name: str = "", is_admin: bool = False) -> str | None:
    """ユーザーを作成する。成功時はuser_id、ユーザー名重複時はNone"""
    user_id = uuid.uuid4().hex[:12]
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO users (id, username, password_hash, display_name, is_admin, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, username, _hash_password(password), display_name or username, int(is_admin), _now()),
        )
        conn.commit()
        return user_id
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()


def authenticate(username: str, password: str) -> dict | None:
    """認証。成功時はユーザー情報dict、失敗時はNone"""
    conn = _get_conn()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    if row and _check_password(password, row["password_hash"]):
        return {"id": row["id"], "username": row["username"], "display_name": row["display_name"], "is_admin": bool(row["is_admin"])}
    return None


def create_token(user_id: str) -> str:
    """認証トークンを発行する"""
    token = uuid.uuid4().hex
    conn = _get_conn()
    conn.execute("INSERT INTO auth_tokens (token, user_id, created_at) VALUES (?, ?, ?)", (token, user_id, _now()))
    conn.commit()
    conn.close()
    return token


def get_user_by_token(token: str) -> dict | None:
    """トークンからユーザー情報を取得する"""
    conn = _get_conn()
    row = conn.execute(
        "SELECT u.id, u.username, u.display_name, u.is_admin FROM users u JOIN auth_tokens t ON u.id = t.user_id WHERE t.token = ?",
        (token,),
    ).fetchone()
    conn.close()
    if row:
        return {"id": row["id"], "username": row["username"], "display_name": row["display_name"], "is_admin": bool(row["is_admin"])}
    return None


def delete_token(token: str):
    """トークンを無効化（ログアウト）"""
    conn = _get_conn()
    conn.execute("DELETE FROM auth_tokens WHERE token = ?", (token,))
    conn.commit()
    conn.close()


def list_users() -> list[dict]:
    """ユーザー一覧を返す"""
    conn = _get_conn()
    rows = conn.execute("SELECT id, username, display_name, is_admin, created_at FROM users ORDER BY created_at").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_user(user_id: str):
    """ユーザーを削除する"""
    conn = _get_conn()
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()


def change_password(user_id: str, old_password: str, new_password: str) -> bool:
    """パスワードを変更する。旧パスワード検証付き"""
    conn = _get_conn()
    row = conn.execute("SELECT password_hash FROM users WHERE id = ?", (user_id,)).fetchone()
    if not row or not _check_password(old_password, row["password_hash"]):
        conn.close()
        return False
    conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (_hash_password(new_password), user_id))
    conn.commit()
    conn.close()
    return True


def reset_password(user_id: str, new_password: str):
    """管理者がパスワードをリセットする（旧パスワード不要）"""
    conn = _get_conn()
    conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (_hash_password(new_password), user_id))
    conn.commit()
    conn.close()


def ensure_admin_exists():
    """管理者ユーザーが存在しない場合はデフォルトで作成する"""
    conn = _get_conn()
    row = conn.execute("SELECT id FROM users WHERE is_admin = 1 LIMIT 1").fetchone()
    conn.close()
    if not row:
        from app.config import ADMIN_PASSWORD
        create_user("admin", ADMIN_PASSWORD, "管理者", is_admin=True)


ensure_admin_exists()
