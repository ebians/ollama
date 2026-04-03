"""chat_store.py のテスト（SQLite、Ollama不要）"""
import os
import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    """テスト用の一時DBを使う"""
    db_path = str(tmp_path / "test_chat.db")
    monkeypatch.setenv("CHAT_DB_PATH", db_path)
    # chat_store を再読み込みして新しいDB pathを使う
    import importlib
    import app.config
    importlib.reload(app.config)
    import app.chat_store as cs
    importlib.reload(cs)
    yield cs


class TestCreateSession:
    def test_creates_session(self, tmp_db):
        cs = tmp_db
        sid = cs.create_session("テストセッション", "rag")
        assert isinstance(sid, str)
        assert len(sid) == 12

    def test_default_title(self, tmp_db):
        cs = tmp_db
        sid = cs.create_session("", "free")
        sessions = cs.get_sessions()
        assert sessions[0]["title"] == "新しいチャット"

    def test_mode_stored(self, tmp_db):
        cs = tmp_db
        sid = cs.create_session("test", "hybrid")
        sessions = cs.get_sessions()
        assert sessions[0]["mode"] == "hybrid"


class TestAddMessage:
    def test_add_user_message(self, tmp_db):
        cs = tmp_db
        sid = cs.create_session("test", "rag")
        cs.add_message(sid, "user", "質問です")
        msgs = cs.get_messages(sid)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "質問です"

    def test_add_assistant_message_with_sources(self, tmp_db):
        cs = tmp_db
        sid = cs.create_session("test", "rag")
        sources = [{"text": "chunk", "source": "doc.pdf", "distance": 0.1}]
        cs.add_message(sid, "assistant", "回答です", sources)
        msgs = cs.get_messages(sid)
        assert msgs[0]["sources"][0]["source"] == "doc.pdf"

    def test_title_auto_update(self, tmp_db):
        cs = tmp_db
        sid = cs.create_session("initial", "rag")
        cs.add_message(sid, "user", "最初の質問を自動タイトルにする")
        sessions = cs.get_sessions()
        assert sessions[0]["title"] == "最初の質問を自動タイトルにする"

    def test_title_truncated_at_50(self, tmp_db):
        cs = tmp_db
        sid = cs.create_session("", "rag")
        long_q = "あ" * 100
        cs.add_message(sid, "user", long_q)
        sessions = cs.get_sessions()
        assert len(sessions[0]["title"]) == 50


class TestGetSessions:
    def test_empty(self, tmp_db):
        cs = tmp_db
        assert cs.get_sessions() == []

    def test_order_by_updated(self, tmp_db):
        cs = tmp_db
        s1 = cs.create_session("old", "rag")
        s2 = cs.create_session("new", "rag")
        # s1 に追加して updated_at を更新
        cs.add_message(s1, "user", "update")
        sessions = cs.get_sessions()
        assert sessions[0]["id"] == s1  # 最後に更新されたのが先頭

    def test_limit(self, tmp_db):
        cs = tmp_db
        for i in range(5):
            cs.create_session(f"session{i}", "rag")
        sessions = cs.get_sessions(limit=3)
        assert len(sessions) == 3


class TestDeleteSession:
    def test_delete(self, tmp_db):
        cs = tmp_db
        sid = cs.create_session("to_delete", "rag")
        cs.add_message(sid, "user", "msg")
        cs.delete_session(sid)
        assert cs.get_sessions() == []
        assert cs.get_messages(sid) == []

    def test_delete_nonexistent(self, tmp_db):
        cs = tmp_db
        cs.delete_session("nonexistent")  # エラーにならないこと
