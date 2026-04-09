"""FastAPI エンドポイントのテスト（Ollamaへの通信はモック）"""
import json
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app, get_current_user, require_user, require_admin_user


_ADMIN_USER = {"id": "admin001", "username": "admin", "display_name": "管理者", "is_admin": True}
_NORMAL_USER = {"id": "user001", "username": "testuser", "display_name": "テスト", "is_admin": False}


async def _mock_stream(text):
    """テスト用: chat_completion_stream のモック"""
    yield text


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    # テスト後に依存関係オーバーライドをリセット
    app.dependency_overrides.clear()


@pytest.fixture
def as_admin():
    """管理者としてリクエスト"""
    app.dependency_overrides[get_current_user] = lambda: _ADMIN_USER
    app.dependency_overrides[require_user] = lambda: _ADMIN_USER
    app.dependency_overrides[require_admin_user] = lambda: _ADMIN_USER


@pytest.fixture
def as_user():
    """一般ユーザーとしてリクエスト"""
    app.dependency_overrides[get_current_user] = lambda: _NORMAL_USER
    app.dependency_overrides[require_user] = lambda: _NORMAL_USER


@pytest.fixture
def as_anonymous():
    """未ログイン"""
    app.dependency_overrides[get_current_user] = lambda: None


class TestStats:
    async def test_stats(self, client):
        r = await client.get("/api/stats")
        assert r.status_code == 200
        data = r.json()
        assert "total_chunks" in data


class TestAdminAuth:
    async def test_upload_no_auth(self, client):
        r = await client.post("/api/upload", files={"file": ("test.txt", b"hello")})
        assert r.status_code == 401

    async def test_upload_as_normal_user(self, client, as_user):
        r = await client.post(
            "/api/upload",
            files={"file": ("test.txt", b"hello")},
        )
        assert r.status_code == 403

    async def test_documents_no_auth(self, client):
        r = await client.get("/api/documents")
        assert r.status_code == 401

    async def test_reset_no_auth(self, client):
        r = await client.post("/api/reset")
        assert r.status_code == 401


class TestUpload:
    @patch("app.main.add_document", new_callable=AsyncMock, return_value=3)
    async def test_upload_txt(self, mock_add, client, as_admin):
        r = await client.post(
            "/api/upload",
            files={"file": ("test.txt", b"hello world")},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["filename"] == "test.txt"
        assert data["chunks"] == 3

    async def test_upload_unsupported_ext(self, client, as_admin):
        r = await client.post(
            "/api/upload",
            files={"file": ("test.exe", b"binary content")},
        )
        data = r.json()
        assert "error" in data or "未対応" in str(data)


class TestAsk:
    @patch("app.main.search", new_callable=AsyncMock, return_value=[])
    async def test_ask_rag_no_docs(self, mock_search, client):
        r = await client.post("/api/ask", json={"question": "test?", "mode": "rag"})
        assert r.status_code == 200
        data = r.json()
        assert "ドキュメントが登録されていません" in data["answer"]

    @patch("app.main.chat_completion", new_callable=AsyncMock, return_value="フリー回答です")
    async def test_ask_free(self, mock_chat, client):
        r = await client.post("/api/ask", json={"question": "hello?", "mode": "free"})
        assert r.status_code == 200
        data = r.json()
        assert data["answer"] == "フリー回答です"
        assert data["sources"] == []

    @patch("app.main.chat_completion", new_callable=AsyncMock, return_value="段階的に分析した回答です")
    @patch("app.main.search", new_callable=AsyncMock, return_value=[
        {"text": "売上は前年比10%増", "source": "report.pdf", "distance": 0.12}
    ])
    async def test_ask_stepwise(self, mock_search, mock_chat, client):
        r = await client.post("/api/ask", json={"question": "業績を分析して", "mode": "stepwise"})
        assert r.status_code == 200
        data = r.json()
        assert data["answer"] == "段階的に分析した回答です"
        assert len(data["sources"]) == 1


class TestSessions:
    async def test_list_sessions(self, client, as_anonymous):
        r = await client.get("/api/sessions")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    async def test_get_nonexistent_session(self, client, as_anonymous):
        r = await client.get("/api/sessions/nonexistent")
        assert r.status_code == 404

    async def test_delete_session(self, client, as_anonymous):
        r = await client.delete("/api/sessions/nonexistent")
        assert r.status_code == 200  # 存在しなくてもOK


class TestAuthAPI:
    @patch("app.main.authenticate", return_value=_NORMAL_USER)
    @patch("app.main.create_token", return_value="test_token_123")
    async def test_login_success(self, mock_token, mock_auth, client):
        r = await client.post("/api/auth/login", json={"username": "testuser", "password": "pass"})
        assert r.status_code == 200
        assert r.json()["user"]["username"] == "testuser"
        assert "auth_token" in r.cookies

    @patch("app.main.authenticate", return_value=None)
    async def test_login_failure(self, mock_auth, client):
        r = await client.post("/api/auth/login", json={"username": "bad", "password": "bad"})
        assert r.status_code == 401

    async def test_me_logged_in(self, client, as_user):
        r = await client.get("/api/auth/me")
        assert r.status_code == 200
        assert r.json()["user"]["username"] == "testuser"

    async def test_me_not_logged_in(self, client, as_anonymous):
        r = await client.get("/api/auth/me")
        assert r.status_code == 200
        assert r.json()["user"] is None

    async def test_logout(self, client):
        r = await client.post("/api/auth/logout")
        assert r.status_code == 200


class TestUserManagement:
    async def test_list_users_as_admin(self, client, as_admin):
        with patch("app.main.list_users", return_value=[]):
            r = await client.get("/api/users")
            assert r.status_code == 200

    async def test_list_users_as_normal_user(self, client, as_user):
        r = await client.get("/api/users")
        assert r.status_code == 403

    async def test_create_user_as_admin(self, client, as_admin):
        with patch("app.main.create_user", return_value="new001"):
            r = await client.post("/api/users", json={"username": "new", "password": "pass"})
            assert r.status_code == 200
            assert r.json()["user_id"] == "new001"

    async def test_delete_user_as_admin(self, client, as_admin):
        with patch("app.main.delete_user"):
            r = await client.delete("/api/users/user001")
            assert r.status_code == 200


class TestModels:
    @patch("app.main.list_models", new_callable=AsyncMock, return_value=[
        {"name": "qwen2.5:7b", "size": 5000000000, "modified_at": "2025-01-01"},
        {"name": "llama3:8b", "size": 4500000000, "modified_at": "2025-01-02"},
    ])
    async def test_list_models(self, mock_models, client):
        r = await client.get("/api/models")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 2
        assert data[0]["name"] == "qwen2.5:7b"

    @patch("app.main.list_models", new_callable=AsyncMock, side_effect=Exception("connection error"))
    async def test_list_models_error(self, mock_models, client):
        r = await client.get("/api/models")
        assert r.status_code == 200
        assert r.json() == []

    async def test_stats_includes_default_model(self, client):
        r = await client.get("/api/stats")
        assert r.status_code == 200
        data = r.json()
        assert "default_model" in data


class TestChunkPreview:
    @patch("app.main.get_document_chunks", return_value=[
        {"id": "abc_chunk0", "chunk_index": 0, "text": "テスト内容", "length": 5},
    ])
    async def test_get_chunks(self, mock_chunks, client, as_admin):
        r = await client.get("/api/documents/test.pdf/chunks")
        assert r.status_code == 200
        data = r.json()
        assert data["source"] == "test.pdf"
        assert len(data["chunks"]) == 1
        assert data["chunks"][0]["text"] == "テスト内容"

    @patch("app.main.get_document_chunks", return_value=[])
    async def test_get_chunks_not_found(self, mock_chunks, client, as_admin):
        r = await client.get("/api/documents/notexist.pdf/chunks")
        assert r.status_code == 404

    async def test_get_chunks_no_auth(self, client):
        r = await client.get("/api/documents/test.pdf/chunks")
        assert r.status_code == 401

    async def test_get_chunks_non_admin(self, client, as_user):
        r = await client.get("/api/documents/test.pdf/chunks")
        assert r.status_code == 403


class TestPasswordChange:
    @patch("app.main.change_password", return_value=True)
    async def test_change_password_success(self, mock_change, client, as_user):
        r = await client.post("/api/auth/password", json={"old_password": "old", "new_password": "newpass"})
        assert r.status_code == 200
        mock_change.assert_called_once_with("user001", "old", "newpass")

    @patch("app.main.change_password", return_value=False)
    async def test_change_password_wrong_old(self, mock_change, client, as_user):
        r = await client.post("/api/auth/password", json={"old_password": "wrong", "new_password": "newpass"})
        assert r.status_code == 400

    async def test_change_password_too_short(self, client, as_user):
        r = await client.post("/api/auth/password", json={"old_password": "old", "new_password": "ab"})
        assert r.status_code == 400

    async def test_change_password_no_auth(self, client):
        r = await client.post("/api/auth/password", json={"old_password": "old", "new_password": "newpass"})
        assert r.status_code == 401


class TestPasswordReset:
    @patch("app.main.reset_password")
    async def test_reset_password_as_admin(self, mock_reset, client, as_admin):
        r = await client.post("/api/users/user001/reset-password", json={"new_password": "newpass"})
        assert r.status_code == 200
        mock_reset.assert_called_once_with("user001", "newpass")

    async def test_reset_password_as_user(self, client, as_user):
        r = await client.post("/api/users/user001/reset-password", json={"new_password": "newpass"})
        assert r.status_code == 403

    async def test_reset_password_too_short(self, client, as_admin):
        r = await client.post("/api/users/user001/reset-password", json={"new_password": "ab"})
        assert r.status_code == 400


class TestStaticPages:
    async def test_index(self, client):
        r = await client.get("/")
        assert r.status_code == 200
        assert "RAG" in r.text

    async def test_admin(self, client):
        r = await client.get("/admin")
        assert r.status_code == 200
        assert "管理" in r.text


class TestRateLimit:
    async def test_rate_limit_429(self, client, as_user):
        """レート制限超過で429"""
        from app.rate_limit import reset_rate_limit, _requests
        from app.config import RATE_LIMIT_PER_MINUTE
        import time
        reset_rate_limit()
        # 上限まで埋める
        _requests["user001"] = [time.time()] * RATE_LIMIT_PER_MINUTE
        with patch("app.main.search", new_callable=AsyncMock, return_value=[]):
            r = await client.post("/api/ask/stream", json={"question": "test", "mode": "free"})
        assert r.status_code == 429
        reset_rate_limit()

    async def test_rate_limit_ok(self, client, as_user):
        """レート制限内なら通過"""
        from app.rate_limit import reset_rate_limit
        reset_rate_limit()
        with patch("app.main.chat_completion_stream", return_value=_mock_stream("ok")):
            r = await client.post("/api/ask/stream", json={"question": "test", "mode": "free"})
        assert r.status_code == 200
        reset_rate_limit()


class TestUsageLogs:
    async def test_get_logs_as_admin(self, client, as_admin):
        r = await client.get("/api/usage-logs")
        assert r.status_code == 200
        data = r.json()
        assert "logs" in data
        assert "stats" in data

    async def test_get_logs_as_user(self, client, as_user):
        r = await client.get("/api/usage-logs")
        assert r.status_code == 403

    async def test_get_logs_no_auth(self, client):
        r = await client.get("/api/usage-logs")
        assert r.status_code == 401


class TestChatExport:
    async def test_export_session(self, client, as_user):
        from app.chat_store import create_session, add_message
        sid = create_session("テスト", "rag", "user001")
        add_message(sid, "user", "テスト質問")
        add_message(sid, "assistant", "テスト回答")
        r = await client.get(f"/api/sessions/{sid}/export")
        assert r.status_code == 200
        assert "テスト質問" in r.text
        assert "テスト回答" in r.text
        assert "text/markdown" in r.headers["content-type"]

    async def test_export_nonexistent(self, client, as_user):
        r = await client.get("/api/sessions/nonexistent123/export")
        assert r.status_code == 404


class TestMultimodalMode:
    @patch("app.main.chat_completion", new_callable=AsyncMock, return_value="図表から読み取りました")
    @patch("app.main.search", new_callable=AsyncMock, return_value=[
        {"text": "表データ", "source": "report.pdf", "distance": 0.1, "page": 1}
    ])
    async def test_ask_multimodal(self, mock_search, mock_chat, client):
        r = await client.post("/api/ask", json={"question": "図表の内容は？", "mode": "multimodal"})
        assert r.status_code == 200
        data = r.json()
        assert len(data["sources"]) == 1


class TestCalculateMode:
    @patch("app.main.chat_completion", new_callable=AsyncMock, return_value="差額は200万円です")
    @patch("app.main.search", new_callable=AsyncMock, return_value=[
        {"text": "| 部門 | 売上 |\n| --- | --- |\n| A | 1000万 |\n| B | 800万 |", "source": "report.pdf", "distance": 0.1}
    ])
    async def test_ask_calculate(self, mock_search, mock_chat, client):
        r = await client.post("/api/ask", json={"question": "A部門とB部門の売上差は？", "mode": "calculate"})
        assert r.status_code == 200
        data = r.json()
        assert "sources" in data


class TestConsistencyMode:
    @patch("app.main.chat_completion", new_callable=AsyncMock, return_value="不一致があります")
    @patch("app.main.search", new_callable=AsyncMock, return_value=[
        {"text": "納期は30日", "source": "a.pdf", "distance": 0.1},
        {"text": "納期は45日", "source": "b.pdf", "distance": 0.2},
    ])
    async def test_ask_consistency(self, mock_search, mock_chat, client):
        r = await client.post("/api/ask", json={"question": "文書間の整合性をチェック", "mode": "consistency"})
        assert r.status_code == 200
        data = r.json()
        assert len(data["sources"]) == 2

    async def test_consistency_check_api_no_auth(self, client):
        r = await client.post("/api/consistency-check", json={"sources": []})
        assert r.status_code == 401

    async def test_consistency_check_api_as_admin(self, client, as_admin):
        r = await client.post("/api/consistency-check", json={"sources": []})
        assert r.status_code == 200
        data = r.json()
        assert "summary" in data
        assert "issues" in data


class TestAutoMode:
    @patch("app.main.classify_intent", new_callable=AsyncMock, return_value="free")
    @patch("app.main.chat_completion", new_callable=AsyncMock, return_value="自動振り分けフリー回答")
    async def test_auto_routes_to_free(self, mock_chat, mock_classify, client):
        r = await client.post("/api/ask", json={"question": "今日は何曜日？", "mode": "auto"})
        assert r.status_code == 200
        data = r.json()
        assert data["answer"] == "自動振り分けフリー回答"
        assert data["sources"] == []
        mock_classify.assert_called_once()

    @patch("app.main.classify_intent", new_callable=AsyncMock, return_value="rag")
    @patch("app.main.chat_completion", new_callable=AsyncMock, return_value="RAG回答です")
    @patch("app.main.search", new_callable=AsyncMock, return_value=[
        {"text": "有給休暇は年20日", "source": "rules.pdf", "distance": 0.1}
    ])
    async def test_auto_routes_to_rag(self, mock_search, mock_chat, mock_classify, client):
        r = await client.post("/api/ask", json={"question": "有給休暇の日数は？", "mode": "auto"})
        assert r.status_code == 200
        data = r.json()
        assert data["answer"] == "RAG回答です"
        assert len(data["sources"]) == 1

    @patch("app.main.chat_completion", new_callable=AsyncMock, return_value="明示モード回答")
    async def test_explicit_mode_skips_classify(self, mock_chat, client):
        """明示的にモード指定した場合はclassify_intentを呼ばない"""
        with patch("app.main.classify_intent", new_callable=AsyncMock) as mock_classify:
            r = await client.post("/api/ask", json={"question": "hello", "mode": "free"})
            assert r.status_code == 200
            mock_classify.assert_not_called()


class TestUploadMode:
    @patch("app.main.chat_completion", new_callable=AsyncMock, return_value="ドキュメントの内容は...")
    async def test_ask_upload_mode(self, mock_chat, client):
        r = await client.post("/api/ask", json={
            "question": "このドキュメントの要点は？",
            "mode": "upload",
            "temp_context": "これはテスト用のドキュメントです。重要な点が3つあります。",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["answer"] == "ドキュメントの内容は..."
        assert data["sources"] == []
        # contextが渡されていることを確認
        mock_chat.assert_called_once()
        call_args = mock_chat.call_args
        assert call_args[0][1] == "これはテスト用のドキュメントです。重要な点が3つあります。"

    async def test_upload_temp_as_user(self, client, as_user):
        r = await client.post("/api/upload-temp", files={"file": ("test.txt", b"hello world text")})
        assert r.status_code == 200
        data = r.json()
        assert data["filename"] == "test.txt"
        assert data["text"] == "hello world text"
        assert data["length"] == 16

    async def test_upload_temp_unsupported_ext(self, client, as_user):
        r = await client.post("/api/upload-temp", files={"file": ("test.exe", b"binary")})
        assert r.status_code == 400

    async def test_upload_temp_no_auth(self, client):
        r = await client.post("/api/upload-temp", files={"file": ("test.txt", b"hello")})
        assert r.status_code == 401

    async def test_upload_temp_empty_file(self, client, as_user):
        r = await client.post("/api/upload-temp", files={"file": ("test.txt", b"")})
        assert r.status_code == 400


class TestKnowledgeUserAccess:
    """ナレッジ形式知化のユーザー権限テスト"""

    async def test_create_as_user(self, client, as_user):
        """一般ユーザーがワークフロー作成できる"""
        r = await client.post("/api/knowledge", json={"title": "テスト"})
        assert r.status_code == 200
        assert "id" in r.json()

    async def test_create_no_auth(self, client):
        """未認証ではワークフロー作成不可"""
        r = await client.post("/api/knowledge", json={"title": "テスト"})
        assert r.status_code == 401

    async def test_list_as_user(self, client, as_user):
        """一般ユーザーがワークフロー一覧を取得できる"""
        r = await client.get("/api/knowledge")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    async def test_stats_admin_only(self, client, as_user):
        """統計は管理者のみ"""
        r = await client.get("/api/knowledge/stats")
        assert r.status_code == 403

    async def test_stats_as_admin(self, client, as_admin):
        """管理者は統計取得可"""
        r = await client.get("/api/knowledge/stats")
        assert r.status_code == 200

    async def test_get_own_workflow(self, client, as_user):
        """自分のワークフローを取得できる"""
        r = await client.post("/api/knowledge", json={"title": "マイナレッジ"})
        wf_id = r.json()["id"]
        r = await client.get(f"/api/knowledge/{wf_id}")
        assert r.status_code == 200
        assert r.json()["title"] == "マイナレッジ"

    async def test_get_others_workflow_forbidden(self, client, as_user):
        """他人のワークフローは取得不可"""
        from app.knowledge_store import create_workflow
        wf_id = create_workflow("他人のナレッジ", interviewer_id="other_user")
        r = await client.get(f"/api/knowledge/{wf_id}")
        assert r.status_code == 403

    async def test_update_own_interview(self, client, as_user):
        """自分のワークフローのインタビューデータを更新できる"""
        r = await client.post("/api/knowledge", json={"title": "テスト"})
        wf_id = r.json()["id"]
        r = await client.put(f"/api/knowledge/{wf_id}/interview", json={
            "interview_data": [{"q": "質問", "a": "回答"}],
        })
        assert r.status_code == 200

    async def test_update_others_interview_forbidden(self, client, as_user):
        """他人のワークフローのインタビューデータは更新不可"""
        from app.knowledge_store import create_workflow
        wf_id = create_workflow("他人のナレッジ", interviewer_id="other_user")
        r = await client.put(f"/api/knowledge/{wf_id}/interview", json={
            "interview_data": [{"q": "Q", "a": "A"}],
        })
        assert r.status_code == 403

    async def test_delete_own_draft(self, client, as_user):
        """自分のdraftは削除可"""
        r = await client.post("/api/knowledge", json={"title": "削除用"})
        wf_id = r.json()["id"]
        r = await client.delete(f"/api/knowledge/{wf_id}")
        assert r.status_code == 200

    async def test_delete_others_forbidden(self, client, as_user):
        """他人のワークフローは削除不可"""
        from app.knowledge_store import create_workflow
        wf_id = create_workflow("他人", interviewer_id="other_user")
        r = await client.delete(f"/api/knowledge/{wf_id}")
        assert r.status_code == 403

    async def test_review_admin_only(self, client, as_user):
        """レビュー操作は管理者のみ"""
        r = await client.post("/api/knowledge/fake_id/review", json={"action": "approve"})
        assert r.status_code == 403

    async def test_publish_admin_only(self, client, as_user):
        """公開操作は管理者のみ"""
        r = await client.post("/api/knowledge/fake_id/publish")
        assert r.status_code == 403
