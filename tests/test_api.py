"""FastAPI エンドポイントのテスト（Ollamaへの通信はモック）"""
import json
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app, get_current_user, require_user, require_admin_user


_ADMIN_USER = {"id": "admin001", "username": "admin", "display_name": "管理者", "is_admin": True}
_NORMAL_USER = {"id": "user001", "username": "testuser", "display_name": "テスト", "is_admin": False}


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


class TestStaticPages:
    async def test_index(self, client):
        r = await client.get("/")
        assert r.status_code == 200
        assert "RAG" in r.text

    async def test_admin(self, client):
        r = await client.get("/admin")
        assert r.status_code == 200
        assert "管理" in r.text
