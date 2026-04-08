import json
import secrets
import uuid

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Request, Response, Cookie
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

from app.config import ADMIN_PASSWORD
from app.ollama_client import chat_completion, chat_completion_stream, list_models
from app.rate_limit import check_rate_limit
from app.audit_log import add_usage_log, get_usage_logs, get_usage_stats
from app.parser import extract_text, SUPPORTED_EXTENSIONS
from app.vectorstore import add_document, search, get_stats, reset_db, list_documents, delete_document, get_document_chunks
from app.chat_store import (
    create_session, add_message, get_sessions, get_messages, delete_session,
)
from app.user_store import (
    authenticate, create_token, get_user_by_token, delete_token,
    create_user, list_users, delete_user, change_password, reset_password,
)

app = FastAPI(title="Ollama RAG App")
security = HTTPBasic()


def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    """管理者パスワードを検証する"""
    if not secrets.compare_digest(credentials.password, ADMIN_PASSWORD):
        raise HTTPException(status_code=401, detail="認証エラー", headers={"WWW-Authenticate": "Basic"})
    return credentials.username


def get_current_user(auth_token: str = Cookie(default="")):
    """Cookieトークンからログインユーザーを取得する"""
    if not auth_token:
        return None
    return get_user_by_token(auth_token)


def require_user(auth_token: str = Cookie(default="")):
    """ログイン必須のエンドポイント用"""
    if not auth_token:
        raise HTTPException(status_code=401, detail="ログインが必要です")
    user = get_user_by_token(auth_token)
    if not user:
        raise HTTPException(status_code=401, detail="セッションが無効です")
    return user


def require_admin_user(user: dict = Depends(require_user)):
    """管理者ユーザー必須のエンドポイント用"""
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="管理者権限が必要です")
    return user


class LoginRequest(BaseModel):
    username: str
    password: str


class AskRequest(BaseModel):
    question: str
    history: list[dict] = []
    mode: str = "rag"  # "rag" | "hybrid" | "free" | "stepwise"
    session_id: str = ""  # 空なら新規セッション
    model: str = ""  # 空ならデフォルトモデル


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


# ---------- API ----------

# ---------- 認証 API ----------

@app.post("/api/auth/login")
async def login(req: LoginRequest, response: Response):
    """ログイン → Cookieにトークン設定"""
    user = authenticate(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="ユーザー名またはパスワードが間違っています")
    token = create_token(user["id"])
    response.set_cookie("auth_token", token, httponly=True, samesite="lax", max_age=60*60*24*30)
    return {"user": user}


@app.post("/api/auth/logout")
async def logout(response: Response, auth_token: str = Cookie(default="")):
    """ログアウト → Cookie削除"""
    if auth_token:
        delete_token(auth_token)
    response.delete_cookie("auth_token")
    return {"status": "ok"}


@app.get("/api/auth/me")
async def get_me(user: dict | None = Depends(get_current_user)):
    """現在のログインユーザー情報を返す"""
    if not user:
        return {"user": None}
    return {"user": user}


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


@app.post("/api/auth/password")
async def api_change_password(req: ChangePasswordRequest, user: dict = Depends(require_user)):
    """ログインユーザー自身のパスワードを変更する"""
    if len(req.new_password) < 4:
        raise HTTPException(status_code=400, detail="パスワードは4文字以上です")
    ok = change_password(user["id"], req.old_password, req.new_password)
    if not ok:
        raise HTTPException(status_code=400, detail="現在のパスワードが正しくありません")
    return {"status": "ok"}


# ---------- ユーザー管理 API（管理者のみ）----------

class CreateUserRequest(BaseModel):
    username: str
    password: str
    display_name: str = ""
    is_admin: bool = False


@app.post("/api/users")
async def api_create_user(req: CreateUserRequest, _admin: dict = Depends(require_admin_user)):
    """ユーザーを作成する（管理者のみ）"""
    user_id = create_user(req.username, req.password, req.display_name, req.is_admin)
    if not user_id:
        raise HTTPException(status_code=409, detail="ユーザー名が既に使われています")
    return {"user_id": user_id, "username": req.username}


@app.get("/api/users")
async def api_list_users(_admin: dict = Depends(require_admin_user)):
    """ユーザー一覧を返す（管理者のみ）"""
    return list_users()


@app.delete("/api/users/{user_id}")
async def api_delete_user(user_id: str, _admin: dict = Depends(require_admin_user)):
    """ユーザーを削除する（管理者のみ）"""
    delete_user(user_id)
    return {"status": "ok"}


class ResetPasswordRequest(BaseModel):
    new_password: str


@app.post("/api/users/{user_id}/reset-password")
async def api_reset_password(user_id: str, req: ResetPasswordRequest, _admin: dict = Depends(require_admin_user)):
    """管理者がユーザーのパスワードをリセットする"""
    if len(req.new_password) < 4:
        raise HTTPException(status_code=400, detail="パスワードは4文字以上です")
    reset_password(user_id, req.new_password)
    return {"status": "ok"}


# ---------- 管理者専用 API ----------

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...), _admin: dict = Depends(require_admin_user)):
    """ファイル(.txt/.pdf/.docx/.xlsx)をアップロードし、ベクトルDBに格納する（管理者専用）"""
    filename = file.filename or "unknown"
    ext = ("." + filename.rsplit(".", 1)[-1]).lower() if "." in filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        return {"error": f"未対応の形式です。対応形式: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"}
    content = await file.read()
    text = extract_text(filename, content)
    doc_id = uuid.uuid4().hex[:12]
    num_chunks = await add_document(doc_id, filename, text)
    return {"doc_id": doc_id, "filename": filename, "chunks": num_chunks}


@app.get("/api/documents")
async def get_documents(_admin: dict = Depends(require_admin_user)):
    """登録済みドキュメント一覧を返す（管理者専用）"""
    return list_documents()


@app.delete("/api/documents/{source:path}")
async def remove_document(source: str, _admin: dict = Depends(require_admin_user)):
    """指定ドキュメントを削除する（管理者専用）"""
    deleted = delete_document(source)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="ドキュメントが見つかりません")
    return {"source": source, "deleted_chunks": deleted}


@app.get("/api/documents/{source:path}/chunks")
async def get_chunks(source: str, _admin: dict = Depends(require_admin_user)):
    """指定ドキュメントのチャンク一覧を返す（管理者専用）"""
    chunks = get_document_chunks(source)
    if not chunks:
        raise HTTPException(status_code=404, detail="ドキュメントが見つかりません")
    return {"source": source, "chunks": chunks}


@app.post("/api/reset")
async def reset(_admin: dict = Depends(require_admin_user)):
    """ドキュメントを全削除する（管理者専用）"""
    reset_db()
    return {"status": "ok"}


@app.post("/api/ask", response_model=AskResponse)
async def ask_question(req: AskRequest):
    """質問に対して回答する（モード対応）"""
    hits = []
    context = None
    if req.mode in ("rag", "hybrid", "stepwise"):
        hits = await search(req.question)
        if not hits and req.mode == "rag":
            return AskResponse(answer="ドキュメントが登録されていません。先にファイルをアップロードしてください。", sources=[])
        if hits:
            context = "\n\n---\n\n".join(h["text"] for h in hits)
    answer = await chat_completion(req.question, context, req.history, req.mode, req.model)
    return AskResponse(answer=answer, sources=hits)


@app.post("/api/ask/stream")
async def ask_question_stream(req: AskRequest, user: dict | None = Depends(get_current_user)):
    """ストリーミング回答（SSE、モード対応、履歴保存）"""
    # レート制限
    user_id = user["id"] if user else "anonymous"
    if not check_rate_limit(user_id):
        raise HTTPException(status_code=429, detail="リクエスト回数の上限に達しました。しばらくお待ちください。")

    # 利用ログ記録
    username = user["username"] if user else "anonymous"
    add_usage_log(user_id, username, req.question, req.mode, req.model)

    # セッション管理
    session_id = req.session_id or create_session(req.question[:50], req.mode, user_id)
    add_message(session_id, "user", req.question)

    hits = []
    context = None
    if req.mode in ("rag", "hybrid", "stepwise"):
        hits = await search(req.question)
        if not hits and req.mode == "rag":
            no_doc_msg = "ドキュメントが登録されていません。先にファイルをアップロードしてください。"
            add_message(session_id, "assistant", no_doc_msg)
            async def no_docs():
                data = json.dumps({"type": "session_id", "session_id": session_id}, ensure_ascii=False)
                yield f"data: {data}\n\n"
                data = json.dumps({"type": "sources", "sources": []}, ensure_ascii=False)
                yield f"data: {data}\n\n"
                data = json.dumps({"type": "token", "token": no_doc_msg}, ensure_ascii=False)
                yield f"data: {data}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(no_docs(), media_type="text/event-stream")
        if hits:
            context = "\n\n---\n\n".join(h["text"] for h in hits)

    async def generate():
        # セッションIDを送信
        data = json.dumps({"type": "session_id", "session_id": session_id}, ensure_ascii=False)
        yield f"data: {data}\n\n"
        data = json.dumps({"type": "sources", "sources": hits}, ensure_ascii=False)
        yield f"data: {data}\n\n"
        full_answer = ""
        async for token in chat_completion_stream(req.question, context, req.history, req.mode, req.model):
            full_answer += token
            data = json.dumps({"type": "token", "token": token}, ensure_ascii=False)
            yield f"data: {data}\n\n"
        # 回答をDBに保存
        add_message(session_id, "assistant", full_answer, hits if hits else None)
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/stats")
async def stats():
    """デフォルトモデル名とDB統計を返す"""
    from app.config import CHAT_MODEL
    s = get_stats()
    s["default_model"] = CHAT_MODEL
    return s


@app.get("/api/models")
async def api_list_models():
    """利用可能なOllamaモデル一覧を返す"""
    try:
        return await list_models()
    except Exception:
        return []


# ---------- チャット履歴 API ----------

@app.get("/api/sessions")
async def list_sessions(user: dict | None = Depends(get_current_user)):
    """チャットセッション一覧を返す（ログイン時は自分のセッションのみ）"""
    user_id = user["id"] if user else ""
    return get_sessions(user_id=user_id)


@app.get("/api/sessions/{session_id}")
async def get_session_messages(session_id: str, user: dict | None = Depends(get_current_user)):
    """セッションのメッセージ一覧を返す"""
    messages = get_messages(session_id)
    if not messages:
        raise HTTPException(status_code=404, detail="セッションが見つかりません")
    return {"session_id": session_id, "messages": messages}


@app.delete("/api/sessions/{session_id}")
async def remove_session(session_id: str, user: dict | None = Depends(get_current_user)):
    """セッションを削除する"""
    delete_session(session_id)
    return {"status": "ok"}


@app.get("/api/sessions/{session_id}/export")
async def export_session(session_id: str, user: dict | None = Depends(get_current_user)):
    """セッションをMarkdown形式でエクスポートする"""
    messages = get_messages(session_id)
    if not messages:
        raise HTTPException(status_code=404, detail="セッションが見つかりません")
    lines = []
    for msg in messages:
        if msg["role"] == "user":
            lines.append(f"**あなた:**\n{msg['content']}")
        else:
            lines.append(f"**アシスタント:**\n{msg['content']}")
    md = "\n\n---\n\n".join(lines)
    return Response(
        content=md,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="chat-{session_id}.md"'},
    )


# ---------- 利用ログ API（管理者のみ）----------

@app.get("/api/usage-logs")
async def api_usage_logs(limit: int = 100, offset: int = 0, _admin: dict = Depends(require_admin_user)):
    """利用ログを返す（管理者のみ）"""
    return {"logs": get_usage_logs(limit, offset), "stats": get_usage_stats()}


# ---------- Static / Frontend ----------

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/admin")
async def admin_page():
    return FileResponse("static/admin.html")
