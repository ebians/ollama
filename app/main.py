import json
import secrets
import uuid

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

from app.config import ADMIN_PASSWORD
from app.ollama_client import chat_completion, chat_completion_stream
from app.parser import extract_text, SUPPORTED_EXTENSIONS
from app.vectorstore import add_document, search, get_stats, reset_db, list_documents, delete_document
from app.chat_store import (
    create_session, add_message, get_sessions, get_messages, delete_session,
)

app = FastAPI(title="Ollama RAG App")
security = HTTPBasic()


def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    """管理者パスワードを検証する"""
    if not secrets.compare_digest(credentials.password, ADMIN_PASSWORD):
        raise HTTPException(status_code=401, detail="認証エラー", headers={"WWW-Authenticate": "Basic"})
    return credentials.username


class AskRequest(BaseModel):
    question: str
    history: list[dict] = []
    mode: str = "rag"  # "rag" | "hybrid" | "free"
    session_id: str = ""  # 空なら新規セッション


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


# ---------- API ----------

# ---------- 管理者専用 API ----------

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...), _user: str = Depends(verify_admin)):
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
async def get_documents(_user: str = Depends(verify_admin)):
    """登録済みドキュメント一覧を返す（管理者専用）"""
    return list_documents()


@app.delete("/api/documents/{source:path}")
async def remove_document(source: str, _user: str = Depends(verify_admin)):
    """指定ドキュメントを削除する（管理者専用）"""
    deleted = delete_document(source)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="ドキュメントが見つかりません")
    return {"source": source, "deleted_chunks": deleted}


@app.post("/api/reset")
async def reset(_user: str = Depends(verify_admin)):
    """ドキュメントを全削除する（管理者専用）"""
    reset_db()
    return {"status": "ok"}


@app.post("/api/ask", response_model=AskResponse)
async def ask_question(req: AskRequest):
    """質問に対して回答する（モード対応）"""
    hits = []
    context = None
    if req.mode in ("rag", "hybrid"):
        hits = await search(req.question)
        if not hits and req.mode == "rag":
            return AskResponse(answer="ドキュメントが登録されていません。先にファイルをアップロードしてください。", sources=[])
        if hits:
            context = "\n\n---\n\n".join(h["text"] for h in hits)
    answer = await chat_completion(req.question, context, req.history, req.mode)
    return AskResponse(answer=answer, sources=hits)


@app.post("/api/ask/stream")
async def ask_question_stream(req: AskRequest):
    """ストリーミング回答（SSE、モード対応、履歴保存）"""
    # セッション管理
    session_id = req.session_id or create_session(req.question[:50], req.mode)
    add_message(session_id, "user", req.question)

    hits = []
    context = None
    if req.mode in ("rag", "hybrid"):
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
        async for token in chat_completion_stream(req.question, context, req.history, req.mode):
            full_answer += token
            data = json.dumps({"type": "token", "token": token}, ensure_ascii=False)
            yield f"data: {data}\n\n"
        # 回答をDBに保存
        add_message(session_id, "assistant", full_answer, hits if hits else None)
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/stats")
async def stats():
    """DB統計を返す"""
    return get_stats()


# ---------- チャット履歴 API ----------

@app.get("/api/sessions")
async def list_sessions():
    """チャットセッション一覧を返す"""
    return get_sessions()


@app.get("/api/sessions/{session_id}")
async def get_session_messages(session_id: str):
    """セッションのメッセージ一覧を返す"""
    messages = get_messages(session_id)
    if not messages:
        raise HTTPException(status_code=404, detail="セッションが見つかりません")
    return {"session_id": session_id, "messages": messages}


@app.delete("/api/sessions/{session_id}")
async def remove_session(session_id: str):
    """セッションを削除する"""
    delete_session(session_id)
    return {"status": "ok"}


# ---------- Static / Frontend ----------

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/admin")
async def admin_page():
    return FileResponse("static/admin.html")
