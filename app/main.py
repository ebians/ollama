import json
import uuid

from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app.ollama_client import chat_completion, chat_completion_stream
from app.parser import extract_text, SUPPORTED_EXTENSIONS
from app.vectorstore import add_document, search, get_stats, reset_db

app = FastAPI(title="Ollama RAG App")


class AskRequest(BaseModel):
    question: str
    history: list[dict] = []
    mode: str = "rag"  # "rag" | "hybrid" | "free"


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


# ---------- API ----------

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """ファイル(.txt/.pdf/.docx)をアップロードし、ベクトルDBに格納する"""
    filename = file.filename or "unknown"
    ext = ("." + filename.rsplit(".", 1)[-1]).lower() if "." in filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        return {"error": f"未対応の形式です。対応形式: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"}
    content = await file.read()
    text = extract_text(filename, content)
    doc_id = uuid.uuid4().hex[:12]
    num_chunks = await add_document(doc_id, filename, text)
    return {"doc_id": doc_id, "filename": filename, "chunks": num_chunks}


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
    """ストリーミング回答（SSE、モード対応）"""
    hits = []
    context = None
    if req.mode in ("rag", "hybrid"):
        hits = await search(req.question)
        if not hits and req.mode == "rag":
            async def no_docs():
                data = json.dumps({"type": "sources", "sources": []}, ensure_ascii=False)
                yield f"data: {data}\n\n"
                data = json.dumps({"type": "token", "token": "ドキュメントが登録されていません。先にファイルをアップロードしてください。"}, ensure_ascii=False)
                yield f"data: {data}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(no_docs(), media_type="text/event-stream")
        if hits:
            context = "\n\n---\n\n".join(h["text"] for h in hits)

    async def generate():
        data = json.dumps({"type": "sources", "sources": hits}, ensure_ascii=False)
        yield f"data: {data}\n\n"
        async for token in chat_completion_stream(req.question, context, req.history, req.mode):
            data = json.dumps({"type": "token", "token": token}, ensure_ascii=False)
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/stats")
async def stats():
    """DB統計を返す"""
    return get_stats()


@app.post("/api/reset")
async def reset():
    """ドキュメントを全削除する"""
    reset_db()
    return {"status": "ok"}


# ---------- Static / Frontend ----------

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")
