import json
import secrets
import uuid

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Request, Response, Cookie
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

from app.config import ADMIN_PASSWORD
from app.ollama_client import chat_completion, chat_completion_stream, chat_completion_vision_stream, list_models, generate_summary_and_faq
from app.rate_limit import check_rate_limit
from app.audit_log import add_usage_log, get_usage_logs, get_usage_stats
from app.parser import extract_text, extract_pdf_page_images, SUPPORTED_EXTENSIONS
from app.vectorstore import add_document, search, get_stats, reset_db, list_documents, delete_document, get_document_chunks
from app.page_image_store import save_page_image, get_page_images, delete_page_images
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
    mode: str = "rag"  # "rag" | "hybrid" | "free" | "stepwise" | "multimodal" | "calculate" | "consistency"
    session_id: str = ""  # 空なら新規セッション
    model: str = ""  # 空ならデフォルトモデル
    doc_filter: list[str] = []  # 整合性チェック用: 対象ドキュメント名リスト


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
    # PDFの場合はページ画像も保存（マルチモーダルRAG用）
    page_images_count = 0
    if ext == ".pdf":
        try:
            page_imgs = extract_pdf_page_images(content)
            for img in page_imgs:
                save_page_image(filename, img["page"], img["image_b64"], img["width"], img["height"])
            page_images_count = len(page_imgs)
        except Exception:
            pass  # 画像抽出失敗時はテキストRAGのみで動作
    return {"doc_id": doc_id, "filename": filename, "chunks": num_chunks, "page_images": page_images_count}


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
    delete_page_images(source)
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
    if req.mode in ("rag", "hybrid", "stepwise", "multimodal", "calculate", "consistency"):
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
    structured = None
    images_b64 = []
    if req.mode in ("rag", "hybrid", "stepwise", "multimodal", "calculate", "consistency"):
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

        # マルチモーダルモード: 参照元のページ画像を取得
        if req.mode == "multimodal" and hits:
            source_pages: dict[str, list[int]] = {}
            for h in hits:
                if h.get("page"):
                    source_pages.setdefault(h["source"], []).append(h["page"])
            for source, pages in source_pages.items():
                page_imgs = get_page_images(source, pages)
                for img in page_imgs:
                    images_b64.append(img["image_b64"])
            # 画像は最大4ページに制限（VRAM節約）
            images_b64 = images_b64[:4]

        # 計算モード: 表データを構造化形式で抽出
        if req.mode == "calculate" and hits:
            from app.calc_engine import extract_tables_from_text, format_structured_tables
            tables = []
            for h in hits:
                tables.extend(extract_tables_from_text(h["text"]))
            if tables:
                structured = format_structured_tables(tables)

        # 整合性チェックモード: ルールベース検証結果をコンテキストに追加
        if req.mode == "consistency" and hits:
            from app.consistency_checker import run_consistency_checks
            check_result = run_consistency_checks(hits)
            rule_context = check_result.to_context()
            context = f"{rule_context}\n\n---\n\n## 文書テキスト\n{context}"

    async def generate():
        # セッションIDを送信
        data = json.dumps({"type": "session_id", "session_id": session_id}, ensure_ascii=False)
        yield f"data: {data}\n\n"
        data = json.dumps({"type": "sources", "sources": hits}, ensure_ascii=False)
        yield f"data: {data}\n\n"
        full_answer = ""
        # マルチモーダルモードで画像がある場合はVision対応ストリーミング
        if req.mode == "multimodal" and images_b64:
            async for token in chat_completion_vision_stream(
                req.question, images_b64, context, req.history, req.model
            ):
                full_answer += token
                data = json.dumps({"type": "token", "token": token}, ensure_ascii=False)
                yield f"data: {data}\n\n"
        else:
            async for token in chat_completion_stream(
                req.question, context, req.history, req.mode, req.model, structured
            ):
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


# ---------- 整合性チェック API（管理者のみ）----------

class ConsistencyCheckRequest(BaseModel):
    sources: list[str] = []  # 空なら全ドキュメント

@app.post("/api/consistency-check")
async def api_consistency_check(req: ConsistencyCheckRequest, _admin: dict = Depends(require_admin_user)):
    """ドキュメント間のルールベース整合性チェックを実行する（管理者専用）"""
    from app.consistency_checker import run_consistency_checks
    from app.vectorstore import _collection

    if _collection.count() == 0:
        return {"summary": "ドキュメントが登録されていません", "issues": []}

    all_data = _collection.get(include=["metadatas", "documents"])
    chunks = []
    for meta, doc in zip(all_data["metadatas"], all_data["documents"]):
        if req.sources and meta.get("source") not in req.sources:
            continue
        chunks.append({"text": doc, "source": meta.get("source", "unknown")})

    result = run_consistency_checks(chunks)
    issues = [
        {
            "category": i.category,
            "severity": i.severity,
            "description": i.description,
            "source_a": i.source_a,
            "source_b": i.source_b,
            "text_a": i.text_a,
            "text_b": i.text_b,
        }
        for i in result.inconsistencies
    ]
    return {"summary": result.summary, "issues": issues}


# ---------- 暗黙知ワークフロー API（管理者のみ）----------

from app.knowledge_store import (
    create_workflow, get_workflow, list_workflows, update_interview_data,
    save_summary, submit_for_review, approve_workflow, reject_workflow,
    set_doc_id, delete_workflow, get_workflow_stats,
)


class CreateWorkflowRequest(BaseModel):
    title: str
    category: str = ""
    interviewee: str = ""

class InterviewDataRequest(BaseModel):
    interview_data: list[dict]  # [{"q": "質問", "a": "回答"}, ...]

class GenerateSummaryRequest(BaseModel):
    model: str = ""

class ReviewRequest(BaseModel):
    action: str  # "approve" | "reject"
    review_notes: str = ""


@app.get("/api/knowledge/stats")
async def api_knowledge_stats(_admin: dict = Depends(require_admin_user)):
    """ワークフロー統計"""
    return get_workflow_stats()


@app.get("/api/knowledge")
async def api_list_workflows(stage: str = "", _admin: dict = Depends(require_admin_user)):
    """ワークフロー一覧"""
    return list_workflows(stage)


@app.post("/api/knowledge")
async def api_create_workflow(req: CreateWorkflowRequest, admin: dict = Depends(require_admin_user)):
    """新規ワークフロー作成"""
    wf_id = create_workflow(req.title, req.category, req.interviewee, admin["id"])
    return {"id": wf_id}


@app.get("/api/knowledge/{wf_id}")
async def api_get_workflow(wf_id: str, _admin: dict = Depends(require_admin_user)):
    """ワークフロー詳細"""
    wf = get_workflow(wf_id)
    if not wf:
        raise HTTPException(status_code=404, detail="ワークフローが見つかりません")
    return wf


@app.put("/api/knowledge/{wf_id}/interview")
async def api_update_interview(wf_id: str, req: InterviewDataRequest, _admin: dict = Depends(require_admin_user)):
    """インタビューデータ保存"""
    ok = update_interview_data(wf_id, req.interview_data)
    if not ok:
        raise HTTPException(status_code=400, detail="draftステージのワークフローのみ編集できます")
    return {"status": "ok"}


@app.post("/api/knowledge/{wf_id}/summarize")
async def api_generate_summary(wf_id: str, req: GenerateSummaryRequest, _admin: dict = Depends(require_admin_user)):
    """LLMで要約・FAQ生成"""
    wf = get_workflow(wf_id)
    if not wf:
        raise HTTPException(status_code=404, detail="ワークフローが見つかりません")
    if not wf["interview_data"]:
        raise HTTPException(status_code=400, detail="インタビューデータが空です")
    result = await generate_summary_and_faq(wf["interview_data"], wf["title"], req.model)
    save_summary(wf_id, result["summary"], result["faq"])
    return result


@app.post("/api/knowledge/{wf_id}/submit-review")
async def api_submit_review(wf_id: str, _admin: dict = Depends(require_admin_user)):
    """レビュー依頼"""
    ok = submit_for_review(wf_id)
    if not ok:
        raise HTTPException(status_code=400, detail="summaryステージのワークフローのみレビュー依頼できます")
    return {"status": "ok"}


@app.post("/api/knowledge/{wf_id}/review")
async def api_review_workflow(wf_id: str, req: ReviewRequest, admin: dict = Depends(require_admin_user)):
    """レビュー承認/差し戻し"""
    if req.action == "approve":
        ok = approve_workflow(wf_id, admin["id"], req.review_notes)
    elif req.action == "reject":
        ok = reject_workflow(wf_id, admin["id"], req.review_notes)
    else:
        raise HTTPException(status_code=400, detail="actionは 'approve' または 'reject' です")
    if not ok:
        raise HTTPException(status_code=400, detail="reviewステージのワークフローのみ操作できます")
    return {"status": "ok"}


@app.post("/api/knowledge/{wf_id}/publish")
async def api_publish_workflow(wf_id: str, _admin: dict = Depends(require_admin_user)):
    """承認済みワークフローをRAGに公開（FAQをベクトルDBに登録）"""
    wf = get_workflow(wf_id)
    if not wf:
        raise HTTPException(status_code=404, detail="ワークフローが見つかりません")
    if wf["stage"] != "published":
        raise HTTPException(status_code=400, detail="承認済み(published)のワークフローのみ公開できます")
    if wf["doc_id"]:
        raise HTTPException(status_code=400, detail="既にRAGに公開済みです")

    # FAQ + 要約をテキスト化してベクトルDBに登録
    parts = [f"# {wf['title']}"]
    if wf["summary"]:
        parts.append(f"\n## 概要\n{wf['summary']}")
    if wf.get("category"):
        parts.append(f"カテゴリ: {wf['category']}")
    if wf.get("interviewee"):
        parts.append(f"ナレッジ提供者: {wf['interviewee']}")
    parts.append("\n## FAQ")
    for i, faq_item in enumerate(wf["faq"], 1):
        q = faq_item.get("q", "")
        a = faq_item.get("a", "")
        parts.append(f"\n### Q{i}. {q}\n{a}")

    text = "\n".join(parts)
    doc_name = f"knowledge_{wf['id']}_{wf['title'][:20]}.txt"
    doc_id = uuid.uuid4().hex[:12]
    num_chunks = await add_document(doc_id, doc_name, text)
    set_doc_id(wf_id, doc_id)
    return {"doc_id": doc_id, "doc_name": doc_name, "chunks": num_chunks}


@app.delete("/api/knowledge/{wf_id}")
async def api_delete_workflow(wf_id: str, _admin: dict = Depends(require_admin_user)):
    """ワークフロー削除"""
    ok = delete_workflow(wf_id)
    if not ok:
        raise HTTPException(status_code=404, detail="ワークフローが見つかりません")
    return {"status": "ok"}


# ---------- Static / Frontend ----------

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/admin")
async def admin_page():
    return FileResponse("static/admin.html")
