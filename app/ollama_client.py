import json
from collections.abc import AsyncIterator

import httpx
from app.config import OLLAMA_BASE_URL, CHAT_MODEL, EMBED_MODEL


SYSTEM_PROMPTS = {
    "rag": (
        "あなたは社内ドキュメントに基づいて質問に回答するアシスタントです。\n"
        "以下の参考情報のみを使って回答してください。"
        "参考情報に答えがない場合は「情報が見つかりませんでした」と答えてください。\n\n"
        "## 参考情報\n{context}"
    ),
    "hybrid": (
        "あなたは社内ドキュメントと一般知識の両方を使って質問に回答するアシスタントです。\n"
        "以下の参考情報がある場合はそれを優先して回答してください。"
        "参考情報にない内容は一般知識で補足して構いませんが、"
        "その場合は「※一般知識に基づく補足です」と明記してください。\n\n"
        "## 参考情報\n{context}"
    ),
    "free": (
        "あなたは優秀な汎用AIアシスタントです。"
        "ユーザーの質問に対して、正確かつ簡潔に回答してください。"
    ),
    "stepwise": (
        "あなたは複雑な社内文書を段階的に読み解くアシスタントです。\n"
        "以下の参考情報を最優先で使い、必ず下記の4つの見出しの順で回答してください。\n"
        "見出しは省略せず、すべて出力してください。\n\n"
        "### 📋 抽出\n"
        "参考情報から質問に関連する重要な事実・数値・キーワードを箇条書きで列挙してください。\n\n"
        "### 📐 前提\n"
        "回答の前提となる条件・定義・制約・期間などを整理してください。\n\n"
        "### 🔍 分析\n"
        "抽出した情報を比較・計算・検証し、論理的に考察してください。\n\n"
        "### ✅ 結論\n"
        "最終的な回答を簡潔に述べ、根拠を1〜3点で示してください。\n\n"
        "不明な点は推測せず「不明」と明記し、必要な追加情報を示してください。\n"
        "回答はすべて日本語で記述してください。\n\n"
        "## 参考情報\n{context}"
    ),
}


def _build_messages(
    prompt: str,
    context: str | None = None,
    history: list[dict] | None = None,
    mode: str = "rag",
) -> list[dict]:
    template = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["rag"])
    if context and "{context}" in template:
        system_message = template.format(context=context)
    else:
        system_message = template.replace("\n\n## 参考情報\n{context}", "")
    messages: list[dict] = [{"role": "system", "content": system_message}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    return messages


async def list_models() -> list[dict]:
    """Ollamaに登録されているモデル一覧を返す"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
        resp.raise_for_status()
        models = resp.json().get("models", [])
        return [
            {
                "name": m["name"],
                "size": m.get("size", 0),
                "modified_at": m.get("modified_at", ""),
            }
            for m in models
        ]


async def get_embedding(text: str) -> list[float]:
    """Ollamaのembedding APIでテキストをベクトル化する"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]


async def chat_completion(
    prompt: str,
    context: str | None = None,
    history: list[dict] | None = None,
    mode: str = "rag",
    model: str = "",
) -> str:
    """Ollamaのチャット補完 API を呼び出す"""
    messages = _build_messages(prompt, context, history, mode)
    use_model = model or CHAT_MODEL
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": use_model,
                "messages": messages,
                "stream": False,
            },
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


async def chat_completion_stream(
    prompt: str,
    context: str | None = None,
    history: list[dict] | None = None,
    mode: str = "rag",
    model: str = "",
) -> AsyncIterator[str]:
    """ストリーミングでチャット補完 API を呼び出す"""
    messages = _build_messages(prompt, context, history, mode)
    use_model = model or CHAT_MODEL
    async with httpx.AsyncClient(timeout=300.0) as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": use_model,
                "messages": messages,
                "stream": True,
            },
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token
                if data.get("done"):
                    break
