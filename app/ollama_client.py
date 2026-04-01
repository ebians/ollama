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
) -> str:
    """Ollamaのチャット補完 API を呼び出す"""
    messages = _build_messages(prompt, context, history, mode)
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": CHAT_MODEL,
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
) -> AsyncIterator[str]:
    """ストリーミングでチャット補完 API を呼び出す"""
    messages = _build_messages(prompt, context, history, mode)
    async with httpx.AsyncClient(timeout=300.0) as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": CHAT_MODEL,
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
