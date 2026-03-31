import httpx
from app.config import OLLAMA_BASE_URL, CHAT_MODEL, EMBED_MODEL


async def get_embedding(text: str) -> list[float]:
    """Ollamaのembedding APIでテキストをベクトル化する"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]


async def chat_completion(prompt: str, context: str) -> str:
    """Ollamaのチャット補完 API を呼び出す"""
    system_message = (
        "あなたは社内ドキュメントに基づいて質問に回答するアシスタントです。\n"
        "以下の参考情報のみを使って回答してください。"
        "参考情報に答えがない場合は「情報が見つかりませんでした」と答えてください。\n\n"
        f"## 参考情報\n{context}"
    )
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": CHAT_MODEL,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            },
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
