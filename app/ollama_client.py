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
    "multimodal": (
        "あなたは社内ドキュメントの図表・画像・テキストを総合的に理解して回答するアシスタントです。\n"
        "添付された画像はドキュメントのページ画像です。画像に含まれる図・表・グラフ・注記を読み取って回答してください。\n"
        "テキスト情報と画像情報の両方がある場合は、両方を照合して正確に回答してください。\n"
        "画像から読み取った数値や情報は具体的に示してください。\n\n"
        "## テキスト参考情報\n{context}"
    ),
    "calculate": (
        "あなたは社内ドキュメントの表データを分析・計算して回答するアシスタントです。\n"
        "以下の参考情報には表データが含まれています。\n"
        "数値の比較・差分・前年同月比・合計・平均など、計算が必要な質問には必ず計算過程を示してください。\n"
        "計算結果は具体的な数値で答え、単位も明記してください。\n"
        "表データの構造を正しく理解し、行と列の対応関係を間違えないでください。\n\n"
        "## 構造化データ\n{structured}\n\n"
        "## 参考情報\n{context}"
    ),
    "consistency": (
        "あなたは複数の社内ドキュメント間の整合性をチェックする専門アシスタントです。\n"
        "以下の文書群を比較し、矛盾・不一致・未定義の要件を検出してください。\n"
        "必ず以下の見出しの順で回答してください。\n\n"
        "### 🔍 検出された不一致\n"
        "矛盾や不一致を箇条書きで列挙してください。各項目に文書名・該当箇所を明記。\n\n"
        "### ⚠️ 潜在的リスク\n"
        "不一致が引き起こしうる問題やリスクを列挙。\n\n"
        "### ✅ 推奨対応\n"
        "各不一致に対する修正案を具体的に示してください。\n\n"
        "## 文書データ\n{context}"
    ),
}


def _build_messages(
    prompt: str,
    context: str | None = None,
    history: list[dict] | None = None,
    mode: str = "rag",
    structured: str | None = None,
) -> list[dict]:
    template = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["rag"])
    if context and "{context}" in template:
        system_message = template.format(
            context=context,
            structured=structured or "",
        ) if "{structured}" in template else template.format(context=context)
    else:
        system_message = template.replace("\n\n## 参考情報\n{context}", "")
        if "{structured}" in system_message:
            system_message = system_message.replace("\n\n## 構造化データ\n{structured}", "")
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
    structured: str | None = None,
) -> str:
    """Ollamaのチャット補完 API を呼び出す"""
    messages = _build_messages(prompt, context, history, mode, structured)
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
    structured: str | None = None,
) -> AsyncIterator[str]:
    """ストリーミングでチャット補完 API を呼び出す"""
    messages = _build_messages(prompt, context, history, mode, structured)
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


async def chat_completion_vision(
    prompt: str,
    images_b64: list[str],
    context: str | None = None,
    history: list[dict] | None = None,
    model: str = "",
) -> str:
    """Vision対応モデルに画像付きでチャット補完を呼び出す"""
    messages = _build_messages(prompt, context, history, mode="multimodal")
    # ユーザーメッセージに画像を添付
    messages[-1]["images"] = images_b64
    use_model = model or CHAT_MODEL
    async with httpx.AsyncClient(timeout=180.0) as client:
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


async def chat_completion_vision_stream(
    prompt: str,
    images_b64: list[str],
    context: str | None = None,
    history: list[dict] | None = None,
    model: str = "",
) -> AsyncIterator[str]:
    """Vision対応モデルにストリーミングで画像付きチャット補完を呼び出す"""
    messages = _build_messages(prompt, context, history, mode="multimodal")
    messages[-1]["images"] = images_b64
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


_SUMMARIZE_SYSTEM = (
    "あなたはインタビュー記録を整理・要約する専門家です。\n"
    "以下のインタビュー記録（質問と回答のペア）を読み、次の2つを生成してください。\n\n"
    "## 出力形式\n"
    "必ず以下のJSON形式で出力してください。JSON以外のテキストは不要です。\n"
    "```json\n"
    '{\n'
    '  "summary": "インタビュー内容の要約（200-400字）",\n'
    '  "faq": [\n'
    '    {"q": "質問文", "a": "回答文"},\n'
    '    {"q": "質問文", "a": "回答文"}\n'
    '  ]\n'
    '}\n'
    "```\n\n"
    "## ルール\n"
    "- FAQは5～15件程度を目安に生成\n"
    "- 口語的な表現は丁寧な文語体に変換\n"
    "- 曖昧な表現は明確化し、具体的な数値や手順を保持\n"
    "- 業務ノウハウの本質を失わないこと\n"
    "- 出力はJSON以外含めないこと\n"
)


async def generate_summary_and_faq(
    interview_data: list[dict],
    title: str = "",
    model: str = "",
) -> dict:
    """インタビューデータからLLMで要約とFAQを生成する。

    Args:
        interview_data: [{"q": "質問", "a": "回答"}, ...]
        title: ワークフローのタイトル（コンテキスト用）

    Returns:
        {"summary": "...", "faq": [{"q": "...", "a": "..."}, ...]}
    """
    qa_text = "\n\n".join(
        f"質問: {item.get('q', '')}\n回答: {item.get('a', '')}"
        for item in interview_data
    )
    prompt = f"テーマ: {title}\n\n## インタビュー記録\n{qa_text}"
    messages = [
        {"role": "system", "content": _SUMMARIZE_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    use_model = model or CHAT_MODEL
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={"model": use_model, "messages": messages, "stream": False},
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]

    # JSONを抽出（```json ... ``` やプレーンJSON対応）
    import re
    json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    if json_match:
        content = json_match.group(1)
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = {"summary": content, "faq": []}
    if "summary" not in result:
        result["summary"] = content
    if "faq" not in result:
        result["faq"] = []
    return result
