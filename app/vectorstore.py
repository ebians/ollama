import re

import chromadb
from chromadb.config import Settings

from app.config import CHROMA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
from app.ollama_client import get_embedding

_client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
_collection = _client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"},
)

# ページ番号コメントのパターン
_PAGE_COMMENT = re.compile(r"<!--\s*page:(\d+)\s*-->")

# 見出し・段落区切りのパターン（Markdown見出し、番号付き見出し、空行2つ以上）
_SPLIT_PATTERN = re.compile(
    r"(?=^#{1,3}\s)"        # Markdown 見出し (# / ## / ###)
    r"|(?=^\d+[\.\)]\s)"    # 番号付き見出し (1. / 2) など)
    r"|(?:\n\s*\n)",         # 空行区切り（段落）
    re.MULTILINE,
)


def _split_text(text: str) -> list[str]:
    """テキストを段落・見出し単位で分割し、CHUNK_SIZE 以下にまとめる"""
    # まず意味単位で分割
    segments = _SPLIT_PATTERN.split(text)
    segments = [s.strip() for s in segments if s and s.strip()]

    # CHUNK_SIZE 以下になるようにセグメントを結合
    chunks: list[str] = []
    current = ""
    for seg in segments:
        # 単体で CHUNK_SIZE を超えるセグメントはさらに文字数で分割
        if len(seg) > CHUNK_SIZE:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(seg):
                end = start + CHUNK_SIZE
                chunks.append(seg[start:end].strip())
                start += CHUNK_SIZE - CHUNK_OVERLAP
            continue

        if len(current) + len(seg) + 1 > CHUNK_SIZE:
            if current:
                chunks.append(current)
            current = seg
        else:
            current = f"{current}\n{seg}" if current else seg

    if current:
        chunks.append(current)

    return [c for c in chunks if c]


async def add_document(doc_id: str, filename: str, text: str) -> int:
    """ドキュメントをチャンク分割し、ベクトルDBに格納する"""
    chunks = _split_text(text)
    ids: list[str] = []
    embeddings: list[list[float]] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk{i}"
        # チャンクからページ番号を抽出
        page_match = _PAGE_COMMENT.search(chunk)
        page_num = int(page_match.group(1)) if page_match else 0
        # ページコメントを除去してembedding
        clean_chunk = _PAGE_COMMENT.sub("", chunk).strip()
        embedding = await get_embedding(clean_chunk)
        ids.append(chunk_id)
        embeddings.append(embedding)
        documents.append(clean_chunk)
        meta = {"source": filename, "chunk_index": i}
        if page_num:
            meta["page"] = page_num
        # 表を含むかどうかのフラグ
        if "| --- |" in chunk or "|---|" in chunk:
            meta["has_table"] = True
        metadatas.append(meta)

    _collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    return len(chunks)


async def search(query: str) -> list[dict]:
    """クエリに類似したチャンクを検索する"""
    if _collection.count() == 0:
        return []
    query_embedding = await get_embedding(query)
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=min(TOP_K, _collection.count()),
    )
    hits: list[dict] = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hit = {"text": doc, "source": meta["source"], "distance": round(dist, 4)}
        if meta.get("page"):
            hit["page"] = meta["page"]
        if meta.get("has_table"):
            hit["has_table"] = True
        hits.append(hit)
    return hits


def get_stats() -> dict:
    """DB統計を返す"""
    return {"total_chunks": _collection.count()}


def list_documents() -> list[dict]:
    """登録済みドキュメントの一覧を返す（ソース名ごとにチャンク数を集計）"""
    if _collection.count() == 0:
        return []
    all_data = _collection.get(include=["metadatas"])
    doc_map: dict[str, int] = {}
    for meta in all_data["metadatas"]:
        source = meta.get("source", "unknown")
        doc_map[source] = doc_map.get(source, 0) + 1
    return [{"source": src, "chunks": cnt} for src, cnt in sorted(doc_map.items())]


def get_document_chunks(source: str) -> list[dict]:
    """指定ドキュメントのチャンク一覧を返す"""
    if _collection.count() == 0:
        return []
    all_data = _collection.get(include=["metadatas", "documents"])
    chunks = []
    for id_, meta, doc in zip(all_data["ids"], all_data["metadatas"], all_data["documents"]):
        if meta.get("source") == source:
            chunks.append({
                "id": id_,
                "chunk_index": meta.get("chunk_index", 0),
                "text": doc,
                "length": len(doc),
            })
    chunks.sort(key=lambda c: c["chunk_index"])
    return chunks


def delete_document(source: str) -> int:
    """指定ソース名のチャンクを全て削除する"""
    if _collection.count() == 0:
        return 0
    all_data = _collection.get(include=["metadatas"])
    ids_to_delete = [
        id_ for id_, meta in zip(all_data["ids"], all_data["metadatas"])
        if meta.get("source") == source
    ]
    if ids_to_delete:
        _collection.delete(ids=ids_to_delete)
    return len(ids_to_delete)


def reset_db():
    """ドキュメントを全削除する"""
    _client.delete_collection("documents")
    global _collection
    _collection = _client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"},
    )
