import chromadb
from chromadb.config import Settings

from app.config import CHROMA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
from app.ollama_client import get_embedding

_client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
_collection = _client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"},
)


def _split_text(text: str) -> list[str]:
    """テキストをチャンクに分割する（シンプルな文字数ベース）"""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c.strip() for c in chunks if c.strip()]


async def add_document(doc_id: str, filename: str, text: str) -> int:
    """ドキュメントをチャンク分割し、ベクトルDBに格納する"""
    chunks = _split_text(text)
    ids: list[str] = []
    embeddings: list[list[float]] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk{i}"
        embedding = await get_embedding(chunk)
        ids.append(chunk_id)
        embeddings.append(embedding)
        documents.append(chunk)
        metadatas.append({"source": filename, "chunk_index": i})

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
        hits.append({"text": doc, "source": meta["source"], "distance": round(dist, 4)})
    return hits


def get_stats() -> dict:
    """DB統計を返す"""
    return {"total_chunks": _collection.count()}


def reset_db():
    """ドキュメントを全削除する"""
    _client.delete_collection("documents")
    global _collection
    _collection = _client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"},
    )
