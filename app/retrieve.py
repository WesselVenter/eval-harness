import pickle
from pathlib import Path
from typing import Literal

from app.ingest import CHROMA_DIR, COLLECTION_NAME, get_embedder
from app.models import Chunk


def _load_bm25_index(chroma_dir: Path = CHROMA_DIR) -> dict:
    with open(chroma_dir / "bm25_index.pkl", "rb") as f:
        return pickle.load(f)


def dense_search(query: str, k: int = 4, chroma_dir: Path = CHROMA_DIR) -> list[Chunk]:
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection(COLLECTION_NAME)
    query_embedding = get_embedder().encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=k)

    chunks = []
    for i, chunk_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        chunks.append(
            Chunk(
                id=chunk_id,
                text=results["documents"][0][i],
                source=meta["source"],
                chunk_index=meta["chunk_index"],
                score=1.0 - distance,
            )
        )
    return chunks


def bm25_search(query: str, k: int = 4, chroma_dir: Path = CHROMA_DIR) -> list[Chunk]:
    index = _load_bm25_index(chroma_dir)
    bm25 = index["bm25"]
    if bm25 is None:
        return []
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        Chunk(
            id=index["ids"][i],
            text=index["texts"][i],
            source=index["sources"][i],
            chunk_index=index["chunk_indices"][i],
            score=float(scores[i]),
        )
        for i in ranked
    ]


def reciprocal_rank_fusion(rankings: list[list[Chunk]], k_const: int = 60) -> list[Chunk]:
    scores: dict[str, float] = {}
    chunk_by_id: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k_const + rank + 1)
            chunk_by_id[chunk.id] = chunk
    ordered_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [
        Chunk(
            id=chunk_by_id[cid].id,
            text=chunk_by_id[cid].text,
            source=chunk_by_id[cid].source,
            chunk_index=chunk_by_id[cid].chunk_index,
            score=scores[cid],
        )
        for cid in ordered_ids
    ]


def retrieve(
    query: str,
    mode: Literal["dense", "bm25", "hybrid"] = "hybrid",
    k: int = 4,
    chroma_dir: Path = CHROMA_DIR,
) -> list[Chunk]:
    if mode == "dense":
        return dense_search(query, k, chroma_dir)
    if mode == "bm25":
        return bm25_search(query, k, chroma_dir)
    if mode == "hybrid":
        pool = k * 3
        dense = dense_search(query, pool, chroma_dir)
        bm25 = bm25_search(query, pool, chroma_dir)
        return reciprocal_rank_fusion([dense, bm25])[:k]
    raise ValueError(f"Unknown mode: {mode}")
