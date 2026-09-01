from pathlib import Path

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CORPUS_DIR = Path("data/corpus")
CHROMA_DIR = Path("data/chroma_db")
COLLECTION_NAME = "eval_harness_docs"
BM25_INDEX_PATH = CHROMA_DIR / "bm25_index.pkl"

_embedder = None


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > chunk_size:
            chunks.append(current)
            current = (current[-overlap:] + "\n\n" + para) if overlap else para
        elif not current:
            current = para
        elif len(para) < chunk_size * 0.1:
            # Small paragraph: keep separate unless current is substantial
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}"
    if current:
        chunks.append(current)
    return chunks


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> list[tuple[str, str]]:
    docs = []
    for path in sorted(corpus_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
            docs.append((str(path.relative_to(corpus_dir)).replace("\\", "/"), path.read_text(encoding="utf-8")))
    return docs


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def ingest(corpus_dir: Path = CORPUS_DIR, chroma_dir: Path = CHROMA_DIR) -> int:
    import pickle

    import chromadb
    from rank_bm25 import BM25Okapi

    docs = load_corpus(corpus_dir)
    chunk_ids, chunk_texts, chunk_sources, chunk_indices = [], [], [], []
    for source, text in docs:
        for i, chunk in enumerate(chunk_text(text)):
            chunk_ids.append(f"{source}::{i}")
            chunk_texts.append(chunk)
            chunk_sources.append(source)
            chunk_indices.append(i)

    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    if chunk_texts:
        embeddings = get_embedder().encode(chunk_texts).tolist()
        collection.add(
            ids=chunk_ids,
            documents=chunk_texts,
            embeddings=embeddings,
            metadatas=[{"source": s, "chunk_index": i} for s, i in zip(chunk_sources, chunk_indices)],
        )

    tokenized = [t.lower().split() for t in chunk_texts]
    bm25 = BM25Okapi(tokenized) if tokenized else None
    bm25_path = chroma_dir / "bm25_index.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "ids": chunk_ids,
                "texts": chunk_texts,
                "sources": chunk_sources,
                "chunk_indices": chunk_indices,
            },
            f,
        )

    return len(chunk_ids)
