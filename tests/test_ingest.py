from pathlib import Path

import pytest

from app.ingest import chunk_text, load_corpus, ingest, CHROMA_DIR


def test_chunk_text_splits_on_paragraphs_within_size():
    text = "Para one.\n\n" + ("word " * 100) + "\n\nPara three."
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) >= 1
    assert all(isinstance(c, str) and c.strip() for c in chunks)


def test_chunk_text_overlap_carries_context(tmp_path):
    paragraphs = "\n\n".join(f"Paragraph number {i} with some extra padding text." for i in range(20))
    chunks = chunk_text(paragraphs, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    # each chunk after the first should start with tail of the previous chunk (the overlap)
    for prev, curr in zip(chunks, chunks[1:]):
        assert curr.startswith(prev[-50:])


def test_load_corpus_reads_md_and_txt(tmp_path):
    (tmp_path / "a.md").write_text("hello md", encoding="utf-8")
    (tmp_path / "b.txt").write_text("hello txt", encoding="utf-8")
    (tmp_path / "ignore.json").write_text("{}", encoding="utf-8")
    docs = load_corpus(tmp_path)
    sources = {source for source, _ in docs}
    assert sources == {"a.md", "b.txt"}


def test_ingest_builds_chroma_and_bm25_index(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    # Paragraphs are long enough that each doc's two paragraphs don't fit in one
    # 500-char chunk, so chunk_text's overflow branch splits them into 2 chunks each.
    para_a = "The sky is blue during the day. " * 9
    para_b = "Water boils at 100 degrees Celsius. " * 9
    (corpus_dir / "doc1.md").write_text(f"{para_a}\n\n{para_b}", encoding="utf-8")
    para_c = "Cats are mammals that sleep most of the day. " * 8
    para_d = "The ocean covers most of the Earth's surface. " * 8
    (corpus_dir / "doc2.md").write_text(f"{para_c}\n\n{para_d}", encoding="utf-8")
    chroma_dir = tmp_path / "chroma_db"

    count = ingest(corpus_dir=corpus_dir, chroma_dir=chroma_dir)

    assert count == 4
    assert (chroma_dir / "bm25_index.pkl").exists()

    import chromadb
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection("eval_harness_docs")
    assert collection.count() == 4
