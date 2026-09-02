import pytest

from app.ingest import ingest
from app.models import Chunk
from app.retrieve import dense_search, bm25_search, reciprocal_rank_fusion, retrieve


@pytest.fixture(scope="module")
def indexed_corpus(tmp_path_factory):
    corpus_dir = tmp_path_factory.mktemp("corpus")
    (corpus_dir / "animals.md").write_text(
        "Dogs are loyal pets that enjoy walks.\n\nCats often sleep sixteen hours a day.",
        encoding="utf-8",
    )
    (corpus_dir / "weather.md").write_text(
        "Hurricanes form over warm ocean water.\n\nSnow requires cold temperatures.",
        encoding="utf-8",
    )
    chroma_dir = tmp_path_factory.mktemp("chroma")
    ingest(corpus_dir=corpus_dir, chroma_dir=chroma_dir)
    return chroma_dir


def test_dense_search_finds_relevant_chunk(indexed_corpus):
    results = dense_search("Tell me about cats sleeping", k=2, chroma_dir=indexed_corpus)
    assert len(results) == 2
    assert any("Cats" in c.text for c in results)
    assert all(isinstance(c, Chunk) for c in results)


def test_bm25_search_finds_relevant_chunk(indexed_corpus):
    results = bm25_search("hurricanes ocean water", k=2, chroma_dir=indexed_corpus)
    assert len(results) == 2
    assert any("Hurricanes" in c.text for c in results)


def test_reciprocal_rank_fusion_combines_rankings():
    a = Chunk(id="a", text="a", source="s", chunk_index=0)
    b = Chunk(id="b", text="b", source="s", chunk_index=1)
    ranking1 = [a, b]
    ranking2 = [b, a]
    fused = reciprocal_rank_fusion([ranking1, ranking2])
    assert {c.id for c in fused} == {"a", "b"}
    assert fused[0].score == fused[1].score  # symmetric rankings tie


def test_retrieve_hybrid_mode_returns_k_results(indexed_corpus):
    results = retrieve("cats and dogs", mode="hybrid", k=2, chroma_dir=indexed_corpus)
    assert len(results) == 2


def test_retrieve_unknown_mode_raises(indexed_corpus):
    with pytest.raises(ValueError):
        retrieve("anything", mode="bogus", k=2, chroma_dir=indexed_corpus)
