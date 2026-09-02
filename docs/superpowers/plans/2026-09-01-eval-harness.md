# Eval Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a toy RAG app (`app/`) and an LLM-judge eval harness (`eval/`) that scores it and gates CI on quality thresholds.

**Architecture:** `app/ingest.py` chunks markdown/text docs, embeds them locally with sentence-transformers, and stores them in ChromaDB (dense) plus a pickled BM25 index (sparse). `app/retrieve.py` exposes dense/bm25/hybrid (reciprocal rank fusion) retrieval. `app/generate.py` calls the Anthropic API to answer questions from retrieved context. `eval/run_eval.py` runs the golden dataset through retrieve→generate→judge and writes a scored results file; `tests/test_regression.py` asserts the latest run clears fixed thresholds, and CI runs that as the quality gate.

**Tech Stack:** Python 3.11+, `uv`, `chromadb`, `sentence-transformers` (`all-MiniLM-L6-v2`), `rank-bm25`, `anthropic` SDK, `pytest`.

## Global Constraints

- Working directory for all commands is the project root: `eval-harness/`.
- Python `>=3.11`. Environment/dependency management is `uv` exclusively (no bare `pip`).
- Embedding model: `all-MiniLM-L6-v2` via `sentence-transformers`, loaded lazily and cached module-level.
- Default LLM for both generation and judging: `claude-sonnet-5` (override via `EVAL_HARNESS_MODEL` env var).
- Chunking: ~500 chars per chunk, 50 chars overlap, paragraph-aware (spec: `docs/superpowers/specs/2026-09-01-eval-harness-design.md`).
- Threshold constants (from spec, live in `eval/metrics.py`): `MIN_FAITHFULNESS = 4.0`, `MIN_CORRECTNESS_RATE = 0.7`, `MAX_HALLUCINATION_RATE = 0.2`.
- Tests that call the real Anthropic API (`tests/test_regression.py`) must skip (not fail) when `ANTHROPIC_API_KEY` is unset. All other tests must mock the Anthropic client and must NOT require an API key or network access.
- Retrieval/ingest code must not require an API key (embeddings are local).

---

### Task 1: Project scaffolding and tooling

**Files:**
- Create: `pyproject.toml`
- Create: `app/__init__.py`
- Create: `eval/__init__.py`
- Create: `.gitignore`
- Create: `results/.gitkeep`

**Interfaces:**
- Produces: an installable `eval-harness` project with `app` and `eval` as importable packages, a `rag` console script entry point (`app.cli:main`, implemented in Task 6), and a synced `.venv` with all runtime deps available for later tasks.

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p app eval tests data/corpus results .github/workflows
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "eval-harness"
version = "0.1.0"
description = "Toy RAG app + LLM-as-judge eval harness"
requires-python = ">=3.11"
dependencies = [
    "anthropic>=0.40.0",
    "chromadb>=0.5.0",
    "sentence-transformers>=3.0.0",
    "rank-bm25>=0.2.2",
]

[project.optional-dependencies]
dev = ["pytest>=8.0.0"]

[project.scripts]
rag = "app.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["app", "eval"]
```

- [ ] **Step 3: Create empty package init files**

`app/__init__.py`: empty file.
`eval/__init__.py`: empty file.

- [ ] **Step 4: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
data/chroma_db/
results/*.json
```

- [ ] **Step 5: Create `results/.gitkeep`**

Empty file at `results/.gitkeep` so the (otherwise gitignored-contents) directory is tracked.

- [ ] **Step 6: Install uv**

Run (PowerShell):
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
Verify: `uv --version` prints a version string.

- [ ] **Step 7: Sync the environment**

Run: `uv sync --extra dev`
Expected: creates `.venv/` and `uv.lock`, installs all dependencies including `pytest`. No errors.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml app/__init__.py eval/__init__.py .gitignore results/.gitkeep uv.lock
git commit -m "chore: scaffold eval-harness project with uv"
```

---

### Task 2: Sample corpus and golden dataset

**Files:**
- Create: `data/corpus/setup-guide.md`
- Create: `data/corpus/faq.md`
- Create: `data/corpus/pricing.md`
- Create: `data/corpus/support-policy.md`
- Create: `data/corpus/security.md`
- Create: `data/golden_dataset.jsonl`

**Interfaces:**
- Produces: a small fictional knowledge base (product: "Nimbus Notes", a note-taking app) and 7 golden Q&A pairs referencing it, used by every later task's tests and by the regression test.

- [ ] **Step 1: Write the corpus docs**

`data/corpus/setup-guide.md`:
```markdown
# Nimbus Notes Setup Guide

Nimbus Notes is a cross-platform note-taking app. To get started, download the
installer from the Nimbus Notes website and run it on Windows, macOS, or Linux.

After installation, sign in with your Nimbus account email. If you don't have
an account, click "Create Account" on the sign-in screen. New accounts start
on the Free plan automatically.

Nimbus Notes syncs your notes across devices every 60 seconds when you have an
internet connection. You can force an immediate sync by pressing Ctrl+Shift+S
(Cmd+Shift+S on macOS).
```

`data/corpus/faq.md`:
```markdown
# Nimbus Notes FAQ

**Can I use Nimbus Notes offline?**
Yes. Notes you've already opened are cached locally and fully editable offline.
Changes sync automatically the next time you're online.

**How many devices can I sync to?**
Free plan accounts can sync to 2 devices. Pro plan accounts can sync to
unlimited devices.

**Can I export my notes?**
Yes, notes can be exported as Markdown or PDF from the note's "..." menu.

**Does Nimbus Notes support end-to-end encryption?**
Yes, all notes are end-to-end encrypted by default on both Free and Pro plans.
```

`data/corpus/pricing.md`:
```markdown
# Nimbus Notes Pricing

Nimbus Notes offers two plans:

- **Free**: $0/month. Sync to 2 devices, 1 GB storage, end-to-end encryption.
- **Pro**: $6/month or $60/year. Unlimited device sync, 50 GB storage,
  end-to-end encryption, and version history going back 90 days.

There is no trial period required to try Pro features — you can upgrade or
downgrade at any time from Account Settings, and downgrades take effect at
the end of the current billing period.
```

`data/corpus/support-policy.md`:
```markdown
# Nimbus Notes Support Policy

Free plan users get community forum support only, with no guaranteed response
time.

Pro plan users get email support with a target first-response time of 24
hours on business days.

Nimbus Notes does not offer phone support on any plan.
```

`data/corpus/security.md`:
```markdown
# Nimbus Notes Security

All notes are encrypted end-to-end using AES-256, meaning Nimbus Notes
employees cannot read your note content. Encryption keys are derived from
your account password and never leave your device unencrypted.

If you forget your password, notes cannot be recovered, since Nimbus Notes
has no access to your encryption key. We strongly recommend saving a backup
of your recovery phrase, shown once during account creation.
```

- [ ] **Step 2: Write the golden dataset**

`data/golden_dataset.jsonl` (one JSON object per line, no trailing blank line):
```jsonl
{"id": "q1", "question": "How many devices can I sync to on the Free plan?", "expected_answer": "2 devices."}
{"id": "q2", "question": "Does Nimbus Notes have end-to-end encryption on the Free plan?", "expected_answer": "Yes, encryption is on by default on both Free and Pro plans."}
{"id": "q3", "question": "How much does the Pro plan cost per year?", "expected_answer": "$60 per year."}
{"id": "q4", "question": "Can I get phone support as a Pro user?", "expected_answer": "No, Nimbus Notes does not offer phone support on any plan."}
{"id": "q5", "question": "What happens if I forget my password?", "expected_answer": "Notes cannot be recovered because Nimbus Notes has no access to your encryption key; you should use your recovery phrase."}
{"id": "q6", "question": "Can I use Nimbus Notes without an internet connection?", "expected_answer": "Yes, previously opened notes are cached locally and editable offline, syncing once back online."}
{"id": "q7", "question": "What is the maximum weight capacity of a Nimbus Notes desk?", "expected_answer": "I don't know based on the provided context — Nimbus Notes is a note-taking app and the corpus contains no information about desks."}
```

- [ ] **Step 3: Commit**

```bash
git add data/corpus data/golden_dataset.jsonl
git commit -m "data: add sample corpus and golden Q&A dataset"
```

---

### Task 3: Chunk model, ingestion, and BM25/Chroma indexing

**Files:**
- Create: `app/models.py`
- Create: `app/ingest.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Produces:
  - `app.models.Chunk` — dataclass `(id: str, text: str, source: str, chunk_index: int, score: float = 0.0)`.
  - `app.ingest.chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]`
  - `app.ingest.load_corpus(corpus_dir: Path = CORPUS_DIR) -> list[tuple[str, str]]` (source relative-path, full text)
  - `app.ingest.get_embedder() -> SentenceTransformer` (lazy singleton)
  - `app.ingest.ingest(corpus_dir: Path = CORPUS_DIR, chroma_dir: Path = CHROMA_DIR) -> int` (returns number of chunks ingested)
  - Constants: `app.ingest.CORPUS_DIR`, `app.ingest.CHROMA_DIR`, `app.ingest.COLLECTION_NAME`, `app.ingest.BM25_INDEX_PATH` (= `CHROMA_DIR / "bm25_index.pkl"`)
- Consumes: nothing from earlier tasks.

- [ ] **Step 1: Write `app/models.py`**

```python
from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    chunk_index: int
    score: float = 0.0
```

- [ ] **Step 2: Write the failing test for `chunk_text`**

`tests/test_ingest.py`:
```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: FAIL with `ImportError: cannot import name 'chunk_text'` (module doesn't exist yet).

- [ ] **Step 4: Write `app/ingest.py` (chunking + corpus loading only, no indexing yet)**

```python
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
        else:
            current = f"{current}\n\n{para}" if current else para
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: 3 passed.

- [ ] **Step 6: Write the failing test for `ingest`**

Append to `tests/test_ingest.py`:
```python
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
```

- [ ] **Step 7: Run test to verify it fails**

Run: `uv run pytest tests/test_ingest.py::test_ingest_builds_chroma_and_bm25_index -v`
Expected: FAIL with `ImportError: cannot import name 'ingest'`.

- [ ] **Step 8: Add `ingest()` to `app/ingest.py`**

Append to `app/ingest.py`:
```python
import pickle


def ingest(corpus_dir: Path = CORPUS_DIR, chroma_dir: Path = CHROMA_DIR) -> int:
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
```

- [ ] **Step 9: Run test to verify it passes**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: 4 passed. (First run downloads the `all-MiniLM-L6-v2` model; may take a minute.)

- [ ] **Step 10: Commit**

```bash
git add app/models.py app/ingest.py tests/test_ingest.py
git commit -m "feat: add chunk model and ingestion into Chroma + BM25"
```

---

### Task 4: Retrieval (dense, BM25, hybrid)

**Files:**
- Create: `app/retrieve.py`
- Test: `tests/test_retrieve.py`

**Interfaces:**
- Consumes: `app.models.Chunk`; `app.ingest.{CHROMA_DIR, COLLECTION_NAME, get_embedder, ingest}`.
- Produces:
  - `app.retrieve.dense_search(query: str, k: int = 4, chroma_dir: Path = CHROMA_DIR) -> list[Chunk]`
  - `app.retrieve.bm25_search(query: str, k: int = 4, chroma_dir: Path = CHROMA_DIR) -> list[Chunk]`
  - `app.retrieve.reciprocal_rank_fusion(rankings: list[list[Chunk]], k_const: int = 60) -> list[Chunk]`
  - `app.retrieve.retrieve(query: str, mode: Literal["dense","bm25","hybrid"] = "hybrid", k: int = 4, chroma_dir: Path = CHROMA_DIR) -> list[Chunk]`

- [ ] **Step 1: Write the failing tests**

`tests/test_retrieve.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_retrieve.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.retrieve'`.

- [ ] **Step 3: Write `app/retrieve.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_retrieve.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add app/retrieve.py tests/test_retrieve.py
git commit -m "feat: add dense, BM25, and hybrid retrieval"
```

---

### Task 5: Answer generation via Anthropic API

**Files:**
- Create: `app/generate.py`
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: `app.models.Chunk`.
- Produces:
  - `app.generate.DEFAULT_MODEL = "claude-sonnet-5"`
  - `app.generate.build_prompt(question: str, chunks: list[Chunk]) -> str`
  - `app.generate.get_client()` (lazy singleton `anthropic.Anthropic()`)
  - `app.generate.generate_answer(question: str, chunks: list[Chunk], model: str | None = None, client=None) -> str`

- [ ] **Step 1: Write the failing test**

`tests/test_generate.py`:
```python
from app.models import Chunk


class FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class FakeResponse:
    def __init__(self, text):
        self.content = [FakeTextBlock(text)]


class FakeMessages:
    def __init__(self, response_text):
        self._response_text = response_text
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return FakeResponse(self._response_text)


class FakeClient:
    def __init__(self, response_text="fake answer"):
        self.messages = FakeMessages(response_text)


def test_build_prompt_includes_question_and_chunk_text():
    from app.generate import build_prompt

    chunks = [Chunk(id="a::0", text="Paris is the capital of France.", source="a.md", chunk_index=0)]
    prompt = build_prompt("What is the capital of France?", chunks)
    assert "What is the capital of France?" in prompt
    assert "Paris is the capital of France." in prompt
    assert "a.md" in prompt


def test_generate_answer_uses_client_and_returns_text():
    from app.generate import generate_answer

    chunks = [Chunk(id="a::0", text="Paris is the capital of France.", source="a.md", chunk_index=0)]
    client = FakeClient(response_text="Paris.")

    answer = generate_answer("What is the capital of France?", chunks, client=client)

    assert answer == "Paris."
    assert client.messages.last_kwargs["model"] == "claude-sonnet-5"
    assert "system" in client.messages.last_kwargs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.generate'`.

- [ ] **Step 3: Write `app/generate.py`**

```python
import os

from app.models import Chunk

DEFAULT_MODEL = "claude-sonnet-5"
SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions using ONLY the provided context. "
    "If the context does not contain the answer, say \"I don't know based on the provided "
    "context.\" Do not use outside knowledge."
)

_client = None


def get_client():
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic()
    return _client


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    context = "\n\n".join(f"[{c.source}]\n{c.text}" for c in chunks)
    return f"Context:\n{context}\n\nQuestion: {question}"


def generate_answer(
    question: str,
    chunks: list[Chunk],
    model: str | None = None,
    client=None,
) -> str:
    client = client or get_client()
    model = model or os.environ.get("EVAL_HARNESS_MODEL", DEFAULT_MODEL)
    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(question, chunks)}],
    )
    return response.content[0].text
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_generate.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add app/generate.py tests/test_generate.py
git commit -m "feat: add Anthropic-backed answer generation"
```

---

### Task 6: CLI

**Files:**
- Create: `app/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `app.ingest.ingest`, `app.retrieve.retrieve`, `app.generate.generate_answer`.
- Produces: `app.cli.main(argv: list[str] | None = None) -> int`, registered as the `rag` console script (already wired in `pyproject.toml` Task 1).

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:
```python
from app.models import Chunk


def test_cli_ingest_command(monkeypatch, capsys):
    import app.cli as cli

    monkeypatch.setattr(cli, "ingest", lambda: 7)

    exit_code = cli.main(["ingest"])

    assert exit_code == 0
    assert "7" in capsys.readouterr().out


def test_cli_ask_command(monkeypatch, capsys):
    import app.cli as cli

    chunk = Chunk(id="a::0", text="Paris is the capital of France.", source="a.md", chunk_index=0)
    monkeypatch.setattr(cli, "retrieve", lambda query, mode, k: [chunk])
    monkeypatch.setattr(cli, "generate_answer", lambda question, chunks: "Paris.")

    exit_code = cli.main(["ask", "What is the capital of France?", "--mode", "dense", "--k", "1"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Paris." in out
    assert "a.md" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.cli'`.

- [ ] **Step 3: Write `app/cli.py`**

```python
import argparse

from app.generate import generate_answer
from app.ingest import ingest
from app.retrieve import retrieve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ingest")

    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid")
    ask_parser.add_argument("--k", type=int, default=4)

    args = parser.parse_args(argv)

    if args.command == "ingest":
        count = ingest()
        print(f"Ingested {count} chunks.")
        return 0

    if args.command == "ask":
        chunks = retrieve(args.question, mode=args.mode, k=args.k)
        answer = generate_answer(args.question, chunks)
        print(answer)
        print("\nSources:")
        for c in chunks:
            print(f"  - {c.source} (chunk {c.chunk_index})")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: 2 passed.

- [ ] **Step 5: Verify the console script is wired up**

Run: `uv sync --extra dev` then `uv run rag --help`
Expected: prints usage with `ingest` and `ask` subcommands.

- [ ] **Step 6: Commit**

```bash
git add app/cli.py tests/test_cli.py
git commit -m "feat: add rag CLI (ingest, ask)"
```

---

### Task 7: Golden dataset loader

**Files:**
- Create: `eval/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Produces:
  - `eval.dataset.GoldenItem` — dataclass `(id: str, question: str, expected_answer: str, expected_source_ids: list[str] | None = None)`
  - `eval.dataset.DEFAULT_DATASET_PATH = Path("data/golden_dataset.jsonl")`
  - `eval.dataset.load_golden_dataset(path: Path = DEFAULT_DATASET_PATH) -> list[GoldenItem]`

- [ ] **Step 1: Write the failing test**

`tests/test_dataset.py`:
```python
def test_load_golden_dataset_parses_jsonl(tmp_path):
    from eval.dataset import load_golden_dataset, GoldenItem

    path = tmp_path / "golden.jsonl"
    path.write_text(
        '{"id": "q1", "question": "Q1?", "expected_answer": "A1"}\n'
        '{"id": "q2", "question": "Q2?", "expected_answer": "A2", "expected_source_ids": ["doc.md::0"]}\n',
        encoding="utf-8",
    )

    items = load_golden_dataset(path)

    assert items == [
        GoldenItem(id="q1", question="Q1?", expected_answer="A1", expected_source_ids=None),
        GoldenItem(id="q2", question="Q2?", expected_answer="A2", expected_source_ids=["doc.md::0"]),
    ]


def test_load_golden_dataset_default_path_loads_real_dataset():
    from eval.dataset import load_golden_dataset

    items = load_golden_dataset()
    assert len(items) == 7
    assert items[0].id == "q1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.dataset'`.

- [ ] **Step 3: Write `eval/dataset.py`**

```python
import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATASET_PATH = Path("data/golden_dataset.jsonl")


@dataclass
class GoldenItem:
    id: str
    question: str
    expected_answer: str
    expected_source_ids: list[str] | None = None


def load_golden_dataset(path: Path = DEFAULT_DATASET_PATH) -> list[GoldenItem]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            items.append(
                GoldenItem(
                    id=data["id"],
                    question=data["question"],
                    expected_answer=data["expected_answer"],
                    expected_source_ids=data.get("expected_source_ids"),
                )
            )
    return items
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add eval/dataset.py tests/test_dataset.py
git commit -m "feat: add golden dataset loader"
```

---

### Task 8: LLM-as-judge

**Files:**
- Create: `eval/judge.py`
- Test: `tests/test_judge.py`

**Interfaces:**
- Consumes: `app.models.Chunk`.
- Produces:
  - `eval.judge.JudgeResult` — dataclass `(faithfulness: int, relevance: int, correctness: bool, reasoning: str)`
  - `eval.judge.judge_answer(question: str, context_chunks: list[Chunk], generated_answer: str, expected_answer: str, model: str | None = None, client=None) -> JudgeResult`

- [ ] **Step 1: Write the failing test**

`tests/test_judge.py`:
```python
from app.models import Chunk


class FakeToolUseBlock:
    def __init__(self, input_data):
        self.type = "tool_use"
        self.input = input_data


class FakeJudgeResponse:
    def __init__(self, input_data):
        self.content = [FakeToolUseBlock(input_data)]


class FakeMessages:
    def __init__(self, input_data):
        self._input_data = input_data
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return FakeJudgeResponse(self._input_data)


class FakeClient:
    def __init__(self, input_data):
        self.messages = FakeMessages(input_data)


def test_judge_answer_parses_tool_use_response():
    from eval.judge import judge_answer, JudgeResult

    chunk = Chunk(id="a::0", text="Paris is the capital of France.", source="a.md", chunk_index=0)
    client = FakeClient(
        {"faithfulness": 5, "relevance": 5, "correctness": True, "reasoning": "Matches context."}
    )

    result = judge_answer(
        question="What is the capital of France?",
        context_chunks=[chunk],
        generated_answer="Paris.",
        expected_answer="Paris",
        client=client,
    )

    assert result == JudgeResult(faithfulness=5, relevance=5, correctness=True, reasoning="Matches context.")
    assert client.messages.last_kwargs["tool_choice"] == {"type": "tool", "name": "submit_judgment"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_judge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.judge'`.

- [ ] **Step 3: Write `eval/judge.py`**

```python
import os
from dataclasses import dataclass

from app.models import Chunk

DEFAULT_MODEL = "claude-sonnet-5"

JUDGE_TOOL = {
    "name": "submit_judgment",
    "description": "Submit a structured judgment of a generated answer's quality.",
    "input_schema": {
        "type": "object",
        "properties": {
            "faithfulness": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "1-5: is the answer fully supported by the retrieved context?",
            },
            "relevance": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "1-5: does the answer address the question asked?",
            },
            "correctness": {
                "type": "boolean",
                "description": "Does the answer match the expected answer in substance?",
            },
            "reasoning": {"type": "string", "description": "Brief justification for the scores."},
        },
        "required": ["faithfulness", "relevance", "correctness", "reasoning"],
    },
}


@dataclass
class JudgeResult:
    faithfulness: int
    relevance: int
    correctness: bool
    reasoning: str


_client = None


def get_client():
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic()
    return _client


def judge_answer(
    question: str,
    context_chunks: list[Chunk],
    generated_answer: str,
    expected_answer: str,
    model: str | None = None,
    client=None,
) -> JudgeResult:
    client = client or get_client()
    model = model or os.environ.get("EVAL_HARNESS_MODEL", DEFAULT_MODEL)
    context = "\n\n".join(f"[{c.source}]\n{c.text}" for c in context_chunks)
    prompt = (
        f"Question: {question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"Generated answer: {generated_answer}\n\n"
        f"Expected answer: {expected_answer}\n\n"
        "Score the generated answer using the submit_judgment tool."
    )
    response = client.messages.create(
        model=model,
        max_tokens=512,
        tools=[JUDGE_TOOL],
        tool_choice={"type": "tool", "name": "submit_judgment"},
        messages=[{"role": "user", "content": prompt}],
    )
    for block in response.content:
        if block.type == "tool_use":
            data = block.input
            return JudgeResult(
                faithfulness=data["faithfulness"],
                relevance=data["relevance"],
                correctness=data["correctness"],
                reasoning=data["reasoning"],
            )
    raise ValueError("No tool_use block in judge response")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_judge.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add eval/judge.py tests/test_judge.py
git commit -m "feat: add LLM-as-judge scoring"
```

---

### Task 9: Metrics and CI thresholds

**Files:**
- Create: `eval/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `eval.judge.JudgeResult`.
- Produces:
  - Constants: `eval.metrics.MIN_FAITHFULNESS = 4.0`, `eval.metrics.MIN_CORRECTNESS_RATE = 0.7`, `eval.metrics.MAX_HALLUCINATION_RATE = 0.2`, `eval.metrics.HALLUCINATION_THRESHOLD = 2`
  - `eval.metrics.EvalSummary` — dataclass `(mean_faithfulness: float, mean_relevance: float, correctness_rate: float, hallucination_rate: float, n_items: int)`
  - `eval.metrics.compute_summary(results: list[JudgeResult]) -> EvalSummary`
  - `eval.metrics.check_thresholds(summary: EvalSummary) -> list[str]` (empty list = pass)

- [ ] **Step 1: Write the failing test**

`tests/test_metrics.py`:
```python
from eval.judge import JudgeResult


def test_compute_summary_averages_and_rates():
    from eval.metrics import compute_summary

    results = [
        JudgeResult(faithfulness=5, relevance=5, correctness=True, reasoning=""),
        JudgeResult(faithfulness=1, relevance=3, correctness=False, reasoning=""),
    ]

    summary = compute_summary(results)

    assert summary.mean_faithfulness == 3.0
    assert summary.mean_relevance == 4.0
    assert summary.correctness_rate == 0.5
    assert summary.hallucination_rate == 0.5
    assert summary.n_items == 2


def test_compute_summary_empty_results():
    from eval.metrics import compute_summary

    summary = compute_summary([])
    assert summary.n_items == 0
    assert summary.mean_faithfulness == 0.0


def test_check_thresholds_passes_when_within_bounds():
    from eval.metrics import check_thresholds, EvalSummary

    summary = EvalSummary(
        mean_faithfulness=4.5, mean_relevance=4.5, correctness_rate=0.9, hallucination_rate=0.0, n_items=5
    )
    assert check_thresholds(summary) == []


def test_check_thresholds_reports_violations():
    from eval.metrics import check_thresholds, EvalSummary

    summary = EvalSummary(
        mean_faithfulness=2.0, mean_relevance=4.0, correctness_rate=0.3, hallucination_rate=0.5, n_items=5
    )
    violations = check_thresholds(summary)
    assert len(violations) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.metrics'`.

- [ ] **Step 3: Write `eval/metrics.py`**

```python
from dataclasses import dataclass

from eval.judge import JudgeResult

MIN_FAITHFULNESS = 4.0
MIN_CORRECTNESS_RATE = 0.7
MAX_HALLUCINATION_RATE = 0.2
HALLUCINATION_THRESHOLD = 2


@dataclass
class EvalSummary:
    mean_faithfulness: float
    mean_relevance: float
    correctness_rate: float
    hallucination_rate: float
    n_items: int


def compute_summary(results: list[JudgeResult]) -> EvalSummary:
    n = len(results)
    if n == 0:
        return EvalSummary(0.0, 0.0, 0.0, 0.0, 0)
    mean_faithfulness = sum(r.faithfulness for r in results) / n
    mean_relevance = sum(r.relevance for r in results) / n
    correctness_rate = sum(1 for r in results if r.correctness) / n
    hallucination_rate = sum(1 for r in results if r.faithfulness <= HALLUCINATION_THRESHOLD) / n
    return EvalSummary(mean_faithfulness, mean_relevance, correctness_rate, hallucination_rate, n)


def check_thresholds(summary: EvalSummary) -> list[str]:
    violations = []
    if summary.mean_faithfulness < MIN_FAITHFULNESS:
        violations.append(f"mean_faithfulness {summary.mean_faithfulness:.2f} < {MIN_FAITHFULNESS}")
    if summary.correctness_rate < MIN_CORRECTNESS_RATE:
        violations.append(f"correctness_rate {summary.correctness_rate:.2f} < {MIN_CORRECTNESS_RATE}")
    if summary.hallucination_rate > MAX_HALLUCINATION_RATE:
        violations.append(f"hallucination_rate {summary.hallucination_rate:.2f} > {MAX_HALLUCINATION_RATE}")
    return violations
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add eval/metrics.py tests/test_metrics.py
git commit -m "feat: add eval metrics aggregation and CI thresholds"
```

---

### Task 10: Eval orchestration (`run_eval`)

**Files:**
- Create: `eval/run_eval.py`
- Test: `tests/test_run_eval.py`

**Interfaces:**
- Consumes: `app.retrieve.retrieve`, `app.generate.generate_answer`, `eval.dataset.{load_golden_dataset, DEFAULT_DATASET_PATH, GoldenItem}`, `eval.judge.{judge_answer, JudgeResult}`, `eval.metrics.{compute_summary, check_thresholds, EvalSummary}`.
- Produces:
  - `eval.run_eval.RESULTS_DIR = Path("results")`
  - `eval.run_eval.run_eval(golden_path: Path = DEFAULT_DATASET_PATH, mode: str = "hybrid", k: int = 4) -> tuple[EvalSummary, list[dict]]`
  - `eval.run_eval.save_results(summary: EvalSummary, per_item: list[dict], timestamp: str, results_dir: Path = RESULTS_DIR) -> Path`
  - `eval.run_eval.print_summary(summary: EvalSummary) -> None`
  - `eval.run_eval.main(argv: list[str] | None = None) -> int` (0 = thresholds pass, 1 = violated)

- [ ] **Step 1: Write the failing test**

`tests/test_run_eval.py`:
```python
import json

from app.models import Chunk
from eval.dataset import GoldenItem
from eval.judge import JudgeResult


def test_run_eval_orchestrates_retrieve_generate_judge(monkeypatch):
    import eval.run_eval as run_eval_module

    items = [
        GoldenItem(id="q1", question="Q1?", expected_answer="A1"),
        GoldenItem(id="q2", question="Q2?", expected_answer="A2"),
    ]
    chunk = Chunk(id="a::0", text="text", source="a.md", chunk_index=0)

    monkeypatch.setattr(run_eval_module, "load_golden_dataset", lambda path: items)
    monkeypatch.setattr(run_eval_module, "retrieve", lambda query, mode, k: [chunk])
    monkeypatch.setattr(run_eval_module, "generate_answer", lambda question, chunks: f"answer to {question}")
    monkeypatch.setattr(
        run_eval_module,
        "judge_answer",
        lambda question, context_chunks, generated_answer, expected_answer: JudgeResult(
            faithfulness=5, relevance=5, correctness=True, reasoning="ok"
        ),
    )

    summary, per_item = run_eval_module.run_eval()

    assert summary.n_items == 2
    assert summary.mean_faithfulness == 5.0
    assert len(per_item) == 2
    assert per_item[0]["id"] == "q1"
    assert per_item[0]["generated_answer"] == "answer to Q1?"
    assert per_item[0]["sources"] == ["a.md"]


def test_save_results_writes_json(tmp_path):
    from eval.metrics import EvalSummary
    from eval.run_eval import save_results

    summary = EvalSummary(
        mean_faithfulness=5.0, mean_relevance=5.0, correctness_rate=1.0, hallucination_rate=0.0, n_items=1
    )
    per_item = [{"id": "q1", "generated_answer": "A1"}]

    path = save_results(summary, per_item, timestamp="20260101T000000Z", results_dir=tmp_path)

    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["summary"]["n_items"] == 1
    assert data["items"][0]["id"] == "q1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_run_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.run_eval'`.

- [ ] **Step 3: Write `eval/run_eval.py`**

```python
import json
import time
from dataclasses import asdict
from pathlib import Path

from app.generate import generate_answer
from app.retrieve import retrieve
from eval.dataset import DEFAULT_DATASET_PATH, load_golden_dataset
from eval.judge import JudgeResult, judge_answer
from eval.metrics import EvalSummary, check_thresholds, compute_summary

RESULTS_DIR = Path("results")


def run_eval(
    golden_path: Path = DEFAULT_DATASET_PATH, mode: str = "hybrid", k: int = 4
) -> tuple[EvalSummary, list[dict]]:
    items = load_golden_dataset(golden_path)
    judge_results: list[JudgeResult] = []
    per_item = []
    for item in items:
        chunks = retrieve(item.question, mode=mode, k=k)
        answer = generate_answer(item.question, chunks)
        verdict = judge_answer(item.question, chunks, answer, item.expected_answer)
        judge_results.append(verdict)
        per_item.append(
            {
                "id": item.id,
                "question": item.question,
                "generated_answer": answer,
                "expected_answer": item.expected_answer,
                "sources": [c.source for c in chunks],
                **asdict(verdict),
            }
        )
    summary = compute_summary(judge_results)
    return summary, per_item


def save_results(
    summary: EvalSummary, per_item: list[dict], timestamp: str, results_dir: Path = RESULTS_DIR
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"eval_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"summary": asdict(summary), "items": per_item}, f, indent=2)
    return path


def print_summary(summary: EvalSummary) -> None:
    print(f"Items evaluated:     {summary.n_items}")
    print(f"Mean faithfulness:   {summary.mean_faithfulness:.2f}")
    print(f"Mean relevance:      {summary.mean_relevance:.2f}")
    print(f"Correctness rate:    {summary.correctness_rate:.2%}")
    print(f"Hallucination rate:  {summary.hallucination_rate:.2%}")


def main(argv: list[str] | None = None) -> int:
    summary, per_item = run_eval()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = save_results(summary, per_item, timestamp)
    print_summary(summary)
    print(f"\nResults written to {path}")
    violations = check_thresholds(summary)
    if violations:
        print("\nTHRESHOLD VIOLATIONS:")
        for v in violations:
            print(f"  - {v}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_run_eval.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add eval/run_eval.py tests/test_run_eval.py
git commit -m "feat: add run_eval orchestration and results writer"
```

---

### Task 11: Regression test (CI gate)

**Files:**
- Create: `tests/test_regression.py`

**Interfaces:**
- Consumes: `eval.run_eval.run_eval`, `eval.metrics.check_thresholds`.

- [ ] **Step 1: Write `tests/test_regression.py`**

```python
import os

import pytest

from eval.metrics import check_thresholds
from eval.run_eval import run_eval

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)


def test_eval_meets_thresholds():
    summary, _ = run_eval()
    violations = check_thresholds(summary)
    assert not violations, f"Threshold violations: {violations}"
```

- [ ] **Step 2: Run it without an API key to confirm the skip path**

Run: `uv run pytest tests/test_regression.py -v`
Expected: 1 skipped, reason "ANTHROPIC_API_KEY not set".

- [ ] **Step 3: If `ANTHROPIC_API_KEY` is available in this environment, run it for real**

Run (only if you have a key): `ANTHROPIC_API_KEY=... uv run rag ingest && ANTHROPIC_API_KEY=... uv run pytest tests/test_regression.py -v`
Expected: 1 passed (or a clear assertion listing which threshold(s) failed, which is useful signal on the sample dataset — adjust `eval/metrics.py` thresholds or the golden dataset if the toy corpus genuinely can't clear them).

- [ ] **Step 4: Commit**

```bash
git add tests/test_regression.py
git commit -m "test: add CI regression gate on eval thresholds"
```

---

### Task 12: CI workflow

**Files:**
- Create: `.github/workflows/eval-ci.yml`

- [ ] **Step 1: Write the workflow**

`.github/workflows/eval-ci.yml`:
```yaml
name: eval-ci

on:
  push:
  pull_request:

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3

      - name: Set up Python
        run: uv python install 3.12

      - name: Sync dependencies
        run: uv sync --extra dev

      - name: Ingest corpus
        run: uv run rag ingest

      - name: Run regression tests
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: uv run pytest tests/ -v
```

- [ ] **Step 2: Validate YAML syntax**

Run: `python -c "import yaml, sys; yaml.safe_load(open('.github/workflows/eval-ci.yml'))" ` (uses any Python with PyYAML; if unavailable, visually confirm indentation instead)
Expected: no exception.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/eval-ci.yml
git commit -m "ci: run eval regression tests on push and pull_request"
```

---

### Task 13: README, requirements.txt export, and end-to-end smoke test

**Files:**
- Create: `README.md`
- Create: `requirements.txt`

**Interfaces:**
- None (documentation + export task).

- [ ] **Step 1: Export `requirements.txt` from the uv lockfile**

Run: `uv export --no-hashes --no-dev -o requirements.txt`
Expected: `requirements.txt` created, listing `anthropic`, `chromadb`, `sentence-transformers`, `rank-bm25`, and their transitive pins.

- [ ] **Step 2: Write `README.md`**

```markdown
# eval-harness

A toy RAG app (`app/`) plus an LLM-as-judge eval harness (`eval/`) that scores
it for faithfulness, relevance, and correctness, and gates CI on the result.

## Setup

```bash
# Install uv if you don't have it: https://docs.astral.sh/uv/getting-started/installation/
uv sync --extra dev
export ANTHROPIC_API_KEY=sk-ant-...
```

## Using the RAG app

```bash
# Chunk, embed, and index data/corpus/ into Chroma + BM25
uv run rag ingest

# Ask a question (mode: dense | bm25 | hybrid, default hybrid)
uv run rag ask "How many devices can I sync to on the Free plan?" --mode hybrid --k 4
```

Drop your own `.md`/`.txt` files into `data/corpus/` and re-run `uv run rag ingest`
to index them instead of the bundled sample docs.

## Running the eval

```bash
uv run rag ingest
uv run python -m eval.run_eval
```

This runs every question in `data/golden_dataset.jsonl` through
retrieve → generate → LLM-judge, prints a summary, and writes a timestamped
results file to `results/`. Exit code is non-zero if any threshold in
`eval/metrics.py` is violated.

Golden dataset format (`data/golden_dataset.jsonl`, one JSON object per line):
```json
{"id": "q1", "question": "...", "expected_answer": "...", "expected_source_ids": ["optional.md::0"]}
```

## Tests

```bash
uv run pytest tests/ -v
```

`tests/test_regression.py` runs a full eval pass against the real Anthropic
API and asserts the summary clears the thresholds in `eval/metrics.py`. It
skips automatically if `ANTHROPIC_API_KEY` is not set. All other tests mock
the Anthropic client and don't require a key.

## CI

`.github/workflows/eval-ci.yml` runs on every push/PR: installs `uv`, syncs
deps, ingests the corpus, and runs the full test suite (including the
regression gate) using an `ANTHROPIC_API_KEY` repository secret.

## Design

See `docs/superpowers/specs/2026-09-01-eval-harness-design.md` for the full
design rationale.
```

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest tests/ -v`
Expected: all tests pass or skip (only `test_regression.py` may skip if no API key is set locally).

- [ ] **Step 4: End-to-end smoke test (manual, requires `ANTHROPIC_API_KEY`)**

```bash
uv run rag ingest
uv run rag ask "How many devices can I sync to on the Free plan?"
uv run python -m eval.run_eval
```
Expected: `ingest` reports a chunk count; `ask` prints an answer plus source list; `run_eval` prints a summary and writes `results/eval_<timestamp>.json`, exiting 0 if thresholds are met.

- [ ] **Step 5: Commit**

```bash
git add README.md requirements.txt
git commit -m "docs: add README and export requirements.txt"
```
