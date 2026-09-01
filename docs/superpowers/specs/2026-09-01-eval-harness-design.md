# Eval Harness — Design Spec

Date: 2026-09-01

## Purpose

Build a toy RAG application (`app/`) as a system-under-test, plus an eval harness (`eval/`) that scores it with an LLM-as-judge and gates CI on quality thresholds. The RAG app is deliberately minimal — the eval harness is the point.

## Project structure

```
eval-harness/
├── app/
│   ├── ingest.py
│   ├── retrieve.py
│   ├── generate.py
│   └── cli.py
├── eval/
│   ├── dataset.py
│   ├── judge.py
│   ├── metrics.py
│   └── run_eval.py
├── tests/
│   └── test_regression.py
├── data/
│   ├── golden_dataset.jsonl
│   └── corpus/
├── results/
├── .github/workflows/eval-ci.yml
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Environment management

`uv`, with `pyproject.toml` as the source of truth (+ `uv.lock`). A `requirements.txt` is also generated (via `uv export` / `uv pip compile`) for tooling that expects it. Python 3.12 target (matches the interpreter available on this machine; 3.11+ required).

## App (system under test)

### Ingest (`app/ingest.py`)
- Walks `data/corpus/**/*.{md,txt}`.
- Chunks text: paragraph-aware, ~500 chars per chunk with 50 chars overlap.
- Embeds each chunk locally with `sentence-transformers` (`all-MiniLM-L6-v2`).
- Upserts chunks (id, text, metadata: source path, chunk index) into a persistent ChromaDB collection at `data/chroma_db/`.
- Also builds a `rank_bm25.BM25Okapi` index over the same chunk texts/ids and pickles it to `data/chroma_db/bm25_index.pkl`, so BM25/hybrid retrieval don't need Chroma's embeddings.
- Idempotent: re-running clears and rebuilds the collection/index from the current corpus.

### Retrieve (`app/retrieve.py`)
- `retrieve(query: str, mode: Literal["dense","bm25","hybrid"] = "hybrid", k: int = 4) -> list[Chunk]`
- `dense`: Chroma similarity search on the embedded query.
- `bm25`: BM25Okapi score over tokenized query against the pickled index.
- `hybrid`: reciprocal rank fusion (RRF, k=60) combining the dense and bm25 rankings.
- Returns chunk text + source metadata + score.

### Generate (`app/generate.py`)
- Builds a grounded-answer prompt: system instructions ("answer only from the provided context; say you don't know if the context doesn't cover it") + retrieved chunks + question.
- Calls the Anthropic Messages API (`claude-sonnet-5` by default, model configurable via env var `EVAL_HARNESS_MODEL`).
- Returns `{answer: str, chunks_used: list[Chunk]}`.

### CLI (`app/cli.py`)
- `uv run rag ingest` — runs ingestion over `data/corpus/`.
- `uv run rag ask "<question>" [--mode dense|bm25|hybrid] [--k N]` — retrieves + generates, prints the answer and the chunks used.
- Registered as a `[project.scripts]` entry point (`rag = "app.cli:main"`).

## Eval harness

### Dataset (`eval/dataset.py`)
- Loads `data/golden_dataset.jsonl`, one JSON object per line: `{"id", "question", "expected_answer", "expected_source_ids"?}`.
- Returns a list of typed records (dataclass).

### Judge (`eval/judge.py`)
- For each item: sends the question, retrieved context, generated answer, and expected answer to Claude with a scoring rubric.
- Forces structured JSON output (via a tool-call or strict JSON instruction) with fields:
  - `faithfulness` (1-5): is the generated answer fully supported by the retrieved context?
  - `relevance` (1-5): does the answer address the question?
  - `correctness` (bool): does the answer match the expected_answer in substance?
  - `reasoning` (str): brief justification.
- Uses the same model as generation by default (configurable).

### Metrics (`eval/metrics.py`)
- Aggregates a list of per-item judge results into a summary:
  - `mean_faithfulness`, `mean_relevance`, `correctness_rate`, `hallucination_rate` (fraction with faithfulness ≤ 2).
- Defines the CI gate thresholds as module-level constants: `MIN_FAITHFULNESS = 4.0`, `MIN_CORRECTNESS_RATE = 0.7`, `MAX_HALLUCINATION_RATE = 0.2`.

### Run eval (`eval/run_eval.py`)
- CLI entry point (`uv run python -m eval.run_eval`).
- For each golden item: retrieve → generate → judge → collect.
- Writes `results/eval_<UTC timestamp>.json` containing per-item results + the aggregate summary.
- Prints a human-readable summary table to stdout.
- Exits non-zero if any threshold from `eval/metrics.py` is violated (so it can be used directly as a CI gate, independent of pytest).

## Tests (`tests/test_regression.py`)
- Runs (or reuses a freshly-produced) eval result set and asserts the aggregate metrics meet the thresholds in `eval/metrics.py`.
- Skips (not fails) if `ANTHROPIC_API_KEY` is unset, so local dev without a key doesn't break.

## CI (`.github/workflows/eval-ci.yml`)
- Trigger: push / pull_request.
- Steps: checkout → install `uv` → `uv sync` → `uv run rag ingest` → `uv run pytest tests/test_regression.py`.
- `ANTHROPIC_API_KEY` sourced from `secrets.ANTHROPIC_API_KEY`; job fails if the secret is missing (no silent skip in CI) or if thresholds aren't met.

## Sample data
- `data/corpus/`: ~5 short markdown docs forming a small fictional internal-docs knowledge base (e.g. a product's setup guide, FAQ, pricing/policy notes) — enough topical variety to make retrieval and hallucination checks meaningful.
- `data/golden_dataset.jsonl`: 6-8 Q&A pairs derived from that corpus, including at least one question the corpus does *not* answer (to test that the app says "I don't know" rather than hallucinating).

## Out of scope
- No hosted vector DB, no reranking model, no multi-turn conversation, no auth/UI. Single-shot Q&A only.
