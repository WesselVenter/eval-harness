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
