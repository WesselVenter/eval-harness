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
API and gates on two thresholds, calibrated from real eval runs against the
current corpus/dataset rather than picked arbitrarily:

- **Hallucination rate ≤ 15%** overall (observed baseline: 0%).
- **Per-category correctness floors**, set ~0.15-0.2 below each category's
  observed baseline (see the comment in `tests/test_regression.py` for the
  full table). A flat 0.7 floor across categories was tried first and
  rejected — it failed on every run because `factual`/`multi-hop` retrieval
  is a known weak spot, not a regression, so a single global number wasn't a
  useful gate.

On failure, the test prints exactly which question(s) and metric tripped the
threshold (id, score, question text), not just "test failed". It skips
automatically if `ANTHROPIC_API_KEY` is not set. All other tests mock the
Anthropic client and don't require a key.

## CI

`.github/workflows/eval-ci.yml` runs on every push/PR: installs `uv`, syncs
deps, ingests the corpus, verifies the `ANTHROPIC_API_KEY` repository secret
is set (fails fast with a clear error if it's missing), and runs the full
test suite including the regression gate.

The gate has been verified to fail loudly: corrupting a corpus file with
wrong information trips `test_category_correctness_within_threshold` with
the specific questions and scores that regressed, and recovers to green once
reverted.

## Findings

Running the eval suite against the current corpus/dataset produced a 0%
hallucination rate, but correctness varied sharply by category — `factual`
questions scored lowest (~0.41), well below `multi-hop` (~0.56), `edge_case`
(~0.80), and `versioned` (~0.95). That was surprising going in, since factual
lookups should be the easiest case to get right.

Comparing expected_source_ids against actually-retrieved chunks per question
showed the cause wasn't wrong-file retrieval — it was **within-file chunk
ranking**. `dependencies.md` covers several distinct sub-topics (basic
dependency injection, sub-dependencies, and test-time overrides via
`dependency_overrides`). On narrow questions about one specific sub-topic
(e.g. "can a dependency depend on another dependency?"), the top-k retrieved
chunks were correctly pulled from the right file but missed the specific
passage needed, so the app answered "I don't know" despite the answer
existing in the corpus. Shorter, single-topic files like
`background_tasks.md` had no such problem and scored a perfect 1.0.

This is why the regression gate (see Tests, above) uses per-category
correctness floors instead of a single global threshold — `factual` and
`multi-hop` are a known, explained weak spot tied to chunk granularity in
dense source files, not an arbitrary number to chase to 1.0.

**Next step, if I continue tuning:** reduce chunk size (or add overlap)
specifically for longer source files, and re-run the eval to check whether
`factual`/`multi-hop` correctness improves without regressing the categories
that already score well.

## Design

See `docs/superpowers/specs/2026-09-01-eval-harness-design.md` for the full
design rationale.
