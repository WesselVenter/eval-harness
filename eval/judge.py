import os

from pydantic import BaseModel, Field

from app.models import Chunk

DEFAULT_MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """You are grading a RAG (retrieval-augmented generation) system's answer to a \
user question. Score three dimensions, each from 0.0 to 1.0, and give a short justification for \
each. Use the full range, not just the extremes, when an answer is partially right.

faithfulness — is every claim in the answer grounded in the retrieved context?
  1.0: every claim traces back to the retrieved text; no outside knowledge is introduced.
  0.5: partially grounded — some claims are supported, others are not traceable to the context.
  0.0: the answer includes claims absent from the retrieved context (hallucination), or \
contradicts it.
  An answer that correctly says it doesn't know, because the context lacks the information, is \
faithful (score high) — faithfulness measures groundedness, not completeness.

correctness — does the answer convey the same meaning as the expected answer, even if worded \
differently?
  1.0: matches the expected answer's meaning; wording and length may differ freely.
  0.5: partially correct — captures some but not all of the expected meaning, or is imprecise on \
a material point.
  0.0: wrong, or contradicts the expected answer.

retrieval_quality — do the retrieved sources overlap with the sources the question is actually \
answerable from?
  For a normal question (expected_sources is non-empty): 1.0 if the retrieved sources fully cover \
the expected sources, 0.0 if none of them were retrieved, with partial credit for partial overlap.
  For an "out_of_scope" question (expected_sources is empty), the question has no answer in the \
corpus, so score retrieval_quality on whether the system correctly found nothing relevant: 1.0 if \
the retrieved chunks are irrelevant to the question (correctly finding no match), 0.0 if the \
system retrieved and used unrelated context as though it answered the question. Do not penalize \
an empty or irrelevant retrieval set for an out_of_scope question — that is the correct behavior.

For "edge_case" and "multi-hop" questions, note explicitly in the faithfulness or correctness \
justification if the answer only partially addresses the question (e.g. it answers one hop of a \
multi-hop question but not the other, or misses part of an edge-case nuance) — even if the score \
is still high."""


class ScoredAspect(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="Score from 0.0 (worst) to 1.0 (best).")
    justification: str = Field(description="Short justification for the score.")


class JudgeVerdict(BaseModel):
    question: str
    faithfulness: ScoredAspect
    correctness: ScoredAspect
    retrieval_quality: ScoredAspect


_client = None


def get_client():
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic()
    return _client


def build_prompt(
    question: str,
    context_chunks: list[Chunk],
    generated_answer: str,
    expected_answer: str,
    expected_sources: list[str],
    category: str,
) -> str:
    context = "\n\n".join(f"[{c.source}]\n{c.text}" for c in context_chunks)
    retrieved_sources = [c.source for c in context_chunks]
    expected_sources_text = (
        ", ".join(expected_sources)
        if expected_sources
        else "(none — this is an out_of_scope question with no answer in the corpus)"
    )
    return (
        f"Category: {category}\n\n"
        f"Question: {question}\n\n"
        f"Retrieved context:\n{context if context else '(nothing retrieved)'}\n\n"
        f"Retrieved sources: {retrieved_sources}\n\n"
        f"Expected sources: {expected_sources_text}\n\n"
        f"Generated answer: {generated_answer}\n\n"
        f"Expected answer: {expected_answer}\n\n"
        "Score faithfulness, correctness, and retrieval_quality per the rubric in the system prompt."
    )


def judge_answer(
    question: str,
    context_chunks: list[Chunk],
    generated_answer: str,
    expected_answer: str,
    expected_sources: list[str] | None = None,
    category: str = "factual",
    model: str | None = None,
    client=None,
) -> JudgeVerdict:
    client = client or get_client()
    model = model or os.environ.get("EVAL_HARNESS_MODEL", DEFAULT_MODEL)
    expected_sources = expected_sources or []
    prompt = build_prompt(
        question, context_chunks, generated_answer, expected_answer, expected_sources, category
    )
    response = client.messages.parse(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        output_format=JudgeVerdict,
        thinking={"type": "disabled"},
    )
    verdict = response.parsed_output
    verdict.question = question
    return verdict
