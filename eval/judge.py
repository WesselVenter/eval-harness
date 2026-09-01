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
        thinking={"type": "disabled"},
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
