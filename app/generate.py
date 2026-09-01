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
