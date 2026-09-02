from app.models import Chunk


class FakeParsedResponse:
    def __init__(self, parsed_output):
        self.parsed_output = parsed_output


class FakeMessages:
    def __init__(self, verdict):
        self._verdict = verdict
        self.last_kwargs = None

    def parse(self, **kwargs):
        self.last_kwargs = kwargs
        return FakeParsedResponse(self._verdict)


class FakeClient:
    def __init__(self, verdict):
        self.messages = FakeMessages(verdict)


def _make_verdict(question="placeholder"):
    from eval.judge import JudgeVerdict, ScoredAspect

    return JudgeVerdict(
        question=question,
        faithfulness=ScoredAspect(score=1.0, justification="Fully grounded."),
        correctness=ScoredAspect(score=1.0, justification="Matches expected answer."),
        retrieval_quality=ScoredAspect(score=1.0, justification="Retrieved the expected source."),
    )


def test_judge_answer_parses_structured_response():
    from eval.judge import JudgeVerdict, judge_answer

    chunk = Chunk(id="a::0", text="Paris is the capital of France.", source="a.md", chunk_index=0)
    client = FakeClient(_make_verdict())

    result = judge_answer(
        question="What is the capital of France?",
        context_chunks=[chunk],
        generated_answer="Paris.",
        expected_answer="Paris",
        expected_sources=["a.md"],
        category="factual",
        client=client,
    )

    assert isinstance(result, JudgeVerdict)
    assert result.faithfulness.score == 1.0
    assert result.correctness.score == 1.0
    assert result.retrieval_quality.score == 1.0
    assert client.messages.last_kwargs["output_format"] is JudgeVerdict


def test_judge_answer_overrides_question_with_input_question():
    from eval.judge import judge_answer

    chunk = Chunk(id="a::0", text="text", source="a.md", chunk_index=0)
    client = FakeClient(_make_verdict(question="something the model echoed differently"))

    result = judge_answer(
        question="What is the capital of France?",
        context_chunks=[chunk],
        generated_answer="Paris.",
        expected_answer="Paris",
        client=client,
    )

    assert result.question == "What is the capital of France?"


def test_judge_answer_prompt_notes_out_of_scope_when_no_expected_sources():
    from eval.judge import judge_answer

    client = FakeClient(_make_verdict())

    judge_answer(
        question="What's the rate limiting story?",
        context_chunks=[],
        generated_answer="Out of scope.",
        expected_answer="Out of scope.",
        expected_sources=[],
        category="out_of_scope",
        client=client,
    )

    prompt = client.messages.last_kwargs["messages"][0]["content"]
    assert "out_of_scope" in prompt
