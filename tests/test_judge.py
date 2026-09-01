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
