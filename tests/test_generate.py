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
