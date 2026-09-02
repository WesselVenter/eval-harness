import pytest


def test_load_golden_dataset_parses_jsonl(tmp_path):
    from eval.dataset import load_golden_dataset, GoldenItem

    path = tmp_path / "golden.jsonl"
    path.write_text(
        '{"question": "Q1?", "expected_answer": "A1", "expected_sources": ["doc.md"], "category": "factual"}\n'
        '{"question": "Q2?", "expected_answer": "A2", "expected_sources": [], "category": "out_of_scope"}\n',
        encoding="utf-8",
    )

    items = load_golden_dataset(path)

    assert items == [
        GoldenItem(question="Q1?", expected_answer="A1", expected_sources=["doc.md"], category="factual"),
        GoldenItem(question="Q2?", expected_answer="A2", expected_sources=[], category="out_of_scope"),
    ]


def test_load_golden_dataset_skips_blank_lines(tmp_path):
    from eval.dataset import load_golden_dataset

    path = tmp_path / "golden.jsonl"
    path.write_text(
        '{"question": "Q1?", "expected_answer": "A1", "expected_sources": [], "category": "factual"}\n'
        "\n"
        '{"question": "Q2?", "expected_answer": "A2", "expected_sources": [], "category": "factual"}\n',
        encoding="utf-8",
    )

    items = load_golden_dataset(path)

    assert len(items) == 2


def test_load_golden_dataset_raises_clear_error_for_missing_field(tmp_path):
    from eval.dataset import load_golden_dataset

    path = tmp_path / "golden.jsonl"
    path.write_text(
        '{"question": "Q1?", "expected_answer": "A1", "expected_sources": [], "category": "factual"}\n'
        '{"question": "Q2?", "expected_answer": "A2", "expected_sources": []}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        load_golden_dataset(path)

    message = str(exc_info.value)
    assert "category" in message
    assert "2" in message


def test_load_golden_dataset_default_path_loads_real_dataset():
    from eval.dataset import load_golden_dataset

    items = load_golden_dataset()
    assert len(items) == 12
    assert items[0].question.startswith("How do I share a database connection")
