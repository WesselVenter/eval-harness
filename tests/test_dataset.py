def test_load_golden_dataset_parses_jsonl(tmp_path):
    from eval.dataset import load_golden_dataset, GoldenItem

    path = tmp_path / "golden.jsonl"
    path.write_text(
        '{"id": "q1", "question": "Q1?", "expected_answer": "A1"}\n'
        '{"id": "q2", "question": "Q2?", "expected_answer": "A2", "expected_source_ids": ["doc.md::0"]}\n',
        encoding="utf-8",
    )

    items = load_golden_dataset(path)

    assert items == [
        GoldenItem(id="q1", question="Q1?", expected_answer="A1", expected_source_ids=None),
        GoldenItem(id="q2", question="Q2?", expected_answer="A2", expected_source_ids=["doc.md::0"]),
    ]


def test_load_golden_dataset_default_path_loads_real_dataset():
    from eval.dataset import load_golden_dataset

    items = load_golden_dataset()
    assert len(items) == 7
    assert items[0].id == "q1"
