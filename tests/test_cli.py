from app.models import Chunk


def test_cli_ingest_command(monkeypatch, capsys):
    import app.cli as cli

    monkeypatch.setattr(cli, "ingest", lambda: 7)

    exit_code = cli.main(["ingest"])

    assert exit_code == 0
    assert "7" in capsys.readouterr().out


def test_cli_ask_command(monkeypatch, capsys):
    import app.cli as cli

    chunk = Chunk(id="a::0", text="Paris is the capital of France.", source="a.md", chunk_index=0)
    monkeypatch.setattr(cli, "retrieve", lambda query, mode, k: [chunk])
    monkeypatch.setattr(cli, "generate_answer", lambda question, chunks: "Paris.")

    exit_code = cli.main(["ask", "What is the capital of France?", "--mode", "dense", "--k", "1"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Paris." in out
    assert "a.md" in out
