import argparse

from app.generate import generate_answer
from app.ingest import ingest
from app.retrieve import retrieve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ingest")

    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid")
    ask_parser.add_argument("--k", type=int, default=4)

    args = parser.parse_args(argv)

    if args.command == "ingest":
        count = ingest()
        print(f"Ingested {count} chunks.")
        return 0

    if args.command == "ask":
        chunks = retrieve(args.question, mode=args.mode, k=args.k)
        answer = generate_answer(args.question, chunks)
        print(answer)
        print("\nSources:")
        for c in chunks:
            print(f"  - {c.source} (chunk {c.chunk_index})")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
