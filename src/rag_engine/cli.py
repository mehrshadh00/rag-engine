"""Small command-line interface for ingestion and asking questions."""

import argparse
import json
from .pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="RAG Engine CLI")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest_parser = commands.add_parser("ingest", help="Index a directory of documents")
    ingest_parser.add_argument("directory", nargs="?", default="data")
    ask_parser = commands.add_parser("ask", help="Ask a question over indexed documents")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    pipeline = RAGPipeline()
    if args.command == "ingest":
        result = pipeline.ingest(args.directory)
    else:
        result = pipeline.ask(args.question, args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
