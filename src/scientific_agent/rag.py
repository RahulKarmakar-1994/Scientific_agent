import argparse
import json

from .core.rag import LocalRAG
from .core.llm import LLMClient


def main():
    parser = argparse.ArgumentParser(description="Local RAG ingest/search utility.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Build a local RAG index.")
    ingest_parser.add_argument("paths", nargs="+", help="PDF/TXT/MD files or directories.")
    ingest_parser.add_argument("--index-dir", default=".vector_store")

    search_parser = subparsers.add_parser("search", help="Search the local RAG index.")
    search_parser.add_argument("query", help="Search query.")
    search_parser.add_argument("--index-dir", default=".vector_store")
    search_parser.add_argument("--top-k", type=int, default=5)

    answer_parser = subparsers.add_parser(
        "answer", help="Answer a question using retrieved local RAG context."
    )
    answer_parser.add_argument("query", help="Question to answer.")
    answer_parser.add_argument("--index-dir", default=".vector_store")
    answer_parser.add_argument("--top-k", type=int, default=5)
    answer_parser.add_argument(
        "--provider",
        choices=["gemini", "groq", "openai", "ollama"],
        default="ollama",
        help="LLM provider used to synthesize the answer.",
    )
    answer_parser.add_argument(
        "--model",
        default=None,
        help="Override the default model for the selected provider.",
    )

    args = parser.parse_args()
    rag = LocalRAG(index_dir=args.index_dir)

    if args.command == "ingest":
        print(json.dumps(rag.ingest(args.paths), indent=2))
        return

    if args.command == "search":
        print(json.dumps(rag.search(args.query, top_k=args.top_k), indent=2))
        return

    if args.command == "answer":
        hits = rag.search(args.query, top_k=args.top_k)
        llm = LLMClient(provider=args.provider, model=args.model)
        answer = None
        if llm.available:
            answer = llm.respond(_answer_prompt(args.query, hits))
        print(
            json.dumps(
                {
                    "query": args.query,
                    "llm": {
                        "enabled": llm.available,
                        "provider": llm.provider,
                        "model": llm.model,
                        "last_successful_model": llm.last_successful_model,
                        "attempts": llm.attempts,
                        "error": llm.error,
                    },
                    "answer": answer,
                    "evidence": _compact_hits(hits),
                },
                indent=2,
            )
        )


def _answer_prompt(query, hits):
    evidence = _compact_hits(hits)
    return f"""
You are a scientific reading assistant. Answer the question using only the
provided evidence. If the evidence is not enough, say what is missing. Cite
sources inline using [source, page N, chunk id].

Question:
{query}

Evidence:
{json.dumps(evidence, indent=2)}
""".strip()


def _compact_hits(hits):
    return [
        {
            "score": hit["score"],
            "source": hit["source"],
            "page": hit.get("page"),
            "chunk_id": hit["chunk_id"],
            "text": hit["text"],
        }
        for hit in hits
    ]


if __name__ == "__main__":
    main()
