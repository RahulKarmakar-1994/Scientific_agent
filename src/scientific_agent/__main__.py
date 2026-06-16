import argparse
import json
import sys

from .agents import ScientificAgent


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "chat":
        chat_main(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "ui":
        from .ui import main as ui_main

        ui_main(sys.argv[2:])
        return
    run_main(sys.argv[1:])


def run_main(argv=None):
    parser = argparse.ArgumentParser(description="Run the Scientific Agent platform.")
    parser.add_argument("request", help="Natural-language scientific request")
    parser.add_argument(
        "--provider",
        choices=["ollama", "gemini", "openai"],
        default="ollama",
        help="LLM provider for planning/generation/reporting.",
    )
    parser.add_argument("--model", default=None, help="Override provider default model.")
    parser.add_argument(
        "--engine",
        choices=["local", "docker", "external-md"],
        default="local",
        help="Simulation execution engine.",
    )
    parser.add_argument("--index-dir", default=".vector_store", help="RAG index directory.")
    parser.add_argument(
        "--session-id",
        default=None,
        help="Optional session id. When set, recent conversation is used as memory.",
    )
    args = parser.parse_args(argv)

    report = ScientificAgent(
        provider=args.provider,
        model=args.model,
        engine=args.engine,
        index_dir=args.index_dir,
    ).run(args.request, session_id=args.session_id)
    print(json.dumps(report, indent=2))


def chat_main(argv=None):
    parser = argparse.ArgumentParser(description="Interactive Scientific Agent chat.")
    parser.add_argument(
        "--session",
        "--session-id",
        dest="session_id",
        default="default",
        help="Session id used to persist conversation memory.",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "gemini", "openai"],
        default="ollama",
        help="LLM provider for planning/generation/reporting.",
    )
    parser.add_argument("--model", default=None, help="Override provider default model.")
    parser.add_argument(
        "--engine",
        choices=["local", "docker", "external-md"],
        default="local",
        help="Simulation execution engine.",
    )
    parser.add_argument("--index-dir", default=".vector_store", help="RAG index directory.")
    args = parser.parse_args(argv)

    agent = ScientificAgent(
        provider=args.provider,
        model=args.model,
        engine=args.engine,
        index_dir=args.index_dir,
    )

    print(f"Scientific Agent chat session: {args.session_id}")
    print("Type 'exit' or 'quit' to stop.")
    while True:
        try:
            request = input("> ").strip()
        except EOFError:
            print()
            break
        if not request:
            continue
        if request.lower() in {"exit", "quit"}:
            break

        report = agent.run(request, session_id=args.session_id)
        result = report.get("result", {})
        print(result.get("final_answer") or result.get("answer") or json.dumps(report, indent=2))
        print(f"\n[job: {report['job_id']}]\n")


if __name__ == "__main__":
    main()
