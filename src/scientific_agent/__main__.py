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
        choices=["ollama", "gemini", "groq", "openai"],
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
    parser.add_argument(
        "--predict",
        default=None,
        help="Optional learner prediction to compare against the demo result.",
    )
    parser.add_argument(
        "--ask-prediction",
        action="store_true",
        help="Pause before running and ask for a learner prediction.",
    )
    args = parser.parse_args(argv)

    learner_prediction = args.predict
    if args.ask_prediction and not learner_prediction:
        print("Before I run the demo, what do you predict will happen?")
        learner_prediction = input("> ").strip()

    report = ScientificAgent(
        provider=args.provider,
        model=args.model,
        engine=args.engine,
        index_dir=args.index_dir,
    ).run(
        args.request,
        session_id=args.session_id,
        learner_prediction=learner_prediction,
    )
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
        choices=["ollama", "gemini", "groq", "openai"],
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
        llm = result.get("llm") or {}
        if llm.get("error") and not llm.get("last_successful_model"):
            print(
                "\nLLM warning: the requested model did not return a response, "
                "so fallback rules were used."
            )
            print(f"Reason: {llm.get('error')}")
        print(f"\n[job: {report['job_id']}]\n")


if __name__ == "__main__":
    main()
