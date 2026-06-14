import argparse
import json

from .analysis import analyze_md_result
from .knowledge import KnowledgeAgent
from .langchain_agent import run_langchain_tool_agent
from .llm import LLMClient
from .parser import parse_md_request
from .runner import MDRunner
from .workflow_orchestrator import WorkflowOrchestrator


class MDAgentWorkbench:
    def __init__(self, use_llm=True, provider="gemini", model=None, engine="local"):
        self.knowledge_agent = KnowledgeAgent()
        self.runner = MDRunner()
        self.engine = engine
        self.llm = LLMClient(provider=provider, model=model) if use_llm else None

    def execute(self, query):
        config = parse_md_request(query)
        evidence = self.knowledge_agent.search(query)
        llm_status = self._llm_status()

        if self.llm and self.llm.available:
            config = self.llm.improve_config(query, config, evidence)

        result = self.runner.run(config, engine=self.engine)
        analysis = analyze_md_result(result)
        llm_report = None

        if self.llm and self.llm.available:
            llm_report = self.llm.write_report(
                query=query,
                config=config,
                evidence=evidence,
                result_summary=result["summary"],
                analysis=analysis,
            )
            llm_status = self._llm_status()

        return {
            "request": query,
            "llm": llm_status,
            "config": config,
            "knowledge_evidence": evidence,
            "engine": self.engine,
            "result_summary": result["summary"],
            "analysis": analysis,
            "llm_report": llm_report,
        }

    def _llm_status(self):
        if not self.llm:
            return {"enabled": False, "reason": "disabled by CLI"}
        if self.llm.available:
            return {
                "enabled": True,
                "provider": self.llm.provider,
                "model": self.llm.model,
                "candidate_models": self.llm.models,
                "last_successful_model": self.llm.last_successful_model,
                "attempts": self.llm.attempts,
                "error": self.llm.error,
            }
        return {
            "enabled": False,
            "provider": self.llm.provider,
            "model": self.llm.model,
            "candidate_models": self.llm.models,
            "reason": self.llm.error,
        }


def main():
    parser = argparse.ArgumentParser(description="Run the MD Agent Workbench prototype.")
    parser.add_argument("query", help="Natural-language MD request")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Run only the local rule-based workflow.",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "openai"],
        default="gemini",
        help="LLM provider to use when LLM mode is enabled.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the default model for the selected provider.",
    )
    parser.add_argument(
        "--engine",
        choices=["local", "docker", "external-md"],
        default="local",
        help="Execution engine for the MD calculation.",
    )
    parser.add_argument(
        "--mode",
        choices=["workflow", "tool-agent", "orchestrate"],
        default="workflow",
        help="Run deterministic workflow, LangChain tool-agent, or YAML orchestrator mode.",
    )
    parser.add_argument(
        "--workflow",
        default="workflows/basic_md_workflow.yaml",
        help="Workflow YAML path for orchestrate mode.",
    )
    args = parser.parse_args()

    if args.mode == "orchestrate":
        report = WorkflowOrchestrator(engine=args.engine).run(
            workflow_path=args.workflow,
            request=args.query,
        )
        print(json.dumps(report, indent=2))
        return

    if args.mode == "tool-agent":
        report = run_langchain_tool_agent(
            query=args.query,
            model=args.model,
            engine=args.engine,
        )
        print(json.dumps(report, indent=2))
        return

    report = MDAgentWorkbench(
        use_llm=not args.no_llm,
        provider=args.provider,
        model=args.model,
        engine=args.engine,
    ).execute(args.query)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
