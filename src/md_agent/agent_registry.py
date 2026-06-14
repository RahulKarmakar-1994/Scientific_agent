from dataclasses import dataclass
from pathlib import Path

from .analysis import analyze_md_result
from .knowledge import KnowledgeAgent
from .parser import parse_md_request
from .runner import MDRunner


@dataclass
class AgentSpec:
    name: str
    description: str
    handler: object


class AgentRegistry:
    """Registry for pluggable agents.

    Each agent receives the user request, shared workflow state, and step config.
    Agents return a dictionary that is merged back into workflow state.
    """

    def __init__(self, engine="local"):
        self.engine = engine
        self.knowledge_agent = KnowledgeAgent()
        self.runner = MDRunner()
        self.agents = {
            "md_setup_agent": AgentSpec(
                name="md_setup_agent",
                description="Builds a structured MD config from the user request.",
                handler=self._run_md_setup_agent,
            ),
            "md_knowledge_agent": AgentSpec(
                name="md_knowledge_agent",
                description="Searches local papers and force-field notes.",
                handler=self._run_md_knowledge_agent,
            ),
            "md_runner_agent": AgentSpec(
                name="md_runner_agent",
                description="Runs an MD tool using the selected execution engine.",
                handler=self._run_md_runner_agent,
            ),
            "md_analysis_agent": AgentSpec(
                name="md_analysis_agent",
                description="Analyzes MD tool outputs.",
                handler=self._run_md_analysis_agent,
            ),
            "md_report_agent": AgentSpec(
                name="md_report_agent",
                description="Writes a short workflow report.",
                handler=self._run_md_report_agent,
            ),
        }
        self._load_manifest_names()

    def run(self, agent_name, request, state, step_config=None):
        if agent_name not in self.agents:
            raise KeyError(f"Unknown agent: {agent_name}")
        return self.agents[agent_name].handler(request, state, step_config or {})

    def describe(self):
        return [
            {"name": spec.name, "description": spec.description}
            for spec in self.agents.values()
        ]

    def _run_md_setup_agent(self, request, state, step_config):
        config = parse_md_request(request)
        return {"config": config}

    def _run_md_knowledge_agent(self, request, state, step_config):
        limit = int(step_config.get("limit", 3))
        evidence = self.knowledge_agent.search(request, limit=limit)
        return {"knowledge_evidence": evidence}

    def _run_md_runner_agent(self, request, state, step_config):
        config = state.get("config") or parse_md_request(request)
        engine = step_config.get("engine", self.engine)
        result = self.runner.run(config, engine=engine)
        return {"engine": engine, "result": result, "result_summary": result["summary"]}

    def _run_md_analysis_agent(self, request, state, step_config):
        result = state.get("result")
        if not result:
            return {
                "analysis": {
                    "stability": "failed",
                    "notes": ["No MD result found in workflow state."],
                }
            }
        return {"analysis": analyze_md_result(result)}

    def _run_md_report_agent(self, request, state, step_config):
        report = [
            "# MD Workflow Report",
            "",
            f"Request: {request}",
            f"Engine: {state.get('engine', self.engine)}",
            "",
            "## Config",
            _format_dict(state.get("config", {})),
            "",
            "## Result Summary",
            _format_dict(state.get("result_summary", {})),
            "",
            "## Analysis",
            _format_dict(state.get("analysis", {})),
        ]
        report_text = "\n".join(report)
        output_path = Path("outputs") / "latest_workflow_report.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_text, encoding="utf-8")
        return {"report": report_text, "report_path": str(output_path)}

    def _load_manifest_names(self):
        """Keep manifest folders visible, even before custom handlers exist."""
        for manifest in Path("agents").glob("*/agent.yaml"):
            name = _read_manifest_name(manifest)
            if name and name not in self.agents:
                self.agents[name] = AgentSpec(
                    name=name,
                    description=f"Manifest-only agent from {manifest}",
                    handler=self._run_unimplemented_agent,
                )

    def _run_unimplemented_agent(self, request, state, step_config):
        return {
            "status": "skipped",
            "notes": ["This agent has a manifest but no Python handler yet."],
        }


def _read_manifest_name(path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("name:"):
            return line.split(":", 1)[1].strip()
    return None


def _format_dict(value):
    if not value:
        return "_None_"
    lines = []
    for key, item in value.items():
        lines.append(f"- {key}: {item}")
    return "\n".join(lines)
