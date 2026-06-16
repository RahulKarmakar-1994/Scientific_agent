from src.md_agent.analysis import analyze_md_result
from src.md_agent.knowledge import KnowledgeAgent
from src.md_agent.parser import parse_md_request
from src.md_agent.runner import MDRunner


class SimulationAgent:
    """Run and analyze simulation workflows."""

    def __init__(self, engine="local"):
        self.engine = engine
        self.knowledge_agent = KnowledgeAgent()
        self.runner = MDRunner()

    def run(self, request):
        config = parse_md_request(request)
        evidence = self.knowledge_agent.search(request)
        result = self.runner.run(config, engine=self.engine)
        analysis = analyze_md_result(result)
        return {
            "agent": "simulation",
            "status": result.get("status", "completed"),
            "request": request,
            "engine": self.engine,
            "config": config,
            "knowledge_evidence": evidence,
            "md_result": result,
            "analysis": analysis,
            "final_answer": _final_answer(request, result, analysis),
        }


def _final_answer(request, result, analysis):
    summary = result.get("summary", {})
    lines = [
        f"Request: {request}",
        "",
        "The simulation workflow ran and returned a structured result.",
        "",
        f"Steps completed: {summary.get('steps_completed')}",
        f"Final energy: {summary.get('final_energy')}",
        f"Mean temperature: {summary.get('mean_temperature_k')} K",
        f"Stability: {analysis.get('stability')}",
    ]
    notes = analysis.get("notes") or []
    if notes:
        lines.extend(["", "Notes:", *[f"- {note}" for note in notes]])
    return "\n".join(lines)
