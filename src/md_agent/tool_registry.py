from dataclasses import dataclass

from .analysis import analyze_md_result
from .knowledge import KnowledgeAgent
from .parser import parse_md_request
from .runner import MDRunner


@dataclass
class ToolSpec:
    name: str
    description: str
    callable: object


class ToolRegistry:
    """Small local registry for agent-callable tools.

    This is the bridge toward a Discovery-like design. Today the Python
    orchestrator calls these tools; later LangChain/LangGraph can wrap them.
    """

    def __init__(self):
        knowledge_agent = KnowledgeAgent()
        runner = MDRunner()
        self.tools = {
            "parse_md_request": ToolSpec(
                name="parse_md_request",
                description="Convert a natural-language MD request into config JSON.",
                callable=parse_md_request,
            ),
            "search_md_knowledge": ToolSpec(
                name="search_md_knowledge",
                description="Search local MD papers and force-field notes.",
                callable=knowledge_agent.search,
            ),
            "run_md": ToolSpec(
                name="run_md",
                description="Run MD using the local or Docker engine.",
                callable=runner.run,
            ),
            "analyze_md_result": ToolSpec(
                name="analyze_md_result",
                description="Analyze MD output stability and summary values.",
                callable=analyze_md_result,
            ),
        }

    def get(self, name):
        return self.tools[name].callable

    def describe(self):
        return [
            {"name": spec.name, "description": spec.description}
            for spec in self.tools.values()
        ]
