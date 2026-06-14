from pathlib import Path

import yaml

from .agent_registry import AgentRegistry


class WorkflowOrchestrator:
    def __init__(self, engine="local"):
        self.agent_registry = AgentRegistry(engine=engine)

    def run(self, workflow_path, request):
        workflow = _load_workflow(workflow_path)
        state = {
            "request": request,
            "workflow": workflow.get("name", str(workflow_path)),
            "steps": [],
        }

        for step in workflow.get("steps", []):
            step_name = step["name"]
            agent_name = step["agent"]
            step_config = step.get("with", {})
            output = self.agent_registry.run(
                agent_name=agent_name,
                request=request,
                state=state,
                step_config=step_config,
            )
            state.update(output)
            state["steps"].append(
                {
                    "name": step_name,
                    "agent": agent_name,
                    "output_keys": sorted(output.keys()),
                }
            )

        return state


def _load_workflow(workflow_path):
    path = Path(workflow_path)
    if not path.exists():
        raise FileNotFoundError(f"Workflow not found: {workflow_path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))
