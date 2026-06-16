import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class LoadedTool:
    name: str
    description: str
    execution_type: str
    manifest_path: Path
    command: list[str] | None = None
    timeout_seconds: int = 30


class ToolLoader:
    """Load user/project tools from tools/*/tool.yaml manifests."""

    def __init__(self, repo_root="."):
        self.repo_root = Path(repo_root).resolve()
        self.tools_dir = self.repo_root / "tools"
        self.tools = self._load_tools()

    def describe(self):
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "execution_type": tool.execution_type,
            }
            for tool in self.tools.values()
        ]

    def describe_executable(self):
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "execution_type": tool.execution_type,
            }
            for tool in self.tools.values()
            if tool.execution_type == "command"
        ]

    def executable_names(self):
        return {
            name
            for name, tool in self.tools.items()
            if tool.execution_type == "command"
        }

    def has(self, name):
        return name in self.tools

    def execute(self, name, payload):
        if name not in self.tools:
            raise KeyError(f"Unknown tool: {name}")

        tool = self.tools[name]
        if tool.execution_type != "command":
            raise ValueError(f"Tool {name} is not executable by ToolLoader yet")
        if not tool.command:
            raise ValueError(f"Tool {name} does not define a command")

        command = self._resolve_command(tool.command)
        completed = subprocess.run(
            command,
            input=json.dumps(payload),
            text=True,
            cwd=self.repo_root,
            capture_output=True,
            timeout=tool.timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            return {
                "status": "failed",
                "tool": name,
                "returncode": completed.returncode,
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            }

        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError:
            return {
                "status": "completed",
                "tool": name,
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            }

    def _load_tools(self):
        tools = {}
        if not self.tools_dir.exists():
            return tools

        for manifest_path in sorted(self.tools_dir.glob("*/tool.yaml")):
            raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
            name = raw.get("name") or manifest_path.parent.name
            execution = raw.get("execution") or {}
            execution_type = raw.get("type") or execution.get("type") or "metadata"
            command = raw.get("command")
            if isinstance(command, str):
                command = shlex.split(command)

            tools[name] = LoadedTool(
                name=name,
                description=raw.get("description", ""),
                execution_type=execution_type,
                command=command,
                timeout_seconds=int(execution.get("timeout_seconds", raw.get("timeout_seconds", 30))),
                manifest_path=manifest_path,
            )
        return tools

    def _resolve_command(self, command):
        if not command:
            raise ValueError("Empty command")

        resolved = list(command)
        if resolved[0] in {"python", "python3"}:
            resolved[0] = sys.executable

        for index, part in enumerate(resolved):
            if part.endswith(".py"):
                script_path = (self.repo_root / part).resolve()
                if not _is_relative_to(script_path, self.repo_root):
                    raise ValueError(f"Tool script is outside repo: {script_path}")
                resolved[index] = str(script_path)
        return resolved


def _is_relative_to(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
