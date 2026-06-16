import json

from src.scientific_agent.core.tool_loader import ToolLoader

from .analysis import analyze_md_result
from .knowledge import KnowledgeAgent
from .llm import LLMClient
from .parser import parse_md_request
from .runner import MDRunner


DEFAULT_MD_TOOL_PLAN = [
    "parse_md_request",
    "search_md_knowledge",
    "run_md",
    "analyze_md_result",
    "write_final_answer",
]


class PortableMDToolAgent:
    """Provider-agnostic scientific tool agent.

    The LLM chooses a plan from built-in trusted tools plus dynamically loaded
    tools from tools/*/tool.yaml. Python still validates and executes.
    """

    def __init__(self, provider="ollama", model=None, engine="local"):
        self.llm = LLMClient(provider=provider, model=model)
        self.engine = engine
        self.knowledge_agent = KnowledgeAgent()
        self.runner = MDRunner()
        self.tool_loader = ToolLoader()

    def execute(self, query):
        plan = self._plan(query)
        state = {
            "request": query,
            "provider": self.llm.provider,
            "model": self.llm.model,
            "engine": self.engine,
            "planned_tools": plan,
            "available_dynamic_tools": self.tool_loader.describe(),
            "tool_calls": [],
        }

        for tool_name in plan:
            if tool_name == "parse_md_request":
                state["config"] = parse_md_request(query)
                self._record(state, tool_name, {"config": state["config"]})
            elif tool_name == "search_md_knowledge":
                state["knowledge_evidence"] = self.knowledge_agent.search(query)
                self._record(
                    state,
                    tool_name,
                    {"evidence_count": len(state["knowledge_evidence"])},
                )
            elif tool_name == "run_md":
                config = state.get("config") or parse_md_request(query)
                state["config"] = config
                state["md_result"] = self.runner.run(config, engine=self.engine)
                self._record(state, tool_name, {"summary": state["md_result"]["summary"]})
            elif tool_name == "analyze_md_result":
                if "md_result" not in state:
                    config = state.get("config") or parse_md_request(query)
                    state["md_result"] = self.runner.run(config, engine=self.engine)
                state["analysis"] = analyze_md_result(state["md_result"])
                self._record(state, tool_name, {"analysis": state["analysis"]})
            elif tool_name == "write_final_answer":
                state["final_answer"] = self._final_answer(query, state)
                self._record(state, tool_name, {"has_final_answer": bool(state["final_answer"])})
            elif self.tool_loader.has(tool_name):
                payload = self._payload_for_dynamic_tool(tool_name, query, state)
                result = self.tool_loader.execute(tool_name, payload)
                state.setdefault("dynamic_tool_results", {})[tool_name] = result
                self._record(
                    state,
                    tool_name,
                    {
                        "status": result.get("status"),
                        "generated_files": result.get("generated_files", []),
                    },
                )

        state["llm"] = self._llm_status()
        return state

    def _plan(self, query):
        fallback = _fallback_plan(query, self.tool_loader)
        if not self.llm.available:
            return fallback

        executable_tools = self.tool_loader.describe_executable()
        text = self.llm.respond(_planning_prompt(query, executable_tools))
        parsed = _extract_json(text)
        if not parsed:
            return fallback

        requested_tools = _normalize_tool_names(parsed.get("tools", []))
        allowed = set(DEFAULT_MD_TOOL_PLAN) | self.tool_loader.executable_names()
        plan = [tool for tool in requested_tools if tool in allowed]
        if not plan:
            return fallback
        if "write_final_answer" not in plan:
            plan.append("write_final_answer")
        return _complete_plan(query, _order_plan(plan, self.tool_loader), self.tool_loader)

    def _payload_for_dynamic_tool(self, tool_name, query, state):
        if tool_name == "python_demo_runner":
            return self._python_demo_payload(query)
        return {"query": query, "state": state}

    def _python_demo_payload(self, query):
        fallback_code = _fallback_python_demo_code(query)
        if not self.llm.available:
            return {"topic": query, "code": fallback_code}

        text = self.llm.respond(_python_demo_prompt(query))
        parsed = _extract_json(text)
        return {
            "topic": (parsed or {}).get("topic") or query,
            "code": (parsed or {}).get("code") or fallback_code,
        }

    def _final_answer(self, query, state):
        if "md_result" in state:
            return _md_final_answer(query, state)
        if _has_python_demo_without_md(state):
            return _python_demo_final_answer(query, state)
        if not self.llm.available:
            return None
        return self.llm.respond(_final_answer_prompt(query, state))

    def _llm_status(self):
        return {
            "enabled": self.llm.available,
            "provider": self.llm.provider,
            "model": self.llm.model,
            "candidate_models": self.llm.models,
            "last_successful_model": self.llm.last_successful_model,
            "attempts": self.llm.attempts,
            "error": self.llm.error,
        }

    @staticmethod
    def _record(state, tool_name, output_summary):
        state["tool_calls"].append(
            {
                "tool": tool_name,
                "output_summary": output_summary,
            }
        )


def run_portable_tool_agent(query, provider="ollama", model=None, engine="local"):
    return PortableMDToolAgent(
        provider=provider,
        model=model,
        engine=engine,
    ).execute(query)


def _planning_prompt(query, dynamic_tools):
    return f"""
You are a scientific workflow planner. Choose which trusted tools should be
called for the user request. Return only JSON with one key: "tools".

Built-in tools:
- parse_md_request: parse natural language into MD config
- search_md_knowledge: search local MD notes and papers
- run_md: run the configured MD engine
- analyze_md_result: analyze simulation output
- write_final_answer: summarize what happened

Discovered dynamic tools:
{json.dumps(dynamic_tools, indent=2)}

For requests that ask to run, design, or analyze MD, use the MD tools.
For requests that ask to learn, teach, explain with Python, demonstrate, plot,
or simulate a concept with simple Python, use python_demo_runner.
Do not invent tools.

User request:
{query}

Return example:
{{"tools": ["python_demo_runner", "write_final_answer"]}}
""".strip()


def _final_answer_prompt(query, state):
    compact_state = {
        "request": query,
        "engine": state.get("engine"),
        "config": state.get("config"),
        "knowledge_evidence": state.get("knowledge_evidence"),
        "md_result_summary": (state.get("md_result") or {}).get("summary"),
        "analysis": state.get("analysis"),
        "dynamic_tool_results": state.get("dynamic_tool_results"),
        "tool_calls": state.get("tool_calls"),
    }
    return f"""
You are a scientific assistant. Write a concise final answer from this trusted
tool state. Do not invent calculation results. If an MD result is present and
the engine is local, say that it is a toy MD calculation. If a Python demo ran,
teach the concept briefly, quote the numerical stdout result, and mention any
generated file paths. If no MD result is present, do not discuss MD.

Tool state:
{json.dumps(compact_state, indent=2)}
""".strip()


def _python_demo_prompt(query):
    return f"""
Write a short Python demo for this learning request. Return only JSON with
"topic" and "code".

Rules:
- code must be self-contained
- use only math, random, statistics, numpy, matplotlib, collections, itertools
- print a short numerical summary
- if plotting, save to demo.png
- do not read files
- do not use network, subprocess, os, sys, pathlib, open, eval, or exec

Learning request:
{query}

For entropy, prefer a coin/Bernoulli entropy demo showing entropy versus
probability. For diffusion, prefer a one-dimensional random-walk demo showing
mean squared displacement growth with time.
""".strip()


def _fallback_python_demo_code(query):
    if "entropy" in query.lower():
        return """
import math
import numpy as np
import matplotlib.pyplot as plt

def entropy_binary(p):
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))

probabilities = np.linspace(0.0, 1.0, 101)
entropies = [entropy_binary(float(p)) for p in probabilities]
max_index = int(np.argmax(entropies))

print("Binary entropy demo")
print(f"maximum entropy = {entropies[max_index]:.3f} bits at p = {probabilities[max_index]:.2f}")
print(f"entropy at p=0.10 is {entropy_binary(0.10):.3f} bits")
print(f"entropy at p=0.50 is {entropy_binary(0.50):.3f} bits")
print("Interpretation: uncertainty is largest when the two outcomes are equally likely.")

plt.figure(figsize=(6, 4))
plt.plot(probabilities, entropies)
plt.xlabel("Probability of heads")
plt.ylabel("Entropy (bits)")
plt.title("Binary entropy")
plt.tight_layout()
plt.savefig("demo.png", dpi=150)
""".strip()

    if "diffusion" in query.lower() or "random walk" in query.lower():
        return """
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
n_walkers = 2000
n_steps = 300
steps = rng.choice([-1, 1], size=(n_walkers, n_steps))
positions = np.cumsum(steps, axis=1)
times = np.arange(1, n_steps + 1)
mean_squared_displacement = np.mean(positions**2, axis=0)

fit_start = 50
slope, intercept = np.polyfit(times[fit_start:], mean_squared_displacement[fit_start:], 1)

print("Diffusion random-walk demo")
print(f"walkers = {n_walkers}")
print(f"steps per walker = {n_steps}")
print(f"final mean squared displacement = {mean_squared_displacement[-1]:.2f}")
print(f"MSD growth slope = {slope:.2f} per step")
print("Interpretation: diffusion spreads because many random steps make the mean squared displacement grow approximately linearly with time.")

plt.figure(figsize=(6, 4))
plt.plot(times, mean_squared_displacement, label="simulation")
plt.plot(times, slope * times + intercept, "--", label="linear fit")
plt.xlabel("Step")
plt.ylabel("Mean squared displacement")
plt.title("1D random walk diffusion")
plt.legend()
plt.tight_layout()
plt.savefig("demo.png", dpi=150)
""".strip()

    return """
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0.0, 10.0, 200)
y = np.sin(x)
print("Simple scientific Python demo")
print(f"sample count = {len(x)}")
print(f"mean value = {float(np.mean(y)):.3f}")

plt.figure(figsize=(6, 4))
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.title("Demo plot")
plt.tight_layout()
plt.savefig("demo.png", dpi=150)
""".strip()


def _has_python_demo_without_md(state):
    return (
        "md_result" not in state
        and "python_demo_runner" in (state.get("dynamic_tool_results") or {})
    )


def _md_final_answer(query, state):
    summary = (state.get("md_result") or {}).get("summary", {})
    analysis = state.get("analysis") or {}
    lines = [
        f"Request: {query}",
        "",
        "The MD workflow ran and returned a structured result.",
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


def _python_demo_final_answer(query, state):
    result = state["dynamic_tool_results"]["python_demo_runner"]
    stdout = result.get("stdout", "").strip()
    files = result.get("generated_files", [])
    concept_lines = _learning_concept_lines(query)
    answer = [
        f"Request: {query}",
        "",
        *concept_lines,
        "",
        "Python demo output:",
        stdout or "(no stdout captured)",
    ]
    if files:
        answer.extend(["", "Generated files:", *[f"- {path}" for path in files]])
    answer.extend(
        [
            "",
            "This was run through the `python_demo_runner` plugin tool, not an MD",
            "calculation.",
        ]
    )
    return "\n".join(answer)


def _learning_concept_lines(query):
    lower_query = query.lower()
    if "entropy" in lower_query:
        return [
            "Entropy is a measure of uncertainty. For a binary event such as a coin",
            "flip, entropy is highest when both outcomes are equally likely and lower",
            "when one outcome is much more predictable.",
        ]
    if "diffusion" in lower_query or "random walk" in lower_query:
        return [
            "Diffusion is the spreading of particles caused by random motion.",
            "A random-walk demo captures the core idea: individual steps are",
            "unpredictable, but the mean squared displacement grows predictably",
            "with time.",
        ]
    return [
        "This learning request was handled by generating and running a small",
        "Python demonstration through a trusted plugin tool.",
    ]


def _fallback_plan(query, tool_loader):
    lower_query = query.lower()
    learning_markers = ["teach", "learn", "explain", "demo", "demonstrate", "python", "plot"]
    md_markers = ["md", "molecular dynamics", "nvt", "nve", "npt", "gromacs", "lammps"]
    if tool_loader.has("python_demo_runner") and any(marker in lower_query for marker in learning_markers):
        return ["python_demo_runner", "write_final_answer"]
    if any(marker in lower_query for marker in md_markers):
        return list(DEFAULT_MD_TOOL_PLAN)
    if tool_loader.has("python_demo_runner"):
        return ["python_demo_runner", "write_final_answer"]
    return ["write_final_answer"]


def _complete_plan(query, plan, tool_loader):
    lower_query = query.lower()
    md_markers = ["md", "molecular dynamics", "nvt", "nve", "npt", "run", "simulate", "simulation"]
    learning_markers = ["teach", "learn", "explain", "demo", "demonstrate", "python", "plot"]
    if (
        any(marker in lower_query for marker in md_markers)
        and "python_demo_runner" not in plan
        and not any(marker in lower_query for marker in learning_markers)
    ):
        return list(DEFAULT_MD_TOOL_PLAN)
    return _order_plan(plan, tool_loader)


def _extract_json(text):
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _normalize_tool_names(raw_tools):
    names = []
    for item in raw_tools:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            name = item.get("name") or item.get("tool")
            if isinstance(name, str):
                names.append(name)
    return names


def _order_plan(plan, tool_loader):
    dynamic_names = sorted(tool_loader.tools)
    preferred_tools = DEFAULT_MD_TOOL_PLAN[:-1] + dynamic_names + ["write_final_answer"]
    preferred_order = {tool: index for index, tool in enumerate(preferred_tools)}
    return sorted(dict.fromkeys(plan), key=lambda tool: preferred_order.get(tool, 999))
