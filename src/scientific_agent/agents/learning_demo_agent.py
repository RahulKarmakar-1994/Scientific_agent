import json
from pathlib import Path

from src.scientific_agent.agents.lesson_interaction_agent import LessonInteractionAgent
from src.scientific_agent.agents.request_understanding_agent import RequestUnderstandingAgent
from src.scientific_agent.agents.simulation_spec_agent import SimulationSpecAgent
from src.scientific_agent.agents.simulation_spec_verifier_agent import (
    SimulationSpecVerifierAgent,
)
from src.scientific_agent.core.demo_primitives import build_demo_code
from src.scientific_agent.core.grounding import GroundingService
from src.scientific_agent.core.llm import LLMClient
from src.scientific_agent.core.tool_loader import ToolLoader


class LearningDemoAgent:
    """Teach a scientific concept with a generated Python demo."""

    def __init__(
        self,
        provider="ollama",
        model=None,
        max_repair_attempts=2,
        index_dir=".vector_store",
    ):
        self.llm = LLMClient(provider=provider, model=model)
        self.tool_loader = ToolLoader()
        self.max_repair_attempts = max_repair_attempts
        self.agent_spec = _load_agent_spec()
        self.request_understanding_agent = RequestUnderstandingAgent(
            provider=provider,
            model=model,
        )
        self.simulation_spec_agent = SimulationSpecAgent(
            provider=provider,
            model=model,
        )
        self.simulation_spec_verifier_agent = SimulationSpecVerifierAgent(
            provider=provider,
            model=model,
        )
        self.lesson_interaction_agent = LessonInteractionAgent(
            provider=provider,
            model=model,
        )
        self.grounding_service = GroundingService(index_dir=index_dir)

    def run(
        self,
        request,
        conversation_context="",
        learner_prediction=None,
        prepared_simulation_spec=None,
        prepared_rich_demo_plan=None,
    ):
        understanding = self.request_understanding_agent.run(request, conversation_context)
        prompt_context = conversation_context if understanding.get("uses_context") else ""
        intent = self._classify_intent(request)
        grounding = self.grounding_service.retrieve(
            request,
            prompt_context,
            search_queries=understanding.get("search_queries"),
        )
        if intent.get("mode") == "concept_explanation":
            return self._answer_concept(
                request,
                intent,
                grounding,
                prompt_context,
                understanding,
            )

        if not self.tool_loader.has("python_demo_runner"):
            return {
                "agent": "learning_demo",
                "status": "failed",
                "error": "python_demo_runner tool is not available",
            }

        simulation_spec = prepared_simulation_spec or self.simulation_spec_agent.run(
            request,
            grounding=grounding,
            conversation_context=prompt_context,
        )
        simulation_spec_verification = self.simulation_spec_verifier_agent.run(
            request,
            simulation_spec,
            grounding=grounding,
        )
        code_payload = self._generate_code(
            request,
            grounding,
            prompt_context,
            simulation_spec=simulation_spec,
            simulation_spec_verification=simulation_spec_verification,
            rich_demo_plan=prepared_rich_demo_plan,
        )
        concept_answer = _clean_concept_answer(
            code_payload.get("concept_answer")
            or self._generate_concept_answer(request, grounding, prompt_context)
        )
        if not code_payload["code"]:
            result = _unreliable_demo_result(request, code_payload["reason"])
            lesson_interaction = self.lesson_interaction_agent.reflect(
                request,
                learner_prediction,
                code_payload.get("simulation_spec") or simulation_spec,
                result,
            )
            return {
                "agent": "learning_demo",
                "status": "unreliable",
                "request": request,
                "tool": "python_demo_runner",
                "attempts": [],
                "generated_code": None,
                "result": result,
                "concept_answer": concept_answer,
                "simulation_spec": code_payload.get("simulation_spec") or simulation_spec,
                "simulation_spec_verification": (
                    code_payload.get("simulation_spec_verification")
                    or simulation_spec_verification
                ),
                "rich_demo_plan": code_payload.get("rich_demo_plan") or prepared_rich_demo_plan,
                "grounding": grounding,
                "understanding": understanding,
                "lesson_interaction": lesson_interaction,
                "final_answer": _final_answer(
                    request,
                    result,
                    concept_answer,
                    grounding,
                    lesson_interaction=lesson_interaction,
                ),
                "llm": self._llm_status(),
            }

        code = code_payload["code"]
        code_source = code_payload["source"]
        attempts = []
        result = None

        for attempt_index in range(self.max_repair_attempts + 1):
            result = self._execute_code(request, code)
            attempts.append(_attempt_summary(attempt_index + 1, code_source, result))
            if result.get("status") == "completed":
                break
            repaired = self._repair_code(request, code, result)
            if not repaired or repaired == code:
                break
            code = repaired
            code_source = "llm_repaired"

        lesson_interaction = self.lesson_interaction_agent.reflect(
            request,
            learner_prediction,
            code_payload.get("simulation_spec") or simulation_spec,
            result or {},
        )
        return {
            "agent": "learning_demo",
            "status": result.get("status") if result else "failed",
            "request": request,
            "tool": "python_demo_runner",
            "code_source": code_source,
            "generated_code": code,
            "attempts": attempts,
            "result": result,
            "concept_answer": concept_answer,
            "simulation_spec": code_payload.get("simulation_spec") or simulation_spec,
            "simulation_spec_verification": (
                code_payload.get("simulation_spec_verification")
                or simulation_spec_verification
            ),
            "rich_demo_plan": code_payload.get("rich_demo_plan") or prepared_rich_demo_plan,
            "grounding": grounding,
            "understanding": understanding,
            "lesson_interaction": lesson_interaction,
            "final_answer": _final_answer(
                request,
                result or {},
                concept_answer,
                grounding,
                lesson_interaction=lesson_interaction,
            ),
            "llm": self._llm_status(),
        }

    def prepare_prediction(self, request, conversation_context=""):
        understanding = self.request_understanding_agent.run(request, conversation_context)
        prompt_context = conversation_context if understanding.get("uses_context") else ""
        grounding = self.grounding_service.retrieve(
            request,
            prompt_context,
            search_queries=understanding.get("search_queries"),
        )
        simulation_spec = self.simulation_spec_agent.run(
            request,
            grounding=grounding,
            conversation_context=prompt_context,
        )
        if simulation_spec.get("status") != "ready":
            if self.llm.available and _has_explicit_execution_request(request):
                rich_demo_plan = self._generate_rich_demo_plan(
                    request,
                    grounding,
                    prompt_context,
                    simulation_spec,
                )
                prediction = _prediction_from_rich_demo_plan(
                    request,
                    simulation_spec,
                    rich_demo_plan,
                )
                final_answer = _prediction_prompt_text(request, prediction, grounding)
                return {
                    "agent": "learning_demo",
                    "status": "awaiting_prediction",
                    "request": request,
                    "tool": None,
                    "attempts": [],
                    "generated_code": None,
                    "simulation_spec": simulation_spec,
                    "rich_demo_plan": rich_demo_plan,
                    "prediction": prediction,
                    "grounding": grounding,
                    "understanding": understanding,
                    "final_answer": final_answer,
                    "llm": self._llm_status(),
                }
            result = _unreliable_demo_result(
                request,
                simulation_spec.get("reason")
                or "No reliable simulation spec was available for this request.",
            )
            concept_answer = _clean_concept_answer(
                self._generate_concept_answer(request, grounding, prompt_context)
            )
            return {
                "agent": "learning_demo",
                "status": "unreliable",
                "request": request,
                "tool": "python_demo_runner",
                "attempts": [],
                "generated_code": None,
                "result": result,
                "concept_answer": concept_answer,
                "simulation_spec": simulation_spec,
                "prediction": None,
                "grounding": grounding,
                "understanding": understanding,
                "final_answer": _final_answer(
                    request,
                    result,
                    concept_answer,
                    grounding,
                ),
                "llm": self._llm_status(),
            }
        prediction = self.lesson_interaction_agent.prediction_choices(
            request,
            simulation_spec=simulation_spec,
        )
        final_answer = _prediction_prompt_text(request, prediction, grounding)
        return {
            "agent": "learning_demo",
            "status": "awaiting_prediction",
            "request": request,
            "tool": None,
            "attempts": [],
            "generated_code": None,
            "simulation_spec": simulation_spec,
            "prediction": prediction,
            "grounding": grounding,
            "understanding": understanding,
            "final_answer": final_answer,
            "llm": self._llm_status(),
        }

    def _classify_intent(self, request):
        if not _has_explicit_execution_request(request):
            return {
                "mode": "concept_explanation",
                "reason": "request does not explicitly ask for code, demo, plotting, calculation, or execution",
                "needs_code": False,
                "source": "policy_gate",
            }

        return {
            "mode": "python_demo",
            "reason": "request explicitly asks for code, Python, demo, plotting, calculation, saved output, or execution",
            "needs_code": True,
            "source": "policy_gate",
        }

    def _answer_concept(
        self,
        request,
        intent,
        grounding,
        conversation_context="",
        understanding=None,
    ):
        answer = None
        if self.llm.available:
            answer = self.llm.respond(
                _concept_prompt(request, self.agent_spec, grounding, conversation_context)
            )
        if not answer:
            answer = _fallback_concept_answer(request)

        return {
            "agent": "learning_demo",
            "status": "answered",
            "request": request,
            "mode": "concept_explanation",
            "intent": intent,
            "tool": None,
            "attempts": [],
            "generated_code": None,
            "result": {
                "status": "answered",
                "topic": request,
                "stdout": "",
                "stderr": "",
                "generated_files": [],
                "safety_notes": [
                    "No code was executed because this was classified as an explanation-only request.",
                ],
            },
            "grounding": grounding,
            "understanding": understanding or {},
            "final_answer": _with_sources(
                _clean_concept_answer(answer) or _fallback_concept_answer(request),
                grounding,
            ),
            "llm": self._llm_status(),
        }

    def _generate_code(
        self,
        request,
        grounding,
        conversation_context="",
        simulation_spec=None,
        simulation_spec_verification=None,
        rich_demo_plan=None,
    ):
        if (simulation_spec_verification or {}).get("verdict") == "fail":
            return {
                "source": "simulation_spec_rejected",
                "code": None,
                "concept_answer": None,
                "simulation_spec": simulation_spec,
                "simulation_spec_verification": simulation_spec_verification,
                "rich_demo_plan": rich_demo_plan,
                "reason": "Simulation spec rejected before execution: "
                + "; ".join((simulation_spec_verification or {}).get("issues") or []),
            }

        primitive_payload = build_demo_code(simulation_spec or {})
        if primitive_payload["status"] == "ready":
            return {
                "source": "simulation_spec_primitive",
                "code": primitive_payload["code"],
                "concept_answer": None,
                "simulation_spec": primitive_payload["spec"],
                "simulation_spec_verification": simulation_spec_verification,
                "rich_demo_plan": rich_demo_plan,
                "reason": None,
            }

        fallback = _fallback_python_demo_code(request)
        if not self.llm.available:
            fallback["simulation_spec"] = simulation_spec
            fallback["simulation_spec_verification"] = simulation_spec_verification
            return fallback

        response_text = self.llm.respond(
            _code_prompt(
                request,
                self.agent_spec,
                grounding,
                conversation_context,
                simulation_spec=simulation_spec,
                rich_demo_plan=rich_demo_plan,
            )
        )
        parsed = _extract_json(response_text) or {}
        concept_answer = _clean_concept_answer(parsed.get("concept_answer"))
        generated_code = _extract_generated_code(response_text)
        if generated_code and _looks_relevant(request, generated_code):
            return {
                "source": "llm_generated",
                "code": generated_code,
                "concept_answer": concept_answer,
                "simulation_spec": simulation_spec,
                "simulation_spec_verification": simulation_spec_verification,
                "rich_demo_plan": rich_demo_plan,
                "reason": None,
            }
        fallback["concept_answer"] = concept_answer
        fallback["simulation_spec"] = simulation_spec
        fallback["simulation_spec_verification"] = simulation_spec_verification
        fallback["rich_demo_plan"] = rich_demo_plan
        return fallback

    def _generate_concept_answer(self, request, grounding, conversation_context=""):
        if self.llm.available:
            answer = self.llm.respond(
                _concept_prompt(request, self.agent_spec, grounding, conversation_context)
            )
            if answer:
                return _clean_concept_answer(answer)
        return _fallback_concept_answer(request)

    def _generate_rich_demo_plan(self, request, grounding, conversation_context, simulation_spec):
        if self.llm.available:
            parsed = _extract_json(
                self.llm.respond(
                    _rich_demo_plan_prompt(
                        request,
                        grounding,
                        conversation_context,
                        simulation_spec,
                    )
                )
            )
            plan = _normalize_rich_demo_plan(parsed, request, simulation_spec)
            if plan:
                return plan
        return _fallback_rich_demo_plan(request, simulation_spec)

    def _repair_code(self, request, code, result):
        if not self.llm.available:
            return None
        return _extract_generated_code(self.llm.respond(_repair_prompt(request, code, result, self.agent_spec)))

    def _execute_code(self, request, code):
        result = self.tool_loader.execute(
            "python_demo_runner",
            {"topic": request, "code": code},
        )
        if result.get("status") == "completed" and not _has_expected_output(request, result):
            result = dict(result)
            result["status"] = "failed"
            result["stderr"] = (
                result.get("stderr") or ""
            ) + "Code ran but produced no stdout or generated files."
        return result

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


def _load_agent_spec():
    spec_path = Path("agents/physics_learning/system_prompt.md")
    if spec_path.exists():
        return spec_path.read_text(encoding="utf-8")
    return (
        "You are a physics learning agent. Explain concepts first, then create "
        "small runnable Python demos only when the user asks for code."
    )


def _concept_prompt(request, agent_spec, grounding, conversation_context=""):
    return f"""
{agent_spec}

Answer the user's question directly and helpfully. If there is a spelling
mistake, infer the intended physics term. Do not include Python code, code
blocks, or pseudocode; runnable demos are handled by a separate trusted tool.
Keep the answer concise but useful.
If the current request asks for an equation or formula, put the equation first,
then define each symbol briefly.

Use the local evidence when it is relevant. If the grounding status is weak or
missing, say that the local source evidence is limited and separate that from
general model knowledge. When using evidence, cite it with source/page/chunk.

User request:
{request}

Conversation context:
{conversation_context or "(none)"}

Grounding:
{json.dumps(grounding, indent=2)}
""".strip()


def _code_prompt(
    request,
    agent_spec,
    grounding,
    conversation_context="",
    simulation_spec=None,
    rich_demo_plan=None,
):
    return f"""
{agent_spec}

Create a learning response for this request. Return only valid JSON with:
- concept_answer: a concise teaching explanation with no code blocks
- demo_plan: short explanation of what the code will demonstrate
- code: runnable Python code, or null if you are not confident

Rules:
- code must be self-contained
- use only math, random, statistics, numpy, matplotlib, collections, itertools
- print a short numerical summary
- printed output is automatically saved by the runner as stdout.txt
- if plotting, save to demo.png
- do not read or write files directly
- do not use network, subprocess, os, sys, pathlib, open, eval, or exec
- for arithmetic requests, print the computed numeric answer, not code text
- if you cannot produce correct topic-specific code, set code to null
- do not return placeholder code
- do not put code inside concept_answer
- use the local evidence to choose the equation or demonstration when available
- if local evidence is weak or missing and you are uncertain, set code to null
- prefer the simulation_spec when it is ready
- if simulation_spec is unavailable, generate code only when you can still make a correct, topic-specific demo
- if rich_demo_plan is provided, treat it as the contract for the demo
- do not ask whether the user wants the demo; this request is already approved for execution
- save every plot requested in rich_demo_plan
- do not call plt.show()

Learning request:
{request}

Conversation context:
{conversation_context or "(none)"}

Grounding:
{json.dumps(grounding, indent=2)}

Simulation spec:
{json.dumps(simulation_spec or {}, indent=2)}

Rich demo plan:
{json.dumps(rich_demo_plan or {}, indent=2)}

Example JSON:
{{
  "concept_answer": "Multiplication combines equal groups. Here the requested calculation is 3 times 5.",
  "demo_plan": "Compute the product and print the numeric result.",
  "code": "result = 3 * 5\\nprint(f'3 * 5 = {{result}}')"
}}
""".strip()


def _rich_demo_plan_prompt(request, grounding, conversation_context="", simulation_spec=None):
    return f"""
Create a structured plan for an interactive scientific Python demo.

Return only valid JSON with:
{{
  "learning_goal": "what the learner should understand",
  "simulation_model": "short model description",
  "prediction_question": "one question to ask before running the demo",
  "prediction_options": [
    {{"id": "A", "text": "option text", "is_expected": true}},
    {{"id": "B", "text": "option text", "is_expected": false}},
    {{"id": "C", "text": "option text", "is_expected": false}}
  ],
  "expected_behavior": "what the plots/stdout should show",
  "plots": [
    {{"filename": "plot_name.png", "description": "what this plot reveals"}}
  ],
  "parameters": {{"short_name": "value or meaning"}},
  "code_requirements": [
    "specific runnable-code requirement"
  ],
  "result_interpretation_goals": [
    "what the final explanation should connect back to"
  ]
}}

Rules:
- Do not write Python code here.
- Keep the plan specific enough that a code generator can implement it.
- Use only local file outputs such as PNG plots and printed stdout.
- Prefer simple robust simulations over fragile advanced numerics.
- Include exactly three prediction options.

User request:
{request}

Conversation context:
{conversation_context or "(none)"}

Simulation spec:
{json.dumps(simulation_spec or {}, indent=2)}

Grounding:
{json.dumps(grounding, indent=2)}
""".strip()


def _attempt_summary(attempt_number, code_source, result):
    return {
        "attempt": attempt_number,
        "code_source": code_source,
        "status": result.get("status"),
        "returncode": result.get("returncode"),
        "stderr": result.get("stderr"),
        "safety_errors": result.get("safety_errors"),
    }


def _repair_prompt(request, code, result, agent_spec):
    return f"""
{agent_spec}

The Python demo failed or was rejected. Repair the code while following the
safety rules. Return only JSON with "code".

Important:
- do not use open()
- do not write files directly
- print the output; the runner automatically saves stdout as stdout.txt

Learning request:
{request}

Original code:
{code}

Tool result:
{json.dumps(result, indent=2)}
""".strip()


def _fallback_python_demo_code(request):
    return {
        "source": "unavailable",
        "code": None,
        "reason": "No reliable generated code or reusable simulation primitive is available for this request.",
    }


def _has_explicit_execution_request(request):
    lower_request = request.lower()
    markers = [
        "python",
        "code",
        "demo",
        "demonstrate",
        "plot",
        "simulate",
        "simulation",
        "calculate",
        "compute",
        "run",
        "save",
        "output",
        "script",
    ]
    return any(marker in lower_request for marker in markers)


def _fallback_concept_answer(request):
    return (
        f"Request: {request}\n\n"
        "I could not get a model-generated explanation for this request in the "
        "current run. I will avoid inventing a detailed lesson. If you asked for "
        "a demo, I will only run it when the request maps to a reliable verified "
        "demo spec or trusted code path."
    )


def _clean_concept_answer(text):
    if not text:
        return None

    cleaned_lines = []
    in_code_block = False
    for line in str(text).splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if ("python" in lower or "code" in lower or "demo" in lower) and lower.startswith(
            (
                "to demonstrate",
                "here is",
                "here's",
                "please let me know",
                "now, i",
                "now i",
                "we can use",
                "this code",
                "the code",
            )
        ):
            break
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned or None


def _final_answer(
    request,
    result,
    concept_answer=None,
    grounding=None,
    lesson_interaction=None,
):
    concept_answer = (concept_answer or "").strip()
    if result.get("status") == "unreliable":
        lines = [f"Request: {request}"]
        if concept_answer:
            lines.extend(["", "Concept explanation:", concept_answer])
        lines.extend(
            [
                "",
                "Python demo status:",
                "I could not produce a reliable runnable Python demo for this concept yet. "
                "A wrong or unrelated plot would be misleading, so no demo was run.",
                "",
                f"Reason: {result.get('reason')}",
            ]
        )
        _add_lesson_interaction(lines, lesson_interaction)
        return _with_sources("\n".join(lines), grounding)
    if result.get("status") == "rejected":
        return _with_sources(
            _final_answer_with_demo_issue(
                request,
                concept_answer,
                "The generated code was rejected by the safety checker, so it was not run.",
                f"Safety errors: {result.get('safety_errors')}",
            ),
            grounding,
        )
    if result.get("status") == "failed":
        return _with_sources(
            _final_answer_with_demo_issue(
                request,
                concept_answer,
                "The generated code ran but failed. No successful demo result was produced.",
                f"stderr: {result.get('stderr')}",
            ),
            grounding,
        )

    stdout = (result.get("stdout") or "").strip()
    files = result.get("generated_files") or []
    lines = [f"Request: {request}"]
    if concept_answer:
        lines.extend(["", "Concept explanation:", concept_answer])
    else:
        lines.extend(["", *_concept_lines(request)])
    lines.extend(["", "Python demo output:", stdout or "(no stdout captured)"])
    if files:
        lines.extend(["", "Generated files:", *[f"- {path}" for path in files]])
    lines.extend(["", "This was run through the `python_demo_runner` plugin tool."])
    _add_lesson_interaction(lines, lesson_interaction)
    return _with_sources("\n".join(lines), grounding)


def _final_answer_with_demo_issue(request, concept_answer, issue, details):
    lines = [f"Request: {request}"]
    if concept_answer:
        lines.extend(["", "Concept explanation:", concept_answer])
    lines.extend(["", "Python demo status:", issue, "", details])
    return "\n".join(lines)


def _with_sources(text, grounding):
    if not grounding:
        return text

    status = grounding.get("status", "missing")
    reason = grounding.get("reason", "")
    lines = [text.rstrip(), "", f"Grounding status: {status}"]
    if reason:
        lines.append(reason)

    sources = grounding.get("sources") or []
    if sources:
        lines.append("")
        lines.append("Local sources:")
        for source in sources[:4]:
            lines.append(
                f"- {source['source']} page {source.get('page')} chunk {source['chunk_id']} "
                f"(score {source['score']})"
            )
    return "\n".join(lines)


def _prediction_prompt_text(request, prediction, grounding=None):
    lines = [
        f"Request: {request}",
        "",
        prediction.get("question") or "Before I run the demo, choose your prediction:",
    ]
    for option in prediction.get("options") or []:
        lines.append(f"{option.get('id')}. {option.get('text')}")
    reason = prediction.get("reason")
    if reason:
        lines.extend(["", f"Why I am asking: {reason}"])
    lines.extend(
        [
            "",
            "Reply with A, B, C, or your own prediction. I will run the demo after that.",
        ]
    )
    return _with_sources("\n".join(lines), grounding)


def _normalize_rich_demo_plan(parsed, request, simulation_spec):
    if not isinstance(parsed, dict):
        return None

    plan = {
        "learning_goal": _short_plan_text(
            parsed.get("learning_goal"),
            f"Understand {request} through a runnable simulation.",
        ),
        "simulation_model": _short_plan_text(
            parsed.get("simulation_model"),
            (simulation_spec or {}).get("reason") or "Custom scientific simulation.",
        ),
        "prediction_question": _short_plan_text(
            parsed.get("prediction_question"),
            "Before running the demo, what trend do you expect to see?",
        ),
        "prediction_options": _normalize_prediction_options(parsed.get("prediction_options")),
        "expected_behavior": _short_plan_text(
            parsed.get("expected_behavior"),
            (simulation_spec or {}).get("expected_behavior")
            or "The simulation output should reveal the requested physical behavior.",
        ),
        "plots": _normalize_plots(parsed.get("plots")),
        "parameters": parsed.get("parameters") if isinstance(parsed.get("parameters"), dict) else {},
        "code_requirements": _string_list(parsed.get("code_requirements")),
        "result_interpretation_goals": _string_list(parsed.get("result_interpretation_goals")),
        "source": "llm",
    }
    if not plan["prediction_options"]:
        plan["prediction_options"] = _fallback_prediction_options()
    if not plan["plots"]:
        plan["plots"] = [{"filename": "demo.png", "description": "Main demo result."}]
    return plan


def _fallback_rich_demo_plan(request, simulation_spec):
    return {
        "learning_goal": f"Understand {request} through a runnable simulation.",
        "simulation_model": (simulation_spec or {}).get("reason") or "Custom scientific simulation.",
        "prediction_question": "Before running the demo, what trend do you expect to see?",
        "prediction_options": _fallback_prediction_options(),
        "expected_behavior": (
            (simulation_spec or {}).get("expected_behavior")
            or "The simulation should evolve toward the requested physical pattern."
        ),
        "plots": [{"filename": "demo.png", "description": "Main demo result."}],
        "parameters": {},
        "code_requirements": [
            "Use only allowed imports.",
            "Print a concise numerical summary.",
            "Save plots as PNG files.",
            "Do not call plt.show().",
        ],
        "result_interpretation_goals": [
            "Connect the plotted trend to the learner prediction.",
        ],
        "source": "fallback",
    }


def _prediction_from_rich_demo_plan(request, simulation_spec, rich_demo_plan):
    options = _normalize_prediction_options((rich_demo_plan or {}).get("prediction_options"))
    if not options:
        options = _fallback_prediction_options()
    return {
        "question": (
            (rich_demo_plan or {}).get("prediction_question")
            or "Before I generate and run the rich demo, choose your prediction:"
        ),
        "options": options,
        "reason": (
            (rich_demo_plan or {}).get("expected_behavior")
            or (simulation_spec or {}).get("expected_behavior")
            or (simulation_spec or {}).get("reason")
            or "the system should evolve toward the pattern described by the demo"
        ),
        "source": (rich_demo_plan or {}).get("source") or "rich_demo_plan",
    }


def _custom_demo_prediction_choices(request, simulation_spec):
    behavior = (
        (simulation_spec or {}).get("expected_behavior")
        or (simulation_spec or {}).get("reason")
        or "the system should evolve toward the pattern described by the demo"
    )
    return {
        "question": "Before I generate and run the rich demo, choose your prediction:",
        "options": [
            {
                "id": "A",
                "text": "The system relaxes toward a stable pattern rather than staying in its initial state.",
                "is_expected": True,
            },
            {
                "id": "B",
                "text": "The system remains essentially unchanged throughout the simulation.",
                "is_expected": False,
            },
            {
                "id": "C",
                "text": "The output becomes purely random with no interpretable trend.",
                "is_expected": False,
            },
        ],
        "reason": behavior,
        "source": "custom_demo_policy",
    }


def _normalize_prediction_options(raw_options):
    if not isinstance(raw_options, list):
        return []
    options = []
    for index, option in enumerate(raw_options[:3]):
        if isinstance(option, dict):
            text = str(option.get("text") or "").strip()
            option_id = str(option.get("id") or chr(ord("A") + index)).strip().upper()[:1]
            is_expected = bool(option.get("is_expected"))
        else:
            text = str(option or "").strip()
            option_id = chr(ord("A") + index)
            is_expected = index == 0
        if text:
            options.append({"id": option_id, "text": text, "is_expected": is_expected})
    if options and not any(option.get("is_expected") for option in options):
        options[0]["is_expected"] = True
    return options


def _fallback_prediction_options():
    return [
        {
            "id": "A",
            "text": "The system relaxes toward a stable pattern rather than staying in its initial state.",
            "is_expected": True,
        },
        {
            "id": "B",
            "text": "The system remains essentially unchanged throughout the simulation.",
            "is_expected": False,
        },
        {
            "id": "C",
            "text": "The output becomes purely random with no interpretable trend.",
            "is_expected": False,
        },
    ]


def _normalize_plots(raw_plots):
    if not isinstance(raw_plots, list):
        return []
    plots = []
    for plot in raw_plots[:6]:
        if isinstance(plot, dict):
            filename = str(plot.get("filename") or "").strip()
            description = str(plot.get("description") or "").strip()
        else:
            filename = str(plot or "").strip()
            description = ""
        if filename:
            plots.append({"filename": filename, "description": description})
    return plots


def _string_list(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _short_plan_text(value, fallback):
    text = str(value or "").strip()
    return text[:1200] if text else fallback


def _add_lesson_interaction(lines, lesson_interaction):
    if not lesson_interaction:
        return
    lines.extend(["", "Learning interaction:"])
    prediction = lesson_interaction.get("learner_prediction")
    if prediction:
        lines.append(f"Prediction: {prediction}")
        lines.append(f"Comparison: {lesson_interaction.get('comparison')}")
        feedback = lesson_interaction.get("feedback")
        if feedback:
            lines.append(f"Feedback: {feedback}")
    else:
        question = lesson_interaction.get("prediction_question")
        if question:
            lines.append(f"Prediction prompt: {question}")
    follow_up = lesson_interaction.get("follow_up_question")
    if follow_up:
        lines.append(f"Follow-up question: {follow_up}")


def _concept_lines(request):
    lower_request = request.lower()
    if "multiply" in lower_request:
        return [
            "This request was handled as a small safe Python calculation.",
            "The runner captures printed output and saves it as `stdout.txt`.",
        ]
    return [
        "This learning request was handled by generating and running a small",
        "Python demonstration.",
    ]


def _has_expected_output(request, result):
    lower_request = request.lower()
    expected_text = _expected_numeric_result_text(request)
    if expected_text and expected_text not in (result.get("stdout") or ""):
        return False
    if any(marker in lower_request for marker in ["output", "save", "print", "plot", "demo"]):
        return bool((result.get("stdout") or "").strip() or result.get("generated_files"))
    return True


def _expected_numeric_result_text(request):
    lower_request = request.lower()
    numbers = _number_terms(request)
    if "multiply" in lower_request and len(numbers) >= 2:
        product = 1.0
        for number in numbers[:2]:
            product *= float(number)
        if product.is_integer():
            return str(int(product))
        return str(product)
    return None


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


def _extract_generated_code(text):
    parsed = _extract_json(text)
    generated_code = (parsed or {}).get("code")
    if isinstance(generated_code, list):
        generated_code = "\n".join(str(line) for line in generated_code)
    if generated_code is not None and not isinstance(generated_code, str):
        generated_code = str(generated_code)
    if generated_code:
        return generated_code
    loose_code = _extract_loose_code_field(text)
    if loose_code:
        return loose_code
    return _extract_fenced_code(text)


def _extract_fenced_code(text):
    if not text or "```" not in text:
        return None
    parts = text.split("```")
    for index in range(1, len(parts), 2):
        block = parts[index].strip()
        if block.startswith("python"):
            block = block[len("python") :].strip()
        elif block.startswith("json"):
            block = _extract_loose_code_field(block) or ""
        if block:
            return block
    return None


def _extract_loose_code_field(text):
    if not text or '"code"' not in text:
        return None

    marker_index = text.find('"code"')
    colon_index = text.find(":", marker_index)
    if colon_index < 0:
        return None

    value = text[colon_index + 1 :].lstrip()
    if not value.startswith('"'):
        return None

    value = value[1:]
    end_markers = ['"\n}', '"\n,', '"\r\n}', '"\r\n,']
    end_positions = [value.find(marker) for marker in end_markers]
    end_positions = [position for position in end_positions if position >= 0]
    if not end_positions:
        return None

    raw_code = value[: min(end_positions)]
    try:
        decoded = json.loads(f'"{raw_code}"')
    except json.JSONDecodeError:
        decoded = raw_code
    return decoded.strip() or None


def _looks_relevant(request, code):
    request_terms = _important_terms(request)
    request_numbers = _number_terms(request)
    code_lower = code.lower()
    if request_numbers and all(number in code_lower for number in request_numbers):
        return True
    if not request_terms:
        return True
    return any(term in code_lower for term in request_terms)


def _important_terms(request):
    stopwords = {
        "with",
        "python",
        "demo",
        "teach",
        "learn",
        "explain",
        "show",
        "effect",
        "short",
        "code",
    }
    terms = []
    for raw in request.lower().replace("-", " ").split():
        term = "".join(ch for ch in raw if ch.isalnum())
        if len(term) >= 5 and term not in stopwords:
            terms.append(term)
    return terms


def _number_terms(request):
    numbers = []
    for raw in request.lower().replace("-", " ").split():
        term = "".join(ch for ch in raw if ch.isdigit() or ch == ".")
        if term and any(ch.isdigit() for ch in term):
            numbers.append(term)
    return numbers


def _unreliable_demo_result(request, reason):
    return {
        "status": "unreliable",
        "topic": request,
        "reason": reason,
        "stdout": "",
        "stderr": "",
        "generated_files": [],
        "safety_notes": [
            "No code was executed because the agent could not produce a reliable topic-specific demo.",
        ],
    }
