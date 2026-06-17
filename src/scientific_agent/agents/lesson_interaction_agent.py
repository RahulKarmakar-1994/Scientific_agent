import json
import re

from src.scientific_agent.core.llm import LLMClient


class LessonInteractionAgent:
    """Manage prediction-before-demo and reflection-after-demo pedagogy."""

    def __init__(self, provider="ollama", model=None):
        self.llm = LLMClient(provider=provider, model=model)

    def prediction_question(self, request, simulation_spec=None):
        expected_behavior = _expected_behavior(simulation_spec)
        if expected_behavior:
            return (
                "Before running the demo, predict what you expect to see. "
                f"Focus on this behavior: {expected_behavior}"
            )
        return (
            "Before running the demo, predict what should happen in the plot or "
            "numerical output. Which quantity should increase, decrease, stay "
            "constant, or oscillate?"
        )

    def reflect(self, request, learner_prediction, simulation_spec=None, tool_result=None):
        prediction = str(learner_prediction or "").strip()
        if not prediction:
            return {
                "status": "prediction_missing",
                "prediction_question": self.prediction_question(request, simulation_spec),
                "learner_prediction": None,
                "comparison": "not_provided",
                "feedback": (
                    "No learner prediction was provided before the demo. In interactive "
                    "mode, ask for a prediction first so the result can be used as feedback."
                ),
                "follow_up_question": _fallback_follow_up(request, simulation_spec),
                "source": "policy",
            }

        if (tool_result or {}).get("status") != "completed":
            return {
                "status": "demo_not_completed",
                "learner_prediction": prediction,
                "comparison": "not_compared",
                "feedback": (
                    "I saved your prediction, but the demo did not complete, so it "
                    "would be misleading to mark the prediction right or wrong."
                ),
                "follow_up_question": _fallback_follow_up(request, simulation_spec),
                "source": "policy",
            }

        expected_behavior = _expected_behavior(simulation_spec)
        stdout = (tool_result or {}).get("stdout") or ""
        deterministic = _fallback_reflection(
            request,
            prediction,
            expected_behavior,
            stdout,
            simulation_spec,
        )
        if deterministic.get("comparison") == "matches":
            return deterministic

        if self.llm.available:
            parsed = _extract_json(
                self.llm.respond(
                    _reflection_prompt(
                        request,
                        prediction,
                        simulation_spec or {},
                        stdout,
                    )
                )
            )
            if parsed:
                return _normalize_reflection(parsed, prediction)

        return deterministic


def _reflection_prompt(request, prediction, simulation_spec, stdout):
    return f"""
You are a physics tutor comparing a learner prediction with a demo result.

Return only valid JSON:
{{
  "comparison": "matches" | "partly_matches" | "does_not_match" | "unclear",
  "feedback": "2-4 sentence feedback that connects the prediction to the result",
  "follow_up_question": "one short conceptual question for the learner"
}}

Rules:
- Do not invent new demo results.
- Use the simulation spec and stdout as the evidence.
- Be encouraging but precise.

User request:
{request}

Learner prediction:
{prediction}

Simulation spec:
{json.dumps(simulation_spec, indent=2)}

Demo stdout:
{stdout}
""".strip()


def _normalize_reflection(parsed, prediction):
    comparison = str(parsed.get("comparison") or "unclear").lower()
    if comparison not in {"matches", "partly_matches", "does_not_match", "unclear"}:
        comparison = "unclear"
    feedback = str(parsed.get("feedback") or "").strip()
    follow_up = str(parsed.get("follow_up_question") or "").strip()
    return {
        "status": "completed",
        "learner_prediction": prediction,
        "comparison": comparison,
        "feedback": feedback or "The demo result has been compared with your prediction.",
        "follow_up_question": follow_up or "What assumption in the demo matters most for this result?",
        "source": "llm",
    }


def _fallback_reflection(request, prediction, expected_behavior, stdout, simulation_spec):
    prediction_terms = _terms(prediction)
    evidence_text = f"{expected_behavior} {stdout}"
    evidence_terms = _terms(evidence_text)
    overlap = prediction_terms & evidence_terms

    comparison = "unclear"
    if _predicts_constant(prediction) and _shows_constant_total(stdout):
        comparison = "matches"
    elif overlap and len(overlap) >= min(2, len(prediction_terms)):
        comparison = "matches"
    elif overlap:
        comparison = "partly_matches"
    elif prediction_terms:
        comparison = "does_not_match"

    feedback = _feedback_text(comparison, expected_behavior, stdout)
    return {
        "status": "completed",
        "learner_prediction": prediction,
        "comparison": comparison,
        "feedback": feedback,
        "follow_up_question": _fallback_follow_up(request, simulation_spec),
        "source": "fallback_rules",
    }


def _feedback_text(comparison, expected_behavior, stdout):
    result_hint = _stdout_hint(stdout)
    if comparison == "matches":
        lead = "Your prediction matches the main result."
    elif comparison == "partly_matches":
        lead = "Your prediction captures part of the result, but not all of it."
    elif comparison == "does_not_match":
        lead = "Your prediction does not match the main result of this demo."
    else:
        lead = "Your prediction was hard to compare directly with the demo output."

    details = expected_behavior or result_hint
    if result_hint and result_hint not in details:
        details = f"{details} {result_hint}".strip()
    return f"{lead} {details}".strip()


def _fallback_follow_up(request, simulation_spec):
    concept = (simulation_spec or {}).get("concept") or _short_request(request)
    demo_type = (simulation_spec or {}).get("demo_type")
    if demo_type == "multi_series_time_evolution":
        return (
            f"For {concept}, what physical effect would break the constant-total "
            "behavior shown in the demo?"
        )
    if demo_type == "random_walk":
        return f"For {concept}, why does averaging many walkers make the trend clearer?"
    if demo_type == "relation_plot":
        return f"For {concept}, what would change if the main parameter were doubled?"
    return f"What assumption in this {concept} demo is most important for the result?"


def _expected_behavior(simulation_spec):
    if not simulation_spec:
        return ""
    pieces = [
        simulation_spec.get("expected_behavior"),
        simulation_spec.get("equation_text"),
    ]
    return " ".join(str(piece).strip() for piece in pieces if piece).strip()


def _stdout_hint(stdout):
    stdout = str(stdout or "")
    for line in stdout.splitlines():
        if "total range" in line.lower():
            return f"The output reports {line.strip()}, so the total stayed constant."
    return ""


def _predicts_constant(text):
    lowered = str(text or "").lower()
    return any(marker in lowered for marker in ["constant", "same", "conserved", "stay"])


def _shows_constant_total(stdout):
    match = re.search(r"total range\s*=\s*([-+0-9.eE]+)\s+to\s+([-+0-9.eE]+)", str(stdout))
    if not match:
        return False
    try:
        start = float(match.group(1))
        end = float(match.group(2))
    except ValueError:
        return False
    return abs(start - end) < 1e-9


def _terms(text):
    stopwords = {
        "and",
        "but",
        "demo",
        "does",
        "for",
        "from",
        "plot",
        "result",
        "should",
        "that",
        "the",
        "this",
        "will",
        "with",
    }
    return {
        term
        for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_]+", str(text).lower())
        if len(term) > 2 and term not in stopwords
    }


def _short_request(request):
    text = " ".join(str(request or "scientific concept").split())
    return text[:80]


def _extract_json(text):
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None
