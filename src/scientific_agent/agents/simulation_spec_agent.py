import json
from pathlib import Path

from src.scientific_agent.core.demo_primitives import normalize_simulation_spec
from src.scientific_agent.core.llm import LLMClient


class SimulationSpecAgent:
    """Ask the model for a structured, reusable simulation/demo specification."""

    def __init__(self, provider="ollama", model=None):
        self.llm = LLMClient(provider=provider, model=model)
        self.system_prompt = _load_system_prompt()

    def run(self, request, grounding=None, conversation_context=""):
        if not self.llm.available:
            return _unavailable("LLM is not available for simulation specification.")

        parsed = _extract_json(
            self.llm.respond(
                _spec_prompt(
                    self.system_prompt,
                    request,
                    grounding or {},
                    conversation_context,
                )
            )
        )
        if not parsed:
            return _fallback_spec_from_request(
                request,
                grounding or {},
                "Simulation spec model output was not valid JSON.",
            )

        normalized = normalize_simulation_spec(parsed)
        if normalized.get("status") != "ready" and _looks_like_schema_mismatch(parsed):
            return _fallback_spec_from_request(
                request,
                grounding or {},
                "Simulation spec model output was JSON but did not match the required schema.",
            )
        normalized["source"] = "llm"
        return normalized


def _load_system_prompt():
    prompt_path = Path("agents/simulation_spec/system_prompt.md")
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return "Convert learning requests into reusable simulation specs. Return JSON only."


def _spec_prompt(system_prompt, request, grounding, conversation_context=""):
    return f"""
{system_prompt}

Return only valid JSON with this schema:
{{
  "status": "ready" or "unavailable",
  "concept": "short concept name",
  "demo_type": "relation_plot | random_walk | distribution | time_evolution | multi_series_time_evolution | phase_space",
  "relation_family": "linear | quadratic | inverse_square | threshold_linear | sinusoidal | null",
  "parameters": {{
    "coefficient": number,
    "slope": number,
    "intercept": number,
    "threshold": number,
    "amplitude": number,
    "frequency": number,
    "angular_frequency": number,
    "phase": number,
    "steps": number,
    "walkers": number,
    "mean": number,
    "std": number,
    "samples": number
    "total": number,
    "exchange_amplitude": number
  }},
  "x_range": [number, number],
  "x_label": "short label",
  "y_label": "short label",
  "title": "short title",
  "equation_text": "human-readable equation, if known",
  "expected_behavior": "what the plot or simulation should show",
  "reason": "short reason"
}}

Important rules:
- Do not write Python code.
- Use a reusable primitive, not a topic-specific script.
- If a primitive cannot represent the concept correctly, use status "unavailable".
- For relation_plot, choose a relation_family from the allowed set.
- Keep numeric parameters simple and finite.
- Use the local evidence when it is relevant.

Examples:
{{
  "status": "ready",
  "concept": "threshold linear relation",
  "demo_type": "relation_plot",
  "relation_family": "threshold_linear",
  "parameters": {{"slope": 1.0, "threshold": 2.0, "intercept": 0.0}},
  "x_range": [0.0, 6.0],
  "x_label": "driving variable",
  "y_label": "response",
  "title": "Threshold-linear response",
  "equation_text": "y = max(0, m(x - x0))",
  "expected_behavior": "response is zero below threshold and linear above it",
  "reason": "the requested concept can be represented by a threshold relation"
}}

{{
  "status": "ready",
  "concept": "diffusive spreading",
  "demo_type": "random_walk",
  "relation_family": null,
  "parameters": {{"walkers": 1000, "steps": 300}},
  "x_range": [0.0, 300.0],
  "x_label": "step",
  "y_label": "mean squared displacement",
  "title": "Random walk spreading",
  "equation_text": "MSD proportional to time",
  "expected_behavior": "mean squared displacement grows approximately linearly",
  "reason": "random walk is a reusable primitive for diffusion-like spreading"
}}

{{
  "status": "ready",
  "concept": "coupled exchange with conserved total",
  "demo_type": "multi_series_time_evolution",
  "relation_family": null,
  "parameters": {{"total": 10.0, "exchange_amplitude": 4.0, "angular_frequency": 1.0}},
  "x_range": [0.0, 10.0],
  "x_label": "time",
  "y_label": "quantity",
  "title": "Exchange between two components with constant total",
  "equation_text": "component A + component B = constant",
  "expected_behavior": "one component decreases while the other increases and the total remains constant",
  "reason": "multi-series evolution is appropriate when the learning goal is a conserved total"
}}

Conversation context:
{conversation_context or "(none)"}

User request:
{request}

Grounding:
{json.dumps(_compact_grounding(grounding), indent=2)}
""".strip()


def _compact_grounding(grounding):
    sources = []
    for source in (grounding.get("sources") or [])[:4]:
        sources.append(
            {
                "score": source.get("score"),
                "source": source.get("source"),
                "page": source.get("page"),
                "chunk_id": source.get("chunk_id"),
                "text": (source.get("text") or "")[:600],
            }
        )
    return {
        "status": grounding.get("status"),
        "reason": grounding.get("reason"),
        "sources": sources,
    }


def _fallback_spec_from_request(request, grounding, reason):
    """Build a generic primitive spec when the small model fails formatting.

    This is intentionally primitive-based rather than topic-script-based: the
    fallback describes the requested phenomenon, then demo_primitives decides
    whether any reusable simulation form can safely represent it.
    """

    evidence_text = " ".join(
        (source.get("text") or "")[:500]
        for source in (grounding.get("sources") or [])[:3]
    )
    fields = _fallback_demo_fields(request, evidence_text)
    if not fields.get("demo_type"):
        return {
            "status": "unavailable",
            "concept": fields["concept"],
            "demo_type": None,
            "relation_family": None,
            "parameters": {},
            "x_range": [0.0, 10.0],
            "x_label": fields["x_label"],
            "y_label": fields["y_label"],
            "title": fields["title"],
            "equation_text": fields["equation_text"],
            "expected_behavior": fields["expected_behavior"],
            "reason": (
                "No safe fallback demo primitive matched the request after the "
                f"model returned an unusable simulation spec: {reason}"
            ),
            "source": "heuristic_fallback_from_request",
            "model_error": reason,
        }
    normalized = normalize_simulation_spec(
        {
            "status": "ready",
            "concept": fields["concept"],
            "demo_type": fields["demo_type"],
            "relation_family": None,
            "parameters": fields["parameters"],
            "x_range": [0.0, 10.0],
            "x_label": fields["x_label"],
            "y_label": fields["y_label"],
            "title": fields["title"],
            "equation_text": fields["equation_text"],
            "expected_behavior": fields["expected_behavior"],
            "reason": (
                "Fallback spec built from the request and retrieved evidence "
                f"because: {reason}"
            ),
        }
    )
    normalized["source"] = "heuristic_fallback_from_request"
    if normalized["status"] != "ready":
        normalized["model_error"] = reason
    return normalized


def _looks_like_schema_mismatch(parsed):
    if not isinstance(parsed, dict):
        return True
    required_signal = {"status", "demo_type", "concept"}
    if required_signal & set(parsed):
        return False
    return True


def _fallback_demo_fields(request, evidence_text):
    concept = _short_concept(request)
    request_text = str(request or "").lower()
    if "random walk" in request_text or "diffusion" in request_text or "diffusive" in request_text:
        return {
            "concept": concept,
            "demo_type": "random_walk",
            "parameters": {
                "walkers": 1000,
                "steps": 300,
            },
            "x_label": "step",
            "y_label": "mean squared displacement",
            "title": _short_title(request),
            "equation_text": "mean squared displacement grows approximately linearly with time",
            "expected_behavior": (
                f"{concept}: a collection of random walkers spreads out over time, "
                "so the mean squared displacement increases with step number."
            ),
        }

    quantity = _conserved_quantity(request)
    if quantity:
        quantity = quantity.replace("mechanical energy", "energy").strip()
        return {
            "concept": concept,
            "demo_type": "multi_series_time_evolution",
            "parameters": {
                "total": 10.0,
                "exchange_amplitude": 4.0,
                "angular_frequency": 1.0,
            },
            "x_label": "time",
            "y_label": quantity,
            "title": _short_title(request),
            "equation_text": (
                f"component A {quantity} + component B {quantity} = total {quantity}"
            ),
            "expected_behavior": (
                f"{concept}: one component of {quantity} decreases while another "
                f"increases, and the total {quantity} remains constant. "
                f"{evidence_text[:500]}"
            ),
        }

    return {
        "concept": concept,
        "demo_type": None,
        "parameters": {},
        "x_label": "time",
        "y_label": "quantity",
        "title": _short_title(request),
        "equation_text": "",
        "expected_behavior": request,
    }


def _conserved_quantity(text):
    lowered = str(text or "").lower()
    marker = "conservation of "
    index = lowered.find(marker)
    if index >= 0:
        tail = lowered[index + len(marker) :]
        words = []
        for word in tail.replace("-", " ").split():
            clean = "".join(char for char in word if char.isalpha())
            if not clean or clean in {"and", "in", "with", "for", "the", "a", "an"}:
                break
            words.append(clean)
            if len(words) >= 2:
                break
        if words:
            return " ".join(words)

    for quantity in [
        "mechanical energy",
        "energy",
        "momentum",
        "charge",
        "mass",
    ]:
        if quantity in lowered and any(
            marker in lowered for marker in ["conserved", "constant", "remains"]
        ):
            return quantity
    return None


def _short_concept(request):
    text = " ".join(str(request or "").split())
    prefixes = ["teach me", "explain", "demonstrate", "show me"]
    lowered = text.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            text = text[len(prefix) :].strip()
            break
    for marker in [" with ", " using ", " in python"]:
        index = text.lower().find(marker)
        if index > 0:
            text = text[:index].strip()
    return text[:80] or "scientific concept"


def _short_title(request):
    concept = _short_concept(request)
    if not concept:
        return "Scientific demo"
    return concept[:1].upper() + concept[1:80]


def _unavailable(reason):
    return {
        "status": "unavailable",
        "reason": reason,
        "source": "fallback_rules",
    }


def _extract_json(text):
    if not text:
        return None
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None
