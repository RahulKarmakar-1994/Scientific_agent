import json
import re
from pathlib import Path

from src.scientific_agent.core.demo_primitives import normalize_simulation_spec
from src.scientific_agent.core.llm import LLMClient


class SimulationSpecVerifierAgent:
    """Validate a simulation spec before code generation."""

    def __init__(self, provider="ollama", model=None):
        self.llm = LLMClient(provider=provider, model=model)
        self.system_prompt = _load_system_prompt()

    def run(self, request, simulation_spec, grounding=None):
        structural = _structural_verification(request, simulation_spec or {})
        if structural["verdict"] in {"fail", "skip"}:
            return structural

        if not self.llm.available:
            return structural

        parsed = _extract_json(
            self.llm.respond(
                _verification_prompt(
                    self.system_prompt,
                    request,
                    simulation_spec or {},
                    grounding or {},
                )
            )
        )
        if not parsed:
            return structural

        return _merge_verification(structural, _normalize_llm_verification(parsed))


def _load_system_prompt():
    prompt_path = Path("agents/simulation_spec_verifier/system_prompt.md")
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return "Verify simulation specs before code generation. Return JSON only."


def _verification_prompt(system_prompt, request, simulation_spec, grounding):
    return f"""
{system_prompt}

User request:
{request}

Proposed simulation spec:
{json.dumps(simulation_spec, indent=2)}

Retrieved evidence:
{json.dumps(_compact_grounding(grounding), indent=2)}
""".strip()


def _structural_verification(request, spec):
    normalized_spec = normalize_simulation_spec(spec or {})
    if normalized_spec.get("status") != "ready":
        return {
            "agent": "simulation_spec_verifier",
            "verdict": "skip",
            "confidence": "medium",
            "issues": [normalized_spec.get("reason") or "No ready simulation spec was proposed."],
            "suggested_fix": "Use free-form code generation only if it can be verified separately.",
            "source": "structural_checks",
        }

    issues = []
    demo_type = str(normalized_spec.get("demo_type") or "").lower()
    relation_family = str(normalized_spec.get("relation_family") or "").lower()
    expected_behavior = str(normalized_spec.get("expected_behavior") or "").lower()
    equation_text = str(normalized_spec.get("equation_text") or "").lower()
    x_label = str(normalized_spec.get("x_label") or "").lower()
    y_label = str(normalized_spec.get("y_label") or "").lower()
    parameters = normalized_spec.get("parameters") or {}

    if demo_type == "relation_plot":
        issues.extend(
            _relation_plot_issues(
                relation_family,
                parameters,
                expected_behavior,
                equation_text,
                x_label,
                y_label,
            )
        )

    if _has_unrelated_equation_terms(request, equation_text, x_label, y_label):
        issues.append(
            "The proposed equation/axis labels appear weakly related to the requested concept."
        )

    if issues:
        return {
            "agent": "simulation_spec_verifier",
            "verdict": "fail",
            "confidence": "high",
            "issues": issues,
            "suggested_fix": _suggested_fix(issues),
            "source": "structural_checks",
        }

    return {
        "agent": "simulation_spec_verifier",
        "verdict": "pass",
        "confidence": "medium",
        "issues": ["No structural simulation-spec issues found."],
        "suggested_fix": "",
        "source": "structural_checks",
    }


def _relation_plot_issues(
    relation_family,
    parameters,
    expected_behavior,
    equation_text,
    x_label,
    y_label,
):
    issues = []
    if "constant" in expected_behavior and not _relation_is_constant(
        relation_family,
        parameters,
    ):
        issues.append(
            "Expected behavior says a quantity remains constant, but the relation_plot primitive varies a single y value with x."
        )

    equation_terms = _terms(equation_text)
    label_terms = _terms(f"{x_label} {y_label}")
    if equation_terms and label_terms and equation_terms.isdisjoint(label_terms):
        issues.append(
            "Equation terms do not overlap with the axis labels, so the plot may not represent the stated equation."
        )

    if "frequency" in equation_text and "wavelength" in x_label:
        issues.append(
            "Equation uses frequency but x-axis label uses wavelength."
        )

    return issues


def _relation_is_constant(relation_family, parameters):
    if relation_family == "linear":
        return abs(_number(parameters.get("slope"), 1.0)) < 1e-12
    if relation_family == "threshold_linear":
        return abs(_number(parameters.get("slope"), 1.0)) < 1e-12
    if relation_family == "quadratic":
        return (
            abs(_number(parameters.get("coefficient"), 1.0)) < 1e-12
            and abs(_number(parameters.get("slope"), 0.0)) < 1e-12
        )
    if relation_family == "sinusoidal":
        return abs(_number(parameters.get("amplitude"), 1.0)) < 1e-12
    return False


def _has_unrelated_equation_terms(request, equation_text, x_label, y_label):
    request_terms = _terms(request)
    spec_terms = _terms(f"{equation_text} {x_label} {y_label}")
    if not request_terms or not spec_terms:
        return False
    important_spec_terms = spec_terms - {
        "energy",
        "force",
        "mass",
        "time",
        "total",
        "value",
    }
    if not important_spec_terms:
        return False
    return request_terms.isdisjoint(important_spec_terms) and len(request_terms & spec_terms) < 1


def _normalize_llm_verification(parsed):
    verdict = str(parsed.get("verdict", "caution")).lower()
    if verdict not in {"pass", "caution", "fail"}:
        verdict = "caution"
    confidence = str(parsed.get("confidence", "low")).lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    issues = _string_list(parsed.get("issues")) or ["Model verifier did not list issues."]
    return {
        "agent": "simulation_spec_verifier",
        "verdict": verdict,
        "confidence": confidence,
        "issues": issues,
        "suggested_fix": str(parsed.get("suggested_fix") or ""),
        "source": "llm",
    }


def _merge_verification(structural, llm_result):
    if structural["verdict"] == "fail":
        return structural
    if llm_result["verdict"] == "fail":
        return llm_result
    if llm_result["verdict"] == "caution":
        return llm_result
    return {
        **structural,
        "source": "structural_checks_with_llm_pass",
    }


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


def _suggested_fix(issues):
    if any("remains constant" in issue or "constant" in issue for issue in issues):
        return (
            "Use a primitive that can show multiple series, such as potential energy, "
            "kinetic energy, and their constant sum, or mark the demo unavailable."
        )
    return "Repair the simulation spec so equation, labels, primitive, and expected behavior agree."


def _terms(text):
    stopwords = {
        "and",
        "are",
        "can",
        "demo",
        "effect",
        "for",
        "from",
        "plot",
        "python",
        "show",
        "the",
        "this",
        "with",
    }
    return {
        term
        for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_]+", str(text).lower())
        if len(term) > 2 and term not in stopwords
    }


def _number(value, default):
    if isinstance(value, bool):
        return float(default)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return float(default)
    return float(default)


def _string_list(value):
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


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
