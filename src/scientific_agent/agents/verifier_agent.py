import json
from pathlib import Path

from src.scientific_agent.core.llm import LLMClient


class PhysicsVerifierAgent:
    """Review learning-agent output against grounding evidence and tool status."""

    def __init__(self, provider="ollama", model=None):
        self.llm = LLMClient(provider=provider, model=model)
        self.agent_spec = _load_verifier_spec()

    def run(self, request, result):
        fallback = _fallback_verification(result)
        if not self.llm.available:
            return fallback

        parsed = _extract_json(self.llm.respond(_verification_prompt(self.agent_spec, request, result)))
        if not parsed:
            return fallback

        return _normalize_verification(parsed, fallback)


def _load_verifier_spec():
    spec_path = Path("agents/physics_verifier/system_prompt.md")
    if spec_path.exists():
        return spec_path.read_text(encoding="utf-8")
    return "Review physics answers against evidence. Return JSON only."


def _verification_prompt(agent_spec, request, result):
    compact_result = {
        "agent": result.get("agent"),
        "status": result.get("status"),
        "request": result.get("request") or request,
        "final_answer": (result.get("final_answer") or "")[:5000],
        "grounding": _compact_grounding(result.get("grounding") or {}),
        "tool": result.get("tool"),
        "tool_result": _compact_tool_result(result.get("result") or {}),
        "code_source": result.get("code_source"),
    }
    return f"""
{agent_spec}

User request:
{request}

Candidate result:
{json.dumps(compact_result, indent=2)}
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
                "text": (source.get("text") or "")[:700],
            }
        )
    return {
        "status": grounding.get("status"),
        "reason": grounding.get("reason"),
        "sources": sources,
    }


def _compact_tool_result(tool_result):
    return {
        "status": tool_result.get("status"),
        "returncode": tool_result.get("returncode"),
        "stdout": (tool_result.get("stdout") or "")[:1200],
        "stderr": (tool_result.get("stderr") or "")[:1200],
        "generated_files": tool_result.get("generated_files") or [],
    }


def _fallback_verification(result):
    grounding = result.get("grounding") or {}
    tool_result = result.get("result") or {}
    final_answer = (result.get("final_answer") or result.get("answer") or "").lower()
    issues = []
    verdict = "pass"
    confidence = "medium"

    if grounding.get("status") == "missing":
        verdict = "caution"
        confidence = "low"
        issues.append("No local source evidence was found.")
    elif grounding.get("status") == "weak":
        verdict = "caution"
        confidence = "low"
        issues.append("Only weak or partial local source evidence was found.")

    if result.get("tool") and tool_result.get("status") != "completed":
        verdict = "caution" if verdict != "fail" else verdict
        confidence = "low"
        issues.append("The requested tool/code demo did not complete successfully.")

    if result.get("status") in {"failed", "rejected", "unreliable"}:
        verdict = "caution"
        confidence = "low"
        issues.append(f"Agent status is {result.get('status')}.")

    generated_files = tool_result.get("generated_files") or result.get("generated_files") or []
    if any(phrase in final_answer for phrase in ["image below", "plot below", "figure below"]):
        if not generated_files:
            verdict = "caution"
            confidence = "low"
            issues.append("The answer refers to a visual artifact, but no artifact was generated.")

    physics_flags = _physics_consistency_issues(final_answer)
    if physics_flags:
        verdict = "fail" if any(flag["severity"] == "fail" for flag in physics_flags) else "caution"
        confidence = "low"
        issues.extend(flag["message"] for flag in physics_flags)

    if not issues:
        issues.append("No obvious verification issues from deterministic checks.")

    return {
        "agent": "physics_verifier",
        "verdict": verdict,
        "confidence": confidence,
        "issues": issues,
        "supported_claims": [],
        "unsupported_claims": [],
        "suggested_note": _suggested_note(verdict, issues),
        "source": "fallback_checks",
    }


def _physics_consistency_issues(final_answer):
    issues = []
    if "newton" in final_answer and "third law" in final_answer:
        if "same direction" in final_answer:
            issues.append(
                {
                    "severity": "fail",
                    "message": "Newton's third-law forces must be opposite in direction, but the answer says same direction.",
                }
            )
        if "action = force" in final_answer and "distance" in final_answer:
            issues.append(
                {
                    "severity": "fail",
                    "message": "The answer gives Action = Force * Distance, which is not Newton's third-law equation.",
                }
            )
    return issues


def _normalize_verification(parsed, fallback):
    verdict = str(parsed.get("verdict", fallback["verdict"])).lower()
    if verdict not in {"pass", "caution", "fail"}:
        verdict = fallback["verdict"]

    confidence = str(parsed.get("confidence", fallback["confidence"])).lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = fallback["confidence"]

    issues = _string_list(parsed.get("issues")) or fallback["issues"]
    supported_claims = _string_list(parsed.get("supported_claims"))
    unsupported_claims = _string_list(parsed.get("unsupported_claims"))
    if fallback["verdict"] in {"caution", "fail"} and verdict == "pass":
        verdict = fallback["verdict"]
        confidence = fallback["confidence"]
        issues = issues + fallback["issues"]
    if unsupported_claims and verdict == "pass":
        verdict = "caution"
        confidence = "medium" if confidence == "high" else confidence
        issues = issues + ["Verifier listed unsupported claims, so verdict was downgraded."]
    suggested_note = str(parsed.get("suggested_note") or _suggested_note(verdict, issues))

    return {
        "agent": "physics_verifier",
        "verdict": verdict,
        "confidence": confidence,
        "issues": issues,
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported_claims,
        "suggested_note": suggested_note,
        "source": "llm",
    }


def _suggested_note(verdict, issues):
    if verdict == "pass":
        return "Verifier: answer is reasonably supported by the retrieved local evidence."
    if verdict == "fail":
        return "Verifier: answer should not be treated as reliable without correction."
    return "Verifier: use this answer with caution. " + " ".join(issues[:2])


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
