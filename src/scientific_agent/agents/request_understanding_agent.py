import json
from pathlib import Path

from src.scientific_agent.core.llm import LLMClient


class RequestUnderstandingAgent:
    """Resolve a user message into a standalone scientific request."""

    def __init__(self, provider="ollama", model=None):
        self.llm = LLMClient(provider=provider, model=model)
        self.system_prompt = _load_system_prompt()

    def run(self, request, conversation_context=""):
        if self.llm.available:
            parsed = _extract_json(
                self.llm.respond(
                    _understanding_prompt(
                        request,
                        conversation_context,
                        self.system_prompt,
                    )
                )
            )
            if parsed:
                understood = _normalize_understanding(parsed, request, source="llm")
                if not conversation_context:
                    understood["uses_context"] = False
                    if _mentions_prior_context(understood.get("reason")):
                        understood["reason"] = "self-contained request understood by model"
                if not _is_untrustworthy_context_resolution(
                    understood,
                    request,
                    conversation_context,
                ):
                    return understood
        return _fallback_understanding(request, conversation_context)


def _load_system_prompt():
    prompt_path = Path("agents/request_understanding/system_prompt.md")
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return "You are the request-understanding agent for a scientific learning platform."


def _understanding_prompt(request, conversation_context="", system_prompt=""):
    return f"""
{system_prompt}

Return only valid JSON with these fields:
- uses_context: boolean
- standalone_request: string
- search_queries: list of 1 to 4 concise physics textbook search queries
- reason: short string

Rules:
- Use conversation context only for short/vague follow-ups such as "show me the equation", "explain that", "plot it", or "now demonstrate it".
- If the current request names a clear new topic, set uses_context to false.
- Fix obvious spelling mistakes in standalone_request and search queries.
- Search queries should include scientific topic names, equation names, and synonyms.
- Do not answer the user's science question.

Conversation context:
{conversation_context or "(none)"}

Current request:
{request}

Example:
{{
  "uses_context": true,
  "standalone_request": "show me the entropy microstate equation",
  "search_queries": ["entropy microstates Boltzmann constant S k ln W"],
  "reason": "short follow-up resolved from prior entropy discussion"
}}
""".strip()


def _normalize_understanding(parsed, request, source):
    uses_context = bool(parsed.get("uses_context"))
    standalone_request = parsed.get("standalone_request")
    if not isinstance(standalone_request, str) or not standalone_request.strip():
        standalone_request = request
    if standalone_request.strip().lower() != request.strip().lower() and _needs_conversation_context(request):
        uses_context = True

    raw_queries = parsed.get("search_queries")
    if isinstance(raw_queries, str):
        raw_queries = [raw_queries]
    if not isinstance(raw_queries, list):
        raw_queries = []

    search_queries = [
        str(query).strip()
        for query in raw_queries
        if isinstance(query, (str, int, float)) and str(query).strip()
    ]
    if standalone_request.strip() not in search_queries:
        search_queries.insert(0, standalone_request.strip())

    return {
        "uses_context": uses_context,
        "standalone_request": standalone_request.strip(),
        "search_queries": _dedupe_search_queries(search_queries)[:4],
        "reason": str(parsed.get("reason") or "request understood by model"),
        "source": source,
    }


def _fallback_understanding(request, conversation_context=""):
    uses_context = _needs_conversation_context(request)
    standalone_request = request
    search_queries = [request]
    if uses_context and conversation_context:
        previous_topic = _last_user_message(conversation_context) or _last_assistant_topic(
            conversation_context
        )
        if previous_topic:
            standalone_request = f"{previous_topic}; {request}"
            search_queries.append(standalone_request)
        else:
            standalone_request = _with_conversation_context(request, conversation_context)
            search_queries.append(standalone_request)
    return {
        "uses_context": uses_context,
        "standalone_request": standalone_request,
        "search_queries": _dedupe_search_queries(search_queries),
        "reason": "fallback understanding used because model output was unavailable or invalid",
        "source": "fallback_rules",
    }


def _is_untrustworthy_context_resolution(understanding, request, conversation_context):
    if not conversation_context or not _needs_conversation_context(request):
        return False

    if not understanding.get("uses_context"):
        return True

    previous_topic = _last_user_message(conversation_context) or _last_assistant_topic(
        conversation_context
    )
    previous_terms = set(_important_terms(previous_topic or ""))
    standalone_terms = set(_important_terms(understanding.get("standalone_request") or ""))

    if not previous_terms or not standalone_terms:
        return False

    return previous_terms.isdisjoint(standalone_terms)


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


def _dedupe_search_queries(search_queries):
    seen = set()
    deduped = []
    for query in search_queries:
        normalized = " ".join(str(query).split())
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            deduped.append(normalized)
    return deduped


def _mentions_prior_context(text):
    lowered = str(text or "").lower()
    return any(
        marker in lowered
        for marker in [
            "follow-up",
            "followup",
            "prior",
            "previous",
            "resolved from",
            "conversation",
        ]
    )


def _last_user_message(conversation_context):
    for line in reversed((conversation_context or "").splitlines()):
        if line.startswith("user:"):
            return line.split("user:", 1)[1].strip()
    return None


def _last_assistant_topic(conversation_context):
    for line in reversed((conversation_context or "").splitlines()):
        if line.startswith("assistant:"):
            text = line.split("assistant:", 1)[1].strip()
            return text[:180]
    return None


def _with_conversation_context(request, conversation_context):
    if not conversation_context:
        return request
    return (
        "Conversation context:\n"
        f"{conversation_context}\n\n"
        "Current user request:\n"
        f"{request}"
    )


def _needs_conversation_context(request):
    lower_request = request.lower()
    words = {
        "".join(ch for ch in raw if ch.isalnum())
        for raw in lower_request.replace("-", " ").split()
    }
    word_markers = {
        "it",
        "that",
        "this",
        "same",
        "above",
        "previous",
        "equation",
        "equatuin",
        "equatoin",
        "eqn",
        "formula",
        "derive",
        "now",
        "again",
    }
    phrase_markers = [
        "show me",
        "tell me",
        "what about",
        "plot it",
        "demo it",
    ]
    if words.intersection(word_markers) or any(marker in lower_request for marker in phrase_markers):
        content_terms = [
            term
            for term in _important_terms(request)
            if term
            not in {
                "equation",
                "equatuin",
                "equatoin",
                "formula",
                "derive",
                "previous",
                "above",
            }
        ]
        pronoun_markers = {"it", "that", "this", "same", "above", "previous"}
        if content_terms and not words.intersection(pronoun_markers):
            return False
        return True
    return False


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
