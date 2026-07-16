import json
import os
import time
from urllib import error as urlerror
from urllib import request as urlrequest


class LLMClient:
    def __init__(self, provider=None, model=None, max_retries=None):
        self.provider = provider or os.getenv("LLM_PROVIDER", "gemini")
        self.model = model or _default_model(self.provider)
        self.models = _candidate_models(self.provider, self.model)
        self.max_retries = _int_from_env("LLM_MAX_RETRIES", max_retries, default=2)
        self.error = None
        self.last_successful_model = None
        self.attempts = []
        self.client = None

        if self.provider == "gemini":
            self._init_gemini()
        elif self.provider == "groq":
            self._init_groq()
        elif self.provider == "openai":
            self._init_openai()
        elif self.provider == "ollama":
            self._init_ollama()
        else:
            self.error = f"Unsupported LLM_PROVIDER: {self.provider}"

    @property
    def available(self):
        return self.client is not None

    def improve_config(self, query, fallback_config, evidence):
        if not self.available:
            return fallback_config

        prompt = _config_prompt(query, fallback_config, evidence)
        text = self._respond(prompt)
        if not text:
            return fallback_config

        parsed = _extract_json_object(text)
        if not parsed:
            return fallback_config

        improved = dict(fallback_config)
        for key in fallback_config:
            if key in parsed:
                improved[key] = parsed[key]
        improved["raw_query"] = query
        return improved

    def write_report(self, query, config, evidence, result_summary, analysis):
        if not self.available:
            return None

        prompt = _report_prompt(query, config, evidence, result_summary, analysis)
        return self._respond(prompt)

    def respond(self, prompt):
        if not self.available:
            return None
        return self._respond(prompt)

    def _init_gemini(self):
        if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
            self.error = "GEMINI_API_KEY is not set"
            return

        try:
            from google import genai

            self.client = genai.Client()
        except ImportError:
            self.error = "google-genai package is not installed"

    def _init_openai(self):
        if not os.getenv("OPENAI_API_KEY"):
            self.error = "OPENAI_API_KEY is not set"
            return

        try:
            from openai import OpenAI

            self.client = OpenAI()
        except ImportError:
            self.error = "openai package is not installed"

    def _init_groq(self):
        if not os.getenv("GROQ_API_KEY"):
            self.error = "GROQ_API_KEY is not set"
            return

        self.client = {
            "api_key": os.getenv("GROQ_API_KEY"),
            "base_url": os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/"),
        }

    def _init_ollama(self):
        self.client = {
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        }

    def _respond(self, prompt):
        self.attempts = []

        for model in self.models:
            for retry_index in range(self.max_retries + 1):
                try:
                    text = self._single_response(prompt, model)
                    self.error = None
                    self.model = model
                    self.last_successful_model = model
                    self.attempts.append(
                        {
                            "model": model,
                            "attempt": retry_index + 1,
                            "status": "success",
                        }
                    )
                    return text
                except Exception as exc:
                    error_text = str(exc)
                    self.error = error_text
                    self.attempts.append(
                        {
                            "model": model,
                            "attempt": retry_index + 1,
                            "status": "error",
                            "error": error_text,
                        }
                    )

                    if retry_index < self.max_retries and _is_temporary_error(error_text):
                        time.sleep(2**retry_index)
                        continue
                    break

        if not self.error:
            self.error = "LLM response failed"
        return None

    def _single_response(self, prompt, model):
        if self.provider == "gemini":
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
            )
            return response.text

        if self.provider == "openai":
            response = self.client.responses.create(
                model=model,
                input=prompt,
            )
            return response.output_text

        if self.provider == "groq":
            return self._groq_response(prompt, model)

        if self.provider == "ollama":
            return self._ollama_response(prompt, model)

        raise ValueError(f"Unsupported LLM_PROVIDER: {self.provider}")

    def _ollama_response(self, prompt, model):
        payload = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
            }
        ).encode("utf-8")
        request = urlrequest.Request(
            f"{self.client['base_url']}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlrequest.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urlerror.URLError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        if "error" in data:
            raise RuntimeError(data["error"])
        return data.get("response", "")

    def _groq_response(self, prompt, model):
        payload = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8")
        request = urlrequest.Request(
            f"{self.client['base_url']}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.client['api_key']}",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "md-agent-workbench/0.1",
            },
            method="POST",
        )

        try:
            with urlrequest.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urlerror.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Groq request failed: HTTP {exc.code}: {body}") from exc
        except urlerror.URLError as exc:
            raise RuntimeError(f"Groq request failed: {exc}") from exc

        return (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")


def _default_model(provider):
    if provider == "gemini":
        return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    if provider == "groq":
        return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    if provider == "openai":
        return os.getenv("OPENAI_MODEL", "gpt-5.2")
    if provider == "ollama":
        return os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    return os.getenv("LLM_MODEL", "")


def _candidate_models(provider, primary_model):
    if provider == "gemini":
        raw = os.getenv("GEMINI_FALLBACK_MODELS", "")
    elif provider == "groq":
        raw = os.getenv("GROQ_FALLBACK_MODELS", "")
    elif provider == "openai":
        raw = os.getenv("OPENAI_FALLBACK_MODELS", "")
    elif provider == "ollama":
        raw = os.getenv("OLLAMA_FALLBACK_MODELS", "")
    else:
        raw = os.getenv("LLM_FALLBACK_MODELS", "")

    candidates = [primary_model]
    candidates.extend(model.strip() for model in raw.split(",") if model.strip())
    return _dedupe(candidates)


def _dedupe(items):
    seen = set()
    result = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _int_from_env(name, value, default):
    if value is not None:
        return int(value)

    raw = os.getenv(name)
    if not raw:
        return default

    try:
        return int(raw)
    except ValueError:
        return default


def _is_temporary_error(error_text):
    markers = [
        "503",
        "unavailable",
        "temporarily",
        "high demand",
        "rate limit",
        "timeout",
    ]
    lower_error = error_text.lower()
    return any(marker in lower_error for marker in markers)


def _config_prompt(query, fallback_config, evidence):
    return f"""
You are an MD setup assistant. Convert the user's request into a conservative
molecular dynamics configuration. Use the fallback config as the schema.

Return only valid JSON. Do not include markdown.

Allowed keys:
- raw_query: string
- system: string
- ensemble: NVE, NVT, or NPT
- temperature_k: number
- pressure: number or null
- steps: integer
- timestep_fs: number
- thermostat: string
- force_field: string

User request:
{query}

Fallback config:
{json.dumps(fallback_config, indent=2)}

Knowledge evidence:
{json.dumps(evidence, indent=2)}
""".strip()


def _report_prompt(query, config, evidence, result_summary, analysis):
    return f"""
You are an MD research assistant. Write a concise research-style report for this
prototype MD workflow. Be clear that the current calculation is a toy placeholder
unless the result says otherwise. Use the evidence sources when relevant.

User request:
{query}

Simulation config:
{json.dumps(config, indent=2)}

Knowledge evidence:
{json.dumps(evidence, indent=2)}

Result summary:
{json.dumps(result_summary, indent=2)}

Analysis:
{json.dumps(analysis, indent=2)}
""".strip()


def _extract_json_object(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
