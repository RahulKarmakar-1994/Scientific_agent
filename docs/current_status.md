# Current Product Status

Last updated: 2026-07-12

This repo is currently a working research prototype for an agentic scientific
learning and molecular-dynamics workbench. It is not yet a polished product.

## What Works Now

- Provider-flexible LLM client for Ollama, Gemini, and OpenAI-compatible runs.
- Top-level `ScientificAgent` entrypoint with routing across:
  - physics learning/demo requests
  - molecular simulation requests
  - local document/RAG questions
- Local RAG retrieval over ingested PDF/text sources.
- Job persistence under `outputs/jobs/<job_id>/`.
- Session memory under `outputs/sessions/<session_id>/`.
- Interactive chat mode:
  - `python -m src.scientific_agent chat`
  - session-aware follow-up handling
  - prediction-before-demo state for prepared demo requests
- Trusted local Python demo execution through `python_demo_runner`.
- Reusable demo primitives instead of topic-specific scripts:
  - relation plots
  - random walks
  - distributions
  - time evolution
  - multi-series conserved-total evolution
  - phase-space plots
- Simulation-spec generation and verification before demo code is run.
- Physics answer verifier with `pass`, `caution`, and `fail` style outcomes.
- Discovery-style product catalog metadata:
  - `agents/*/agent.yaml`
  - `tools/*/tool.yaml`
  - `starter_kits/physics-learning/kit.json`

## Recent Product Changes

- Added catalog manifests for product agents:
  - `router_agent`
  - `rag_agent`
  - `simulation_agent`
  - `scientific_agent`
  - `lesson_interaction`
- Upgraded `tools/python_demo_runner/tool.yaml` with:
  - version
  - category
  - risk metadata
  - action schema
  - input/output schemas
- Added the `physics-learning` starter kit.
- Added chat/CLI warning when the requested LLM model fails and fallback rules
  are used.
- Made broad unsupported demo requests safer:
  - no fake prediction prompt
  - no stored pending demo state
  - no misleading fallback demo
- Added safe fallback mappings for request-level primitives:
  - conservation-style requests -> conserved-total exchange demo
  - diffusion/random-walk requests -> random-walk demo
- Hardened demo primitives:
  - zero-total conserved demos are normalized to visible nonzero behavior
  - verifier no longer applies relation-plot-only checks to random-walk specs
- Cleaned fallback learner text so it does not expose stale roadmap language.

## Tested Smoke Paths

Manifest validation:

```bash
python - <<'PY'
import json
from pathlib import Path
import yaml

for path in sorted(Path("agents").glob("*/agent.yaml")):
    data = yaml.safe_load(path.read_text())
    assert data.get("name"), path

for path in sorted(Path("tools").glob("*/tool.yaml")):
    data = yaml.safe_load(path.read_text())
    assert data.get("name"), path

for path in sorted(Path("starter_kits").glob("*/*.json")):
    data = json.loads(path.read_text())
    assert data.get("name"), path

print("validated manifests")
PY
```

Tool discovery and execution:

```bash
python - <<'PY'
from src.scientific_agent.core.tool_loader import ToolLoader

loader = ToolLoader()
result = loader.execute("python_demo_runner", {
    "topic": "smoke test",
    "code": "print('demo smoke ok')",
})
print(result["status"])
print(result["stdout"])
PY
```

Missing-model/fallback warning:

```bash
printf 'Teach me quantum mechanics with a Python demo\nexit\n' | \
python -m src.scientific_agent chat \
  --session smoke-chat-warning \
  --provider ollama \
  --model llama3.1:8b
```

Expected behavior: no fake quantum demo, no pending prediction, visible LLM
warning if `llama3.1:8b` is not installed.

Diffusion random-walk demo:

```bash
python -m src.scientific_agent \
  --provider ollama \
  --model llama3.2:1b \
  "Teach me diffusion as a non-equilibrium process with a Python random walk demo"
```

Expected behavior: trusted random-walk demo runs and reports mean squared
displacement growth.

Conservation prediction loop:

```bash
python -m src.scientific_agent chat \
  --session physics-test-new \
  --provider ollama \
  --model llama3.2:1b
```

Then:

```text
Teach me conservation of energy with a Python demo
A
```

Expected behavior: first turn asks for a prediction; second turn runs the same
prepared demo and compares the prediction.

## Known Limitations

- This is still a prototype, not a finished product.
- Output is still too raw for end users. The CLI/UI need a product response
  layer that hides internal fields and presents answer, demo, citations, model
  status, verification, and artifacts cleanly.
- Verifier warnings currently append caution notes. They do not yet trigger
  automatic answer repair.
- Small local models can return invalid JSON or weak scientific explanations.
- The app can be correct operationally while still producing mediocre teaching
  text if the model is weak.
- Halliday/Resnick is not enough source material for advanced topics like
  non-equilibrium statistical mechanics.
- The web UI has not yet been upgraded to show model status, warnings,
  verifier state, artifact preview, or structured sections.
- There is no full automated test suite yet.

## Best Next Step

Build a product-quality response layer shared by CLI and UI:

```text
raw agent result
-> hide/debug internal fields
-> expose model status clearly
-> repair or block weak answers
-> format citations
-> format demo/artifact output
-> show concise learner-facing answer
```

After that, add answer repair and a focused regression test suite for:

- missing Ollama model
- unsupported broad demo
- supported conservation demo
- supported diffusion/random-walk demo
- prediction-before-demo session flow
- verifier `caution` and `fail` behavior
