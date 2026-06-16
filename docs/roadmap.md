# Scientific Agent Roadmap

This file is the project memory for what to do next. If work pauses, restart by
reading this file, `docs/product_architecture.md`, `docs/architecture.md`, and
`docs/discovery_lessons.md`.

## Current State

The repo currently supports:

- provider-flexible LLM calls through `LLMClient`
  - `ollama`
  - `gemini`
  - `openai`
- deterministic MD workflow mode
- portable tool-agent mode where the LLM proposes a trusted tool plan
- local MD toy runner
- external MD runner for the sibling `MD_simulation/` folder
- local RAG over PDFs/text files
- RAG answer mode with Ollama/Gemini/OpenAI-compatible provider layer
- dynamic discovery of tool manifests from `tools/*/tool.yaml`
- a first Python learning-demo runner for generated code demos
- product-level specialized agents under `src/scientific_agent/agents/`
  - `RouterAgent`
  - `RequestUnderstandingAgent`
  - `LearningDemoAgent`
  - `SimulationAgent`
  - `RAGAgent`
  - `ScientificAgent`
- Discovery-style physics learning agent spec under `agents/physics_learning/`
- request-understanding agent spec under `agents/request_understanding/`
- reusable grounding/RAG retrieval service under `src/scientific_agent/core/grounding.py`
- job tracking under `outputs/jobs/<job_id>/`
- honest unreliable-demo handling when generated code is not trustworthy
- explanation-only learning mode for concept questions that do not explicitly
  ask for code/demo/plot/execution
- RAG-grounded learning answers using the local physics textbook index
- grounding status in learning results: `grounded`, `weak`, or `missing`
- session memory under `outputs/sessions/<session_id>/messages.jsonl`
- interactive chat mode through `python -m src.scientific_agent chat`
- minimal web UI through `python -m src.scientific_agent ui`
- `PhysicsVerifierAgent` with verdict/confidence/issues on learning and RAG answers
- model-driven request understanding for follow-up detection and RAG query
  rewriting, with deterministic fallback only as a safety guardrail
- deterministic verifier checks for obvious physics contradictions such as
  incorrect Newton's-third-law direction/formula claims

Useful smoke tests:

```bash
python -m src.scientific_agent \
  --provider ollama \
  --model llama3.2:1b \
  "Do you know quantum physcis?"
```

```bash
python -m src.scientific_agent chat \
  --session physics-test \
  --provider ollama \
  --model llama3.2:1b
```

```bash
python -m src.scientific_agent \
  --provider ollama \
  --model llama3.2:1b \
  "Teach me diffusion with a random walk Python demo"
```

```bash
python -m src.scientific_agent \
  --provider ollama \
  --model llama3.2:1b \
  --engine local \
  "Run an NVT MD simulation for water at 300 K for 5 steps and analyze it"
```

```bash
python -m src.md_agent.agent \
  --mode tool-agent \
  --provider ollama \
  --model llama3.2:1b \
  --engine local \
  "Run an NVT MD simulation for water at 300 K for 5 steps and analyze it"
```

```bash
python -m src.scientific_agent.rag answer \
  "What is the Co-Scientist multi-agent architecture?" \
  --provider ollama \
  --model llama3.2:1b \
  --top-k 3
```

```bash
python -m src.md_agent.agent \
  --mode tool-agent \
  --provider ollama \
  --model llama3.2:1b \
  "Teach me entropy and demonstrate it with a short Python code"
```

```bash
conda run -n ai-agent python -m src.scientific_agent.rag search \
  "entropy microstates statistical mechanics" \
  --top-k 3
```

## Current Step

Harden the product architecture before adding more UI features.

The repo now separates request understanding, grounding retrieval, learning
orchestration, execution, verification, job storage, and session memory. The
next architecture work should improve trust and extensibility rather than add
topic-specific Python branches.

Target shape:

```text
user request
-> route
-> understand request
-> retrieve grounding
-> domain agent drafts answer/tool plan
-> trusted tool execution if needed
-> verifier checks answer/evidence/code
-> persisted job + UI/CLI response
```

## Dynamic Tool/Plugin Loading

Goal: build dynamic tool/plugin loading.

Goal: a user should be able to add a new tool by creating a folder under
`tools/` with a manifest, without editing core agent code.

Target shape:

```text
tools/
  my_tool/
    tool.yaml
    run.py
```

Example manifest:

```yaml
name: run_lammps
description: Run a LAMMPS simulation from an input script.
type: command
command: python tools/lammps_tool/run.py
input_schema:
  input_file: string
  steps: integer
  temperature: number
```

Expected behavior:

```text
agent startup
-> scan tools/*/tool.yaml
-> expose tool names/descriptions to planner
-> LLM chooses tools
-> Python validates and executes only trusted registered tools
-> tool returns JSON
-> LLM summarizes result
```

## Implementation Checklist

- [x] Create `src/scientific_agent/core/tool_loader.py`
- [x] Load all `tools/*/tool.yaml`
- [ ] Support at least two tool types:
  - [ ] `python_function` for internal Python functions
  - [x] `command` for safe subprocess wrappers
- [ ] Add validation rules:
  - [x] tool command must live inside the repo
  - [x] tool input must be JSON
  - [x] tool output must be JSON
  - [x] no arbitrary shell execution from the LLM
- [x] Refactor `PortableMDToolAgent` to read available tools from the loader
- [ ] Convert current MD/RAG tools into manifests
- [x] Add one demo non-MD tool, for example an Ising model or harmonic oscillator
- [x] Add smoke tests for tool discovery and one dynamic tool execution
- [ ] Update README with "Bring Your Own Tool"

## Immediate Next Step

Apply the Discovery-style catalog pattern locally:

1. Add catalog manifests for remaining current product agents.
2. Upgrade `tools/python_demo_runner/tool.yaml` with version, category, actions,
   input schema, and output schema.
3. Add a `starter_kits/physics-learning/kit.json` with sample prompts.
4. Improve `PhysicsVerifierAgent` with stricter source-claim checking and
   formula/code consistency checks.
5. Add session summaries so long chats do not need to pass the full recent
   message history forever.

Then finish the plugin system by converting one existing MD/RAG path into a
truly manifest-executable tool and add a short "Bring Your Own Tool" README
section.

After that, add `job_store.py` so every run writes:

```text
outputs/jobs/<job_id>/
  request.json
  route.json
  tool_calls.json
  result.json
  report.md
  generated_code.py
  artifacts/
```

Status: implemented for product-level runs through `python -m src.scientific_agent`.

Learning/code jobs now save the final attempted/executed Python code as
`generated_code.py`.

## Later Steps

- [ ] Add embedding-based RAG behind the same `LocalRAG` interface
- [x] Add job history in `outputs/jobs/`
- [x] Save generated code for each learning/code job
- [ ] Add user approval gates for expensive or external commands
- [ ] Add Docker execution for heavy scientific tools
- [ ] Add LangGraph after tools are dynamic and stable
- [ ] Add a simple UI for users to ask, run, inspect, and learn

## Design Rule

The LLM should plan and explain. Python should validate and execute. New tools
should be added through manifests, not by editing the core agent every time.

For learning requests, the model can classify intent and generate code, but a
policy gate protects the product: no code is executed unless the user explicitly
asks for a runnable demo, plot, calculation, saved output, or execution.
