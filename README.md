# MD Agent Workbench

A small, from-scratch agentic AI workbench for molecular dynamics workflows.

The project is currently one product with modular boundaries. The stable future
namespace is `scientific_agent`; the original `md_agent` package remains as the
working MD prototype and compatibility entrypoint.

For the next implementation step, see `docs/roadmap.md`. The current planned
step is dynamic tool/plugin loading so new tools can be added without editing
the core agent code.

Product architecture notes live in `docs/product_architecture.md`.

This first version is intentionally simple:

- a domain-specific knowledge agent that searches local notes and papers
- an MD setup agent that parses a natural-language request into parameters
- an optional LLM layer for better setup and reporting
- an MD runner that can run locally or through Docker
- an analysis agent that summarizes simulation outputs
- a toy MD calculation that you can later replace with your custom code

## Project Layout

```text
md-agent-workbench/
├── agents/
│   ├── md_analysis/agent.yaml
│   ├── md_knowledge/agent.yaml
│   └── md_runner/agent.yaml
├── knowledgebase/
│   ├── forcefield_docs/
│   └── papers/
├── src/md_agent/
│   ├── agent.py
│   ├── agent_registry.py
│   ├── analysis.py
│   ├── knowledge.py
│   ├── langchain_agent.py
│   ├── llm.py
│   ├── parser.py
│   ├── runner.py
│   ├── tool_registry.py
│   └── workflow_orchestrator.py
├── src/scientific_agent/
│   ├── core/
│   └── domains/md/
├── tools/md_docker_tool/
│   ├── Dockerfile
│   ├── run_md.py
│   └── tool.yaml
├── tools/knowledge_search/tool.yaml
├── tools/md_analysis/tool.yaml
├── inputs/
├── outputs/
├── workflows/basic_md_workflow.yaml
└── requirements.txt
```

## Quick Start

Run the prototype locally:

```bash
python -m src.md_agent.agent --engine local "Run NVT MD for water at 300 K for 1000 steps using 1 fs timestep"
```

Equivalent future-facing entrypoint:

```bash
python -m src.scientific_agent --engine local "Run NVT MD for water at 300 K"
```

Use a session id to preserve conversation memory across turns:

```bash
python -m src.scientific_agent \
  --provider ollama \
  --model llama3.2:1b \
  --session-id physics-test \
  "Teach me entropy and microstates"

python -m src.scientific_agent \
  --provider ollama \
  --model llama3.2:1b \
  --session-id physics-test \
  "show me the equation"
```

Or use interactive chat:

```bash
python -m src.scientific_agent chat \
  --session physics-test \
  --provider ollama \
  --model llama3.2:1b
```

Run the minimal local web UI:

```bash
python -m src.scientific_agent ui \
  --host 127.0.0.1 \
  --port 8765 \
  --provider ollama \
  --model llama3.2:1b
```

Then open:

```text
http://127.0.0.1:8765
```

The product-level entrypoint routes requests to specialized agents:

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

Current product agents:

```text
RouterAgent -> chooses learning_demo, simulation, or rag
RequestUnderstandingAgent -> rewrites follow-ups into standalone scientific requests
SimulationSpecAgent -> maps learning requests to reusable demo specifications
SimulationSpecVerifierAgent -> blocks misleading demo specs before code generation
LearningDemoAgent -> answers concepts or generates/runs Python demos through python_demo_runner
SimulationAgent -> runs MD parse/search/run/analyze workflow
RAGAgent -> answers from the local document index
PhysicsVerifierAgent -> reviews answers against evidence/tool results
```

Shared product services:

```text
GroundingService -> searches local RAG evidence for learning agents
demo_primitives -> builds safe code from reusable demo specs, not topic templates
JobStore -> saves request/result/report/artifacts per run
SessionStore -> preserves multi-turn context by session id
```

`LearningDemoAgent` has two modes:

```text
concept_explanation -> no tool call, no generated code, direct teaching answer
python_demo -> generate/repair/validate/run code through the trusted runner
```

For demo requests, the preferred path is:

```text
request + RAG evidence
-> SimulationSpecAgent
-> SimulationSpecVerifierAgent
-> reusable demo primitive, such as relation_plot or random_walk
-> python_demo_runner
```

This avoids adding hardcoded branches like `if topic == "photoelectric"`.
The model supplies the scientific meaning; Python supplies trusted reusable
plot/simulation primitives and rejects incomplete or inconsistent specs.

The LLM can classify and generate, but execution is protected by a product
policy gate: code is run only when the user explicitly asks for code, Python, a
demo, plot, calculation, saved output, or execution. This keeps broad questions
like "Do you know quantum physics?" from becoming accidental code jobs.

The physics learning behavior is configured through:

```text
agents/physics_learning/
  agent.yaml
  system_prompt.md
```

This keeps product instructions separate from Python orchestration code, similar
to the Discovery catalog pattern.

The learning agent also searches the local RAG index before answering. Its
result includes a grounding status:

```text
grounded -> multiple strong local source chunks were found
weak     -> only partial local evidence was found
missing  -> no local RAG index or no useful source chunks
```

Ingest local physics material:

```bash
conda run -n ai-agent python -m src.scientific_agent.rag ingest \
  knowledgebase/physics/Al_HalidayResnick_Fundamentals.pdf
```

Test retrieval:

```bash
conda run -n ai-agent python -m src.scientific_agent.rag search \
  "entropy microstates statistical mechanics" \
  --top-k 3
```

Each product-level run creates a trackable job folder:

```text
outputs/jobs/<job_id>/
  request.json
  route.json
  result.json
  report.md
  generated_code.py
  artifacts/
```

If the learning agent cannot generate a reliable concept-specific demo, it
returns `status: "unreliable"` and does not run an unrelated fallback plot.
Learning/code jobs save the final executed code as `generated_code.py`.

Run the deterministic workflow without any LLM:

```bash
python -m src.md_agent.agent --mode workflow --no-llm --engine local "Run NVT MD for water at 300 K"
```

Run the YAML workflow orchestrator:

```bash
python -m src.md_agent.agent \
  --mode orchestrate \
  --engine local \
  --workflow workflows/basic_md_workflow.yaml \
  "Run NVT MD for water at 300 K for 20 steps"
```

The app works without an LLM key. In that case it uses the local rule-based
parser and returns `"llm": {"enabled": false, ...}` in the report.

To use Gemini:

```bash
export GEMINI_API_KEY="your_api_key_here"
pip install -r requirements.txt
python -m src.md_agent.agent "Design an NPT MD run for water at 300 K and 1 atm"
```

By default the Gemini wrapper uses `gemini-2.5-flash`. You can override it:

```bash
export GEMINI_MODEL="gemini-2.5-flash"
```

If Gemini returns a temporary `503 UNAVAILABLE` or high-demand error, the LLM
client retries automatically. You can also provide fallback models:

```bash
export GEMINI_FALLBACK_MODELS="gemini-2.0-flash"
export LLM_MAX_RETRIES=2
python -m src.md_agent.agent "Design an NPT MD run for water at 300 K and 1 atm"
```

Or choose a model for one run:

```bash
python -m src.md_agent.agent --model gemini-2.0-flash "Design an NPT MD run for water at 300 K"
```

The provider layer is also compatible with OpenAI later:

```bash
export OPENAI_API_KEY="your_api_key_here"
python -m src.md_agent.agent --provider openai "Design an NPT MD run for water at 300 K"
```

Or run the same workflow with a local Ollama model:

```bash
python -m src.md_agent.agent \
  --provider ollama \
  --model llama3.2:1b \
  --engine local \
  "Design an NVT MD run for water at 300 K for 20 steps"
```

To force local-only mode:

```bash
python -m src.md_agent.agent --no-llm --engine local "Run NVT MD for water at 300 K"
```

## Portable Tool Agent

This mode is closer to your REDOX prototype: the LLM chooses a trusted tool
plan, then Python executes only registered local tools. It can use Ollama,
Gemini, or OpenAI through the same provider layer.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run with local Ollama:

```bash
python -m src.md_agent.agent \
  --mode tool-agent \
  --provider ollama \
  --model llama3.2:1b \
  --engine local \
  "Run an NVT MD simulation for water at 300 K for 20 steps and analyze it"
```

Run an interactive learning demo with a dynamically loaded Python tool:

```bash
python -m src.md_agent.agent \
  --mode tool-agent \
  --provider ollama \
  --model llama3.2:1b \
  "Teach me entropy and demonstrate it with a short Python code"
```

The agent discovers `tools/python_demo_runner/tool.yaml`, asks the LLM for a
small Python demo, executes it through the trusted runner, and saves generated
files under `outputs/python_demos/`.

Or set a Gemini key. Either name works:

```bash
export GEMINI_API_KEY="your_api_key_here"
# or
export GOOGLE_API_KEY="your_api_key_here"
```

Then run the same tool-agent with Gemini:

```bash
python -m src.md_agent.agent \
  --mode tool-agent \
  --provider gemini \
  --engine local \
  --model gemini-2.0-flash \
  "Run an NVT MD simulation for water at 300 K for 100 steps and analyze it"
```

## Local RAG

The project includes a lightweight local RAG index for PDFs and text files. It
does not require Gemini/OpenAI. The first implementation uses local text
extraction, chunking, and lexical search; embeddings can be added later behind
the same interface.

Ingest a PDF or folder:

```bash
python -m src.scientific_agent.rag ingest knowledgebase/papers/s41586-026-10644-y_reference.pdf
```

Search the local index:

```bash
python -m src.scientific_agent.rag search "co-scientist agents" --top-k 5
```

Ask a question using retrieved PDF chunks plus a local Ollama LLM:

```bash
python -m src.scientific_agent.rag answer \
  "What is the Co-Scientist multi-agent architecture?" \
  --provider ollama \
  --model llama3.2:1b \
  --top-k 3
```

The local index is stored in `.vector_store/` and is ignored by git.

Add knowledge files:

```text
knowledgebase/papers/*.txt
knowledgebase/forcefield_docs/*.txt
```

Then ask a question that mentions the topic:

```bash
python -m src.md_agent.agent "Design an NPT MD run for water at 300 K and 1 atm"
```

## Docker Tool

The agent can now select the Docker execution path:

```bash
python -m src.md_agent.agent --no-llm --engine docker "Run NVT MD for water at 300 K for 100 steps"
```

Docker mode requires Docker to be installed and the image to be built first.

Build the example MD container:

```bash
docker build -t md-agent-toy-md tools/md_docker_tool
```

Run the container manually:

```bash
docker run --rm \
  -v "$PWD/inputs:/inputs" \
  -v "$PWD/outputs:/outputs" \
  md-agent-toy-md \
  python /app/run_md.py --config /inputs/config.json --output /outputs/result.json
```

Later, replace `tools/md_docker_tool/run_md.py` with your custom MD code interface.

## External MD Folder

The project can also call the sibling `MD_simulation/MD.py` folder as an
external subprocess tool. The external tool now reads JSON config and writes
JSON result files through `MD_simulation/run_md_config.py`:

```bash
python -m src.md_agent.agent \
  --mode workflow \
  --no-llm \
  --engine external-md \
  "Run NVE MD using Particle-256/system.data for 5 steps with dt 0.005 and cutoff 2.5"
```

The agent converts the natural-language request into config and passes it to
the external MD wrapper. Results are written to `outputs/external_md_result.json`.

## Current Architecture

```text
User query
  -> agent.py
  -> workflow_orchestrator.py
  -> agent_registry.py
       -> md_setup_agent
       -> md_knowledge_agent
       -> md_runner_agent
       -> md_analysis_agent
       -> md_report_agent
  -> tool_registry.py / runner.py
  -> JSON report + outputs/latest_workflow_report.md
```

`tool_registry.py` describes the local Python tools that will later be wrapped
as LangChain tools or loaded by a more Discovery-like runtime.

## Bring Your Own Agent

Add an agent manifest:

```text
agents/my_agent/agent.yaml
```

Example:

```yaml
name: my_agent
description: Does one specialized part of a scientific workflow.
inputs:
  - workflow_state
outputs:
  - result
tools:
  - my_tool
```

Then add a Python handler in `agent_registry.py` and reference it from a
workflow file:

```yaml
steps:
  - name: my_step
    agent: my_agent
```

The orchestrator passes a shared state dictionary from step to step, so agents
can build long workflows without repeating manual work.

## Next Improvements

Recommended expansion order:

1. Connect real Docker execution from the runner.
2. Add plots and trajectory/log analysis: RDF, MSD, energy drift, temperature stability.
3. Replace keyword knowledge search with embeddings/RAG.
4. Add structured outputs for stricter LLM config generation.
5. Add force-field evaluation and optimization loops.
