# MD Agent Workbench

A small, from-scratch agentic AI workbench for molecular dynamics workflows.

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
│   ├── analysis.py
│   ├── knowledge.py
│   ├── langchain_agent.py
│   ├── llm.py
│   ├── parser.py
│   ├── runner.py
│   └── tool_registry.py
├── tools/md_docker_tool/
│   ├── Dockerfile
│   ├── run_md.py
│   └── tool.yaml
├── tools/knowledge_search/tool.yaml
├── tools/md_analysis/tool.yaml
├── inputs/
├── outputs/
└── requirements.txt
```

## Quick Start

Run the prototype locally:

```bash
python -m src.md_agent.agent --engine local "Run NVT MD for water at 300 K for 1000 steps using 1 fs timestep"
```

Run the deterministic workflow without any LLM:

```bash
python -m src.md_agent.agent --mode workflow --no-llm --engine local "Run NVT MD for water at 300 K"
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

To force local-only mode:

```bash
python -m src.md_agent.agent --no-llm --engine local "Run NVT MD for water at 300 K"
```

## LangChain Tool Agent

This mode is closer to your REDOX prototype: Gemini can choose tools instead of
`agent.py` always controlling the order.

Install dependencies:

```bash
pip install -r requirements.txt
```

Set a Gemini key. Either name works:

```bash
export GEMINI_API_KEY="your_api_key_here"
# or
export GOOGLE_API_KEY="your_api_key_here"
```

Run the tool-calling agent:

```bash
python -m src.md_agent.agent \
  --mode tool-agent \
  --engine local \
  --model gemini-2.0-flash \
  "Run an NVT MD simulation for water at 300 K for 100 steps and analyze it"
```

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
external subprocess tool:

```bash
python -m src.md_agent.agent \
  --mode workflow \
  --no-llm \
  --engine external-md \
  "Run NVE MD with my external MD code"
```

The current external script is still hardcoded and may run longer than the
agent timeout. The next refactor should make `MD_simulation` accept config JSON
and write structured result JSON so the agent can analyze physical outputs.

## Current Architecture

```text
User query
  -> agent.py
  -> parser.py
  -> knowledge.py
  -> optional llm.py
  -> runner.py
       -> local toy MD, or
       -> Docker tool: tools/md_docker_tool/run_md.py
  -> analysis.py
  -> JSON report
```

`tool_registry.py` describes the local Python tools that will later be wrapped
as LangChain tools or loaded by a more Discovery-like runtime.

## Next Improvements

Recommended expansion order:

1. Connect real Docker execution from the runner.
2. Add plots and trajectory/log analysis: RDF, MSD, energy drift, temperature stability.
3. Replace keyword knowledge search with embeddings/RAG.
4. Add structured outputs for stricter LLM config generation.
5. Add force-field evaluation and optimization loops.
