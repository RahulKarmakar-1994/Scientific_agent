# Lessons From Microsoft Discovery

Microsoft Discovery's public repository is primarily a catalog and community
surface, not the full internal platform implementation. The useful lesson for
this project is the product structure: agents and tools are described with
metadata, schemas, documentation, and starter-kit manifests.

## What To Learn

Discovery separates the platform into catalog entries:

```text
agents/<agent-name>/
  metadata.yaml
  agent.yaml
  README.md
  tools/<tool-name>/
    tool.yaml
    Dockerfile

starter-kits/<kit-name>/
  kit.json
```

For this project, the analogous product direction is:

```text
src/scientific_agent/
  core/
  agents/
  domains/

tools/
  <tool-name>/
    tool.yaml
    run.py
    Dockerfile optional

starter_kits/
  physics-learning/
  molecular-simulation/
```

## Agent Contract

Discovery-style agents have:

- `metadata.yaml`: identity, version, tags, publisher/support, associated tools
- `agent.yaml`: model settings, instructions, tool wiring, knowledge bases
- `README.md`: user-facing explanation, usage, limitations

Our equivalent should become:

```text
agents/<agent-name>/
  metadata.yaml
  agent.yaml
  README.md
```

The Python implementation can stay under `src/scientific_agent/agents/`, but
the catalog manifest should describe what the agent does.

## Tool Contract

Discovery tools are container-oriented and action-based:

```yaml
name: tool-name
description: What this tool does
version: 1.0.0
category: physics
infra:
  - name: tool-container
    infra_type: container
    image: ...
    compute:
      min_resources: {cpu: 1, ram: 2}
      max_resources: {cpu: 4, ram: 8}
actions:
  - name: run_action
    description: Run one specific action
    input_schema:
      type: object
      properties: ...
```

Our `tools/*/tool.yaml` should evolve toward this:

- versioned tool definition
- explicit category
- one or more actions
- input schema
- output schema
- command or container runtime
- resource/risk metadata

## Prompt Design Lesson

Discovery's scientific agents are highly specialized. For example, simulation
agents include:

- strict response format
- exact helper libraries to use
- workflow order
- validation checks
- failure handling
- final result schema

For our physics-learning product, small local LLMs need the same style:

```text
LearningDemoAgent:
  Your only job is concept explanation + safe Python demo generation.
  Return strict JSON.
  Print output; do not write files directly.
  Use retrieved physics context when available.
  If code is unreliable, return unreliable instead of unrelated output.
```

## Starter Kits

Discovery starter kits bundle multiple agents into a launchable scenario with
sample prompts and expected outputs.

For this project, starter kits should become product presets:

```text
physics-learning
  RouterAgent
  LearningDemoAgent
  RAGAgent
  PythonDemoRunner

molecular-simulation
  SimulationAgent
  RAGAgent
  MD/GROMACS/LAMMPS tools
```

## Product Direction

Do not try to make one giant agent. Build a catalog of small, well-described
agents and tools:

```text
RouterAgent
LearningDemoAgent
RAGAgent
SimulationAgent
ReportAgent
PythonDemoRunner
GROMACSRunner
LAMMPSRunner
```

Each should have:

- manifest
- clear prompt/instructions
- input/output schema
- examples
- limitations
- job artifacts

## Next Changes For This Repo

1. Add catalog manifests for current product agents.
2. Upgrade `tools/python_demo_runner/tool.yaml` with version, category, actions,
   input schema, and output schema.
3. Add a `starter_kits/physics-learning/kit.json` with sample prompts.
4. Add validation for local manifests, similar to Discovery's schema checks.
5. Add RAG-grounded code generation in `LearningDemoAgent`.

The immediate product goal remains:

```text
User asks to learn a physics concept
-> route to LearningDemoAgent
-> retrieve relevant physics context
-> generate safe Python demo
-> run in sandbox/tool
-> validate output
-> save job with generated_code.py, report.md, artifacts/
```
