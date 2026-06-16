# Scientific Agent Workbench Architecture

The repository is intentionally organized so it can stay one product now and
split into packages later.

## Current Product

```text
Scientific Agent Workbench
  core orchestration
  MD domain prototype
  tool manifests
  agent manifests
  workflow YAML files
```

## Future Split Boundary

When the project grows, split along these boundaries:

```text
scientific-agent-core
  agent registry
  tool registry
  workflow orchestrator
  LLM providers
  execution backends

scientific-agent-md
  MD parser
  MD runners
  MD analysis tools
  GROMACS/LAMMPS/custom-MD adapters

scientific-agent-learning
  statistical physics demos
  teaching workflows
  generated plots and lesson reports

scientific-agent-app
  CLI/API/UI layer
```

## Stable Interface

External tools should use this contract:

```text
natural language request
  -> validated config JSON
  -> tool execution
  -> result JSON
  -> analysis/report
```

Keep tool inputs and outputs structured so a tool can move between local Python,
subprocess, Docker, or HPC execution without changing the agent interface.
