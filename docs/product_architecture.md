# Product Architecture

This project should evolve as an agentic scientific learning platform, not as a
collection of hardcoded topic branches.

## Core Principle

The LLM decides intent, context, plans, explanations, and code candidates.
Python enforces product safety, persistence, tool boundaries, and validation.

In other words:

```text
LLM: understand, plan, explain, generate candidates
Python: retrieve, validate, execute trusted tools, save jobs, enforce safety
```

## Current Agent Loop

```text
user message
-> session memory lookup
-> RequestUnderstandingAgent
   - is this a follow-up?
   - what is the standalone request?
   - what search queries should RAG use?
-> GroundingService
   - run concise RAG searches
   - return grounded/weak/missing evidence
-> specialized domain agent prompt
-> optional trusted tool execution
-> PhysicsVerifierAgent
-> final answer with sources, artifacts, and job id
```

## Current Product Layers

- `ScientificAgent`: top-level orchestrator for routing, job persistence,
  session memory, verification, and final reporting.
- `RequestUnderstandingAgent`: converts natural language and short follow-ups
  into standalone requests and RAG search queries.
- `GroundingService`: retrieves local evidence from the PDF/text index.
- `LearningDemoAgent`: teaches concepts and, only when requested, generates and
  runs Python demos through the trusted runner.
- `PhysicsVerifierAgent`: checks draft answers against evidence/tool results and
  flags obvious physics contradictions.

## Product Boundaries

- Agent behavior lives in `agents/*/agent.yaml` and `system_prompt.md`.
- Tools live under `tools/*/tool.yaml`.
- Retrieval/services live under `src/scientific_agent/core/`.
- Jobs live under `outputs/jobs/`.
- Conversations live under `outputs/sessions/`.
- Large PDFs and generated vector stores are not committed.

## What Should Not Happen

- Do not add one Python `if topic == ...` branch for every physics topic.
- Do not let the LLM execute arbitrary shell commands.
- Do not present unverified generated code as trusted science.
- Do not let old session context dominate a new self-contained question.

## Acceptable Guardrails

Some deterministic checks are product safety, not anti-agentic hardcoding:

- code execution policy
- import/call safety checks
- JSON schema normalization
- job/session persistence
- source citation formatting
- fallback behavior when a local model returns invalid JSON

## Next Product Step

Improve the evidence and verification loop:

```text
candidate answer + retrieved evidence + optional generated code
-> verifier checks factual support, equations, and code/artifact consistency
-> bad answers are corrected, cautioned, or blocked before presentation
```

This trust layer is what moves the repo from a demo toward a product.
