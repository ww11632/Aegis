# Aegis Architecture

## System Overview

Aegis is a multi-agent GenAI system for insurance customer support. It covers agent
orchestration, retrieval-augmented generation, tool integration over MCP, and a layered
guardrail design, with a quantitative evaluation harness over all of it.

Implemented: the FastAPI service, supervisor routing, both sub-agents with their grounding
sources (pgvector and MCP), input/output guardrails, and the evaluation harness. Not yet
built and marked *(roadmap)* below: session memory, A2A, observability, and streaming.

## Data Flow

```
1. POST /chat
2. Input guardrail
   a. Regex patterns: PII spans are masked; injection matches block the request outright.
   b. If the patterns cleared it and the classifier is enabled: an LLM verdict on
      manipulation attempts. Classifier failure degrades to (a) instead of failing.
3. Supervisor classifies intent with Gemini structured output -> {intent, confidence, reasoning}
   - classifier call fails -> default to the FAQ agent, recorded in the trace
4. Sub-agent handles the request:
   - FAQ Agent: embed query -> pgvector cosine search (top-k) -> Gemini answers from the
     retrieved sources only, with [n] citations. No chunks retrieved -> abstain, no LLM call.
   - Recommendation Agent: Gemini plans the tool arguments -> MCP `search_products` over
     stdio -> Gemini recommends from the returned catalog rows only.
5. Output guardrail: prompt-leak check, domain risk policy (outcome promises), PII scrub,
   and a standard disclosure appended to product recommendations that carry no uncertainty
   language.
6. Response returns reply, agent_used, guardrail_flags, and the full execution trace.
```

## Component Details

### Supervisor Agent

The supervisor classifies intent with Gemini's structured output (JSON schema), so routing
is a parsed enum rather than free text that has to be string-matched:

```python
# src/aegis/agents/supervisor.py
class RoutingDecision(BaseModel):
    intent: Literal["faq", "recommendation"]
    confidence: float  # 0.0 - 1.0
    reasoning: str
```

Few-shot examples in the system prompt cover the ambiguous middle ("I just bought a house,
what do I need?" → recommendation). Confidence below 0.5 is recorded as its own trace step;
with session memory in place, that is where a clarification turn belongs.

### RAG Pipeline

1. **Ingest** (`python -m aegis.rag.ingest`): each FAQ entry becomes one chunk — the entries
   are short and self-contained, so splitting them would separate questions from answers.
   Chunks are embedded with `text-embedding-004` (`RETRIEVAL_DOCUMENT`) and upserted on
   `faq_id`, making re-ingestion idempotent.
2. **Retrieve**: the query is embedded with `RETRIEVAL_QUERY` — the asymmetric task types
   matter for retrieval quality — then matched with cosine distance (`<=>`) against an HNSW
   index built with `vector_cosine_ops`, so the query path can actually use the index.
3. **Generate**: retrieved chunks are rendered as numbered sources; the system prompt allows
   answers *only* from those sources and requires `[n]` citations.

Vectors are passed as text literals with an explicit `::vector` cast rather than through a
binary codec, which keeps the store free of a numpy dependency.

Why pgvector over FAISS: persistent across restarts, SQL-queryable metadata filtering, and
the same database can hold operational data.

### MCP Tool Integration

`aegis.tools.product_catalog` is an MCP server (`MCPServer`, stdio transport) exposing
`search_products` and `get_product_details`. `aegis.tools.mcp_client` connects to it as a
client over a child process, so the tool boundary is a real protocol boundary rather than a
Python import.

The session is owned by a dedicated background task: anyio task groups must be entered and
exited from the same task, and FastAPI's startup and shutdown hooks give no such guarantee.

Tool arguments are planned by the model (structured output → `CatalogQuery`), not parsed
from keywords, so budget and product-type filters come from what the customer actually said.

### Guardrails

Input and output guardrails are a cross-cutting layer, not agent code, so enforcement is
identical regardless of which agent runs and can be evaluated on its own.

| Stage | Mechanism | Policy |
|-------|-----------|--------|
| Input PII | Regex, Luhn-validated cards | Mask and continue — the customer still gets an answer, the raw identifiers never reach the model or the logs |
| Input injection | Six pattern families | Block before any agent runs |
| Input injection (stage 2) | LLM classifier, structured verdict | Block; skipped when the patterns already fired, and degrades to stage 1 on failure |
| Output | PII regex + prompt-leak patterns | Mask PII; replace the reply entirely on a suspected prompt leak |
| Output risk policy | Outcome-promise patterns, product-term allow-list, hedge detection | Replace the reply on a guarantee; append the standard disclosure to recommendations with no uncertainty language |

#### Domain risk policy

The policy the output guard enforces:

> Recommendation and FAQ outputs must not imply guaranteed eligibility, underwriting
> approval, claim acceptance, investment returns, or coverage outcome. Responses are
> informational guidance and must surface material uncertainty where relevant.

Two tiers, because the failures differ in severity. A stated guarantee is a compliance
failure, so the reply is replaced rather than annotated — a caveat appended to "your claim
will be paid" does not undo the promise. A recommendation that names products without any
uncertainty language is incomplete rather than wrong, so the standard disclosure is
appended and flagged.

Insurance vocabulary makes naive matching wrong in both directions. "Guaranteed renewable",
"guaranteed issue", and "guaranteed level premiums" are contract features, not promises to
this customer, so they are allow-listed. And a hedge in front of a match flips its meaning:
"I can't confirm whether your claim will be approved" is the model behaving correctly, so
matches preceded by a hedge are skipped. Both directions have negative evaluation cases.

The agents' system prompts carry the same rules, so the guard is the enforcement layer
rather than the only defence.

PII scope is email, phone, SSN, and credit card. Addresses and dates of birth are
deliberately out of scope for the MVP and are covered by negative evaluation cases so the
boundary is explicit rather than accidental.

### Failure Modes

Each dependency degrades one capability instead of failing the request, and the trace says
which: no database → FAQ answers ungrounded; no MCP server → recommendations without catalog
data; routing call fails → default to FAQ; tool call fails → "no match" reply; classifier
fails → patterns only.

## Evaluation Framework

`python -m evaluation.run` executes five suites over 116 cases in `evaluation/cases/` and
writes JSON and Markdown reports to `evaluation/results/`.

| Suite | What it measures | How |
|-------|-----------------|-----|
| Routing | Supervisor picks the right agent | Predicted vs. expected agent, plus per-class accuracy and mean confidence |
| Retrieval | The answer-bearing FAQ is retrieved | Recall@k and MRR against expected `faq_id`s |
| PII | Detection without over-masking | Precision/recall/F1 at message level, plus type-level recall |
| Injection | Attacks blocked, customers not | Precision/recall/F1 over attacks and benign lookalikes |
| Risk | Outcome promises caught, correct replies not | Precision/recall/F1 over agent replies, plus whether the right violation type was identified |

Suites that need an unavailable resource (API key, database) are reported as skipped with the
reason, so a partial run still produces a readable report. Every failing case is written to
the report with its inputs and outputs — the failure list is the point, not the headline
number.

## Roadmap

Session memory, A2A inter-agent protocol, OpenTelemetry tracing, SSE streaming, and CI/CD are
not implemented. The trade-offs section of the README explains what each would change.
