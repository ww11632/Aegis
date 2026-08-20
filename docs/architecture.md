# Aegis Architecture

## System Overview

Aegis is a multi-agent GenAI system for insurance customer support. It covers agent
orchestration, retrieval-augmented generation, tool integration over MCP, and a layered
guardrail design, with a quantitative evaluation harness over all of it.

The system splits in two. The **agents** decide what to do; the **harness** decides what
they are allowed to do, what it costs, and what is recorded about it:

    harness = context + tools + state + permission + trace + eval

Implemented: the FastAPI service, supervisor routing, both sub-agents with their grounding
sources (pgvector and MCP), the execution loop, session memory, the permission and approval
layer, per-request cost accounting, input/output guardrails, untrusted-content handling, and
the evaluation harness. Not yet built and marked *(roadmap)* below: A2A, OpenTelemetry
observability, and streaming.

## Data Flow

```
1. POST /chat
2. Input guardrail
   a. Regex patterns: PII spans are masked; injection matches block the request outright.
   b. If the patterns cleared it and the classifier is enabled: an LLM verdict on
      manipulation attempts. Classifier failure degrades to (a) instead of failing.
3. Context assembly: session history and task state are loaded, and a budget, a permission
   set, and a cost meter are attached for this request.
4. Supervisor classifies intent with Gemini structured output -> {intent, confidence, reasoning}
   - earlier turns are supplied, so a follow-up is routed in their light
   - classifier call fails -> default to the FAQ agent, recorded in the trace
5. Sub-agent handles the request:
   - FAQ Agent: embed query -> pgvector cosine search (top-k). Nothing above the floor ->
     reformulate the query once and retry -> still nothing -> abstain without an LLM call.
     Retrieved entries are neutralised and fenced, then Gemini answers from them alone with
     [n] citations.
   - Recommendation Agent: an execution loop — plan a step, call the tool, read the result,
     plan again — bounded by the step ceiling and the budget. Each tool call is checked
     against the permission registry first; an external side effect is previewed and held
     for approval instead of being run.
6. Output guardrail: prompt-leak check, domain risk policy (outcome promises), PII scrub,
   and a standard disclosure appended to product recommendations that carry no uncertainty
   language.
7. Memory: the masked user turn and the guarded reply are written to session memory, along
   with task state (last agent, stop reason, outstanding approval).
8. Response returns reply, agent_used, guardrail_flags, stop_reason, any pending approval,
   the full execution trace with per-step cost, and the request's cost summary.
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
`search_products`, `get_product_details`, and `request_advisor_callback`.
`aegis.tools.mcp_client` connects to it as a client over a child process, so the tool
boundary is a real protocol boundary rather than a Python import.

The first two are reads; the third leaves the system and exists to make the permission layer
real rather than theoretical — a permission model with nothing but read tools cannot be
tested. It queues a callback for a human advisor, and it only runs after approval.

The session is owned by a dedicated background task: anyio task groups must be entered and
exited from the same task, and FastAPI's startup and shutdown hooks give no such guarantee.

Tool calls are planned by the model (structured output → `NextStep`), not parsed from
keywords, so budget and product-type filters come from what the customer actually said —
and so does the choice of *which* tool to call at each step of the loop.

### The Harness

`src/aegis/harness/` holds what surrounds the model. Agents receive one `AgentContext` per
request rather than a growing list of keyword arguments, so a new harness concern does not
change every agent signature:

```python
# src/aegis/harness/context.py
@dataclass
class AgentContext:
    session_id: str
    history: list[Turn]          # what was said before
    state: dict                  # task state carried between turns
    permissions: PermissionSet   # what this request may call
    meter: CostMeter             # what it has spent, and its ceiling
    approvals: ApprovalStore | None
    tainted: bool                # untrusted content reached the context
```

#### Execution loop

The recommendation agent plans, acts, observes, and revises until something stops it. Four
things can, and each is recorded as `stop_reason`:

| Stop reason | Cause | What the customer gets |
|---|---|---|
| `answered` | the model has what it needs | the recommendation |
| `max_steps` | the step ceiling (`MAX_AGENT_STEPS`, default 4) | an answer from what was gathered, and the trace says it was cut short |
| `budget` | calls, tokens, or cost ran out mid-task | the same, plus a `budget_stop` step |
| `awaiting_approval` | the next step is an external side effect | a preview of the call and a request to confirm |

Two smaller guards sit inside the loop. A repeated call — the same tool with the same
arguments — is not executed twice; the loop records `loop_guard` and tells the planner to
choose differently, which is what stops a runaway from spending its whole budget on one
query. A tool error is an observation, not an exception: it goes back to the planner, which
can try a different call or answer without it.

The FAQ agent runs a smaller version of the same idea: retrieve, observe an empty result,
reformulate the query once, retrieve again, then abstain. Empty retrieval is usually a
vocabulary mismatch, which the evaluation quantifies — three of the twelve retrieval cases
fail for exactly that reason.

#### Permissions

Tools are classified by what a wrong call can damage, and each level gets a different
control:

| Level | Tools | Control |
|---|---|---|
| `read` | `search_products`, `get_product_details` | allowed, every call recorded in the trace |
| `local_write` | — | allowed within the system's own storage, recorded |
| `execute` | — | allowed under a timeout and the request budget |
| `external` | `request_advisor_callback` | previewed, held, and run only after a human approves |

Two rules sit on top of the levels. **An unregistered tool is denied** — a tool nobody
classified is a tool nobody reasoned about. And **a tainted request is dropped to
read-only**: if untrusted content had to be neutralised anywhere in this request, nothing
above a read may run, approval or not. That is the link between the injection defence and
the permission layer — external content must not be able to reach the step where an
injection becomes an incident.

Approval is a two-call protocol, because the agent cannot close this loop itself:

```
POST /chat            -> stop_reason "awaiting_approval", pending_approval {id, preview}
POST /approvals/{id}  -> {"decision": "approve"}  -> the call runs, ticket returned
                         {"decision": "deny"}     -> the call is discarded
```

Resolving consumes the approval, so a held call can never run twice, and approvals expire
(`APPROVAL_TTL_SECONDS`, default 600). The store is process-local: a multi-instance
deployment needs it in shared storage so any worker can resolve what another raised.

#### Cost and budget

Every model call records its usage into the meter bound to the current request through a
`ContextVar`, so agents do not thread a counter through their signatures. `measure()`
snapshots the meter around a block and reports the delta plus wall-clock latency, which is
what lets each trace step carry its own `duration_ms`, `tokens_in`, `tokens_out`, and
`cost_usd`.

The budget (`MAX_LLM_CALLS`, `MAX_TOTAL_TOKENS`, `MAX_COST_USD`) is checked before starting
work the request can no longer afford — the ceiling that turns "the loop kept going" from an
invoice into a trace step. Prices come from a table in `harness/cost.py` and are
configuration to refresh, not a fact the code can verify. When a provider reports no usage
metadata the call is still counted, from a character estimate, flagged `estimated` rather
than recorded as free.

#### Session memory

`src/aegis/memory/session.py` implements two of the three memory kinds: conversation turns
and small task state, both in PostgreSQL with an in-process fallback. Long-term memory is
deliberately absent — it needs a retention policy and a deletion path before it should hold
anything about a customer, and `DELETE /sessions/{id}` is that path for what does exist.

What gets written is the *guarded* text on both sides: the user turn after PII masking, the
reply after the output policy has run. Raw identifiers are never persisted, and a reply the
policy replaced is remembered as the replacement.

History earns its cost in two places: the supervisor routes a follow-up in the light of
earlier turns, and the FAQ agent prefixes a short follow-up with the previous question so it
has something to retrieve on — "what about for a family?" carries no retrievable terms of
its own.

#### Untrusted content

Retrieved FAQ entries and MCP tool results arrive mid-request, long after the input guard
has run, and reach the model as prompt text. `guardrails/untrusted.py` treats them as what
they are — data from outside the trust boundary:

1. **Neutralise**: injected instructions are replaced in place, so the model never reads
   them, while the surrounding facts survive. Removing the whole source would silently
   delete the grounding the answer depends on.
2. **Fence**: what is left is wrapped in `<untrusted_data source=...>`, whose meaning is
   stated in every system prompt that can see one. Content that tries to close the fence
   early, or to open a role block of its own, is neutralised too.
3. **Taint**: the request is marked, the trace records it, and the permission layer drops
   the request to read-only for the rest of its life.

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
| Untrusted content | Exhaustive injection scan over retrieved documents and tool results | Neutralise the spans in place, fence the remainder, and taint the request to read-only |

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
which: no database → FAQ answers ungrounded and session memory falls back to in-process; no
MCP server → recommendations without catalog data; routing call fails → default to FAQ; tool
call fails → observed by the loop, which answers without it; classifier fails → patterns
only; planner fails → the loop stops rather than calling tools it could not plan.

The harness adds three stops of its own, and none of them is an error: the budget ends a run
and answers from what was gathered, the step ceiling does the same, and an external side
effect stops to ask. A held call with nowhere to record it is refused rather than run.

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

The A2A inter-agent protocol, OpenTelemetry tracing, SSE streaming, and CI/CD are not
implemented. The trade-offs section of the README explains what each would change.

Within the harness, the honest gaps are: the approval store and the budget are per-process,
so both need shared storage before a second instance is deployed; permission levels are
per-request rather than per-user, so real ACLs are still to come; and there is no semantic
cache or model routing, which are the two cost levers the budget only measures.
