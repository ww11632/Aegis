"""FAQ Agent — answers questions grounded in the retrieved knowledge base.

The retrieval half is itself a small loop: retrieve, look at what came back, and when
nothing clears the score floor, reformulate the question once and try again before
abstaining. Empty retrieval is usually a vocabulary mismatch rather than a missing
answer — "why is my car insurance so expensive" shares no words with "What factors
affect my car insurance premium?" — so one revision is worth the extra call, and the
trace records both attempts.

Retrieved entries are untrusted content: they are neutralised and fenced before they
reach the prompt, so a poisoned knowledge-base row is data the model reads, never
instructions it follows.
"""

import logging

from pydantic import BaseModel, Field

from aegis.agents.base import AgentResult, TraceStep
from aegis.guardrails.untrusted import UNTRUSTED_SYSTEM_RULE, sanitize_and_fence
from aegis.harness.context import AgentContext
from aegis.harness.cost import BudgetExceeded, measure
from aegis.llm import LLMClient, LLMError
from aegis.rag.retriever import Retriever, format_context

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = f"""You are the FAQ agent for Aegis, an insurance customer support assistant.

Answer the customer's question using ONLY the numbered sources provided. Rules:
- Cite the sources you used inline, like [1] or [1][2].
- If the sources do not contain the answer, say so plainly and suggest contacting support.
  Do not fall back on general knowledge.
- Be concise: 2-5 sentences, plain language.
- Never invent policy numbers, prices, claim IDs, or legal guarantees.
- Do not give personalised financial or legal advice; describe how products generally work.
- Never promise an underwriting decision, a claim outcome, or a financial return. Describe
  how the process works and what it depends on, and point to a human advisor for anything
  that turns on the customer's own policy.

{UNTRUSTED_SYSTEM_RULE}"""

UNGROUNDED_SYSTEM_PROMPT = """You are the FAQ agent for Aegis, an insurance support assistant.

The knowledge base is unavailable, so answer from general insurance knowledge. Rules:
- Be concise: 2-5 sentences, plain language.
- Say that the answer is general and that the customer should confirm against their policy.
- Never invent policy numbers, prices, claim IDs, or legal guarantees."""

REPHRASE_SYSTEM_PROMPT = """You rewrite a customer's question as a knowledge-base search query.

The first search found nothing, most likely because the customer's words differ from the
wording used in the knowledge base. Rewrite the question in the vocabulary an insurance
FAQ would use — the formal term for what they described — keeping the same meaning.
Return the rewritten query only."""

NO_CONTEXT_REPLY = (
    "I couldn't find anything in our knowledge base that answers that. "
    "Our support team can help directly at 1-800-AEGIS."
)


class SearchQuery(BaseModel):
    """A reformulated retrieval query."""

    query: str = Field(description="The rewritten question, in knowledge-base vocabulary")


class FAQAgent:
    """Retrieval-augmented answering over the FAQ knowledge base."""

    name = "faq"

    def __init__(self, llm: LLMClient, retriever: Retriever | None = None):
        self._llm = llm
        self._retriever = retriever

    async def run(self, message: str, *, ctx: AgentContext | None = None) -> AgentResult:
        ctx = ctx or AgentContext()
        if self._retriever is None:
            return await self._answer_ungrounded(message)

        trace: list[TraceStep] = []
        query = self._contextual_query(message, ctx)

        with measure() as m:
            chunks = await self._retriever.retrieve(query)
        trace.append(
            TraceStep(
                agent=self.name,
                action="retrieve",
                detail=self._retrieval_detail(chunks),
                **m.trace_fields(),
            )
        )

        # Observe: nothing cleared the floor. Revise the query once, then act again.
        if not chunks and self._can_afford(ctx):
            rewritten = await self._rephrase(message, trace)
            if rewritten and rewritten.lower() != query.lower():
                with measure() as m:
                    chunks = await self._retriever.retrieve(rewritten)
                trace.append(
                    TraceStep(
                        agent=self.name,
                        action="retrieve_retry",
                        detail=f"query={rewritten!r} — {self._retrieval_detail(chunks)}",
                        **m.trace_fields(),
                    )
                )

        if not chunks:
            trace.append(
                TraceStep(
                    agent=self.name,
                    action="abstain",
                    detail="no context to ground an answer",
                )
            )
            return AgentResult(reply=NO_CONTEXT_REPLY, trace=trace, stop_reason="abstained")

        screened = sanitize_and_fence(format_context(chunks), source="knowledge_base")
        if screened.tainted:
            ctx.taint(screened.summary)
            trace.append(
                TraceStep(agent=self.name, action="untrusted_neutralised", detail=screened.summary)
            )

        parts = []
        if ctx.history:
            parts.append(f"Conversation so far:\n{ctx.history_prompt()}")
        parts.append(f"Sources:\n{screened.text}")
        parts.append(f"Customer question: {message}")

        with measure() as m:
            reply = await self._llm.generate(
                "\n\n".join(parts), system=SYSTEM_PROMPT, temperature=0.2
            )
        logger.info("faq_agent answered session=%s sources=%d", ctx.session_id, len(chunks))
        trace.append(
            TraceStep(
                agent=self.name,
                action="llm_answer",
                detail=f"grounded in {len(chunks)} source(s)",
                **m.trace_fields(),
            )
        )
        return AgentResult(reply=reply, trace=trace)

    # --- helpers ----------------------------------------------------------------------

    def _contextual_query(self, message: str, ctx: AgentContext) -> str:
        """Prepend the last customer turn so follow-ups retrieve on their real subject.

        "What about for a family?" carries no retrievable terms on its own.
        """
        previous = [t for t in ctx.history if t.role == "user"]
        if not previous or len(message.split()) > 6:
            return message
        return f"{previous[-1].content} {message}"

    @staticmethod
    def _retrieval_detail(chunks) -> str:
        return (
            ", ".join(f"{c.faq_id}:{c.score:.3f}" for c in chunks)
            if chunks
            else "no chunks above the score floor"
        )

    @staticmethod
    def _can_afford(ctx: AgentContext) -> bool:
        """The retry costs one model call; skip it when the request cannot pay for it."""
        try:
            ctx.meter.ensure_within_budget()
        except BudgetExceeded:
            return False
        return True

    async def _rephrase(self, message: str, trace: list[TraceStep]) -> str:
        """Rewrite the question in knowledge-base vocabulary. Returns '' on failure."""
        try:
            with measure() as m:
                rewritten = await self._llm.generate_structured(
                    f"Question: {message}", schema=SearchQuery, system=REPHRASE_SYSTEM_PROMPT
                )
        except LLMError:
            logger.exception("Query reformulation failed")
            trace.append(
                TraceStep(agent=self.name, action="rephrase_error", detail="reformulation failed")
            )
            return ""
        trace.append(
            TraceStep(
                agent=self.name,
                action="rephrase",
                detail=f"retry with {rewritten.query!r}",
                **m.trace_fields(),
            )
        )
        return rewritten.query

    async def _answer_ungrounded(self, message: str) -> AgentResult:
        """Degraded mode: the knowledge base is unreachable."""
        with measure() as m:
            reply = await self._llm.generate(
                message, system=UNGROUNDED_SYSTEM_PROMPT, temperature=0.2
            )
        return AgentResult(
            reply=reply,
            trace=[
                TraceStep(
                    agent=self.name,
                    action="llm_answer",
                    detail="retriever unavailable — answered without grounding",
                    **m.trace_fields(),
                )
            ],
            stop_reason="degraded",
        )
