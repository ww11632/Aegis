"""FAQ Agent — answers questions grounded in the retrieved knowledge base."""

import logging

from aegis.agents.base import AgentResult, TraceStep
from aegis.llm import LLMClient
from aegis.rag.retriever import Retriever, format_context

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the FAQ agent for Aegis, an insurance customer support assistant.

Answer the customer's question using ONLY the numbered sources provided. Rules:
- Cite the sources you used inline, like [1] or [1][2].
- If the sources do not contain the answer, say so plainly and suggest contacting support.
  Do not fall back on general knowledge.
- Be concise: 2-5 sentences, plain language.
- Never invent policy numbers, prices, claim IDs, or legal guarantees.
- Do not give personalised financial or legal advice; describe how products generally work.
- Never promise an underwriting decision, a claim outcome, or a financial return. Describe
  how the process works and what it depends on, and point to a human advisor for anything
  that turns on the customer's own policy."""

UNGROUNDED_SYSTEM_PROMPT = """You are the FAQ agent for Aegis, an insurance support assistant.

The knowledge base is unavailable, so answer from general insurance knowledge. Rules:
- Be concise: 2-5 sentences, plain language.
- Say that the answer is general and that the customer should confirm against their policy.
- Never invent policy numbers, prices, claim IDs, or legal guarantees."""

NO_CONTEXT_REPLY = (
    "I couldn't find anything in our knowledge base that answers that. "
    "Our support team can help directly at 1-800-AEGIS."
)


class FAQAgent:
    """Retrieval-augmented answering over the FAQ knowledge base."""

    name = "faq"

    def __init__(self, llm: LLMClient, retriever: Retriever | None = None):
        self._llm = llm
        self._retriever = retriever

    async def run(self, message: str, *, session_id: str = "default") -> AgentResult:
        if self._retriever is None:
            # Degraded mode: the service still answers, and the trace says it was ungrounded.
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
                    )
                ],
            )

        chunks = await self._retriever.retrieve(message)
        trace = [
            TraceStep(
                agent=self.name,
                action="retrieve",
                detail=(
                    ", ".join(f"{c.faq_id}:{c.score:.3f}" for c in chunks)
                    if chunks
                    else "no chunks above the score floor"
                ),
            )
        ]
        if not chunks:
            trace.append(
                TraceStep(
                    agent=self.name,
                    action="abstain",
                    detail="no context to ground an answer",
                )
            )
            return AgentResult(reply=NO_CONTEXT_REPLY, trace=trace)

        prompt = f"Sources:\n{format_context(chunks)}\n\nCustomer question: {message}"
        reply = await self._llm.generate(prompt, system=SYSTEM_PROMPT, temperature=0.2)
        logger.info("faq_agent answered session=%s sources=%d", session_id, len(chunks))
        trace.append(
            TraceStep(
                agent=self.name,
                action="llm_answer",
                detail=f"grounded in {len(chunks)} source(s)",
            )
        )
        return AgentResult(reply=reply, trace=trace)
