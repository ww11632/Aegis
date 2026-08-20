"""Output guardrail — PII, prompt leakage, and domain risk policy.

Layered by severity: a stated guarantee or a leaked prompt replaces the reply, PII is
masked in place, and a recommendation with no uncertainty language gets the standard
disclosure appended. See `risk_policy` for the domain policy itself.
"""

import logging
import re

from aegis.guardrails.input_guard import Flag, GuardrailResult
from aegis.guardrails.patterns import mask_pii
from aegis.guardrails.risk_policy import (
    DISCLOSURE,
    GUARANTEE_REPLY,
    find_risk_violations,
    needs_disclosure,
)

logger = logging.getLogger(__name__)

# Phrases that only appear if the model is reciting its own configuration.
LEAK_PATTERNS = [
    re.compile(r"\bsystem (prompt|instruction)s?\b[:\s]", re.I),
    re.compile(r"^\s*you are the (faq|recommendation|supervisor) agent for aegis", re.I),
]

LEAK_REPLY = (
    "I ran into a problem generating that answer. Please rephrase your question, or "
    "contact our support team at 1-800-AEGIS."
)

# Agents whose replies present products and therefore carry the disclosure requirement.
DISCLOSURE_REQUIRED_AGENTS = {"recommendation"}


def check_output(reply: str, agent: str = "") -> GuardrailResult:
    """Screen an agent reply before it is returned to the customer."""
    for pattern in LEAK_PATTERNS:
        if pattern.search(reply):
            logger.warning("Blocked output: possible prompt leakage")
            return GuardrailResult(
                text=LEAK_REPLY,
                blocked=True,
                flags=[Flag(type="prompt_leak", detail="reply recited agent instructions")],
            )

    violations = find_risk_violations(reply)
    if violations:
        # A promise cannot be repaired by appending a caveat, so the reply is replaced.
        logger.warning("Blocked output: risk policy — %s", [v.type for v in violations])
        return GuardrailResult(
            text=GUARANTEE_REPLY,
            blocked=True,
            flags=[
                Flag(type="risk_policy", detail=f"{v.type}: {v.match[:80]}") for v in violations
            ],
        )

    text, detections = mask_pii(reply)
    flags = [Flag(type="pii", detail=f"{d.type} masked on output") for d in detections]
    if detections:
        logger.warning("Masked %d PII span(s) on output", len(detections))

    if agent in DISCLOSURE_REQUIRED_AGENTS and needs_disclosure(text):
        text = f"{text.rstrip()}\n\n{DISCLOSURE}"
        flags.append(Flag(type="disclosure", detail="standard risk disclosure appended"))

    return GuardrailResult(text=text, flags=flags)
