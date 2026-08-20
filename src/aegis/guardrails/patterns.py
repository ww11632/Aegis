"""Detection patterns shared by the input and output guardrails.

Deliberately deterministic: regex detection is cheap, auditable, and reproducible in
evaluation. The LLM classifier in `input_guard` is the second stage for paraphrased
injections, but it does not replace this layer — these patterns also run over retrieved
documents and tool output, where a model call per source would be too expensive.
"""

import re
from typing import NamedTuple


class Detection(NamedTuple):
    """One matched span."""

    type: str
    match: str
    start: int
    end: int


# --- PII -------------------------------------------------------------------------

EMAIL_RE = re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
# US SSN, rejecting the ranges that are never issued.
SSN_RE = re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")
# 13-19 digits, optionally grouped by spaces or hyphens; validated with Luhn below.
CARD_RE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
# +1 (415) 555-0132, 0912-345-678, 415-555-0132 — at least 9 digits total.
PHONE_RE = re.compile(
    r"(?<!\w)(?<!\d\.)(?:\+\d{1,3}[ -]?)?(?:\(\d{2,4}\)[ -]?)?"
    r"\d{2,4}(?:[ -]\d{2,4}){1,3}(?!\w)(?!\.\d)"
)

PII_MASKS = {
    "email": "[EMAIL]",
    "ssn": "[SSN]",
    "credit_card": "[CREDIT_CARD]",
    "phone": "[PHONE]",
}


def _luhn_ok(digits: str) -> bool:
    """Luhn checksum — keeps order numbers and policy ids from being flagged as cards."""
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _digit_count(text: str) -> int:
    return sum(ch.isdigit() for ch in text)


def find_pii(text: str) -> list[Detection]:
    """Find PII spans, most specific pattern first so matches do not overlap."""
    found: list[Detection] = []
    claimed: list[tuple[int, int]] = []

    def claim(kind: str, m: re.Match) -> None:
        if any(m.start() < end and start < m.end() for start, end in claimed):
            return
        claimed.append((m.start(), m.end()))
        found.append(Detection(kind, m.group(), m.start(), m.end()))

    for m in EMAIL_RE.finditer(text):
        claim("email", m)
    for m in SSN_RE.finditer(text):
        claim("ssn", m)
    for m in CARD_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            claim("credit_card", m)
    for m in PHONE_RE.finditer(text):
        if 9 <= _digit_count(m.group()) <= 15:
            claim("phone", m)

    return sorted(found, key=lambda d: d.start)


def mask_pii(text: str) -> tuple[str, list[Detection]]:
    """Replace detected PII with type placeholders."""
    detections = find_pii(text)
    out, cursor = [], 0
    for d in detections:
        out.append(text[cursor : d.start])
        out.append(PII_MASKS.get(d.type, "[REDACTED]"))
        cursor = d.end
    out.append(text[cursor:])
    return "".join(out), detections


# --- Prompt injection ------------------------------------------------------------

INJECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("instruction_override", re.compile(
        r"\b(ignore|disregard|forget|override)\b[^.]{0,40}\b"
        r"(previous|prior|earlier|above|all|any|your)\b[^.]{0,20}"
        r"\b(instruction|instructions|prompt|prompts|rule|rules|direction|directions|context)\b",
        re.I)),
    ("system_prompt_extraction", re.compile(
        r"\b(show|print|reveal|repeat|output|tell me|give me|what (is|are))\b[^.]{0,40}"
        r"\b(system prompt|system message|initial instructions|your instructions|your prompt|"
        r"your rules|api key|secret)\b",
        re.I)),
    ("role_override", re.compile(
        r"\b(you are now|from now on you are|act as|pretend to be|roleplay as|"
        r"simulate being|switch to)\b[^.]{0,40}"
        r"\b(dan|admin|administrator|developer|root|jailbroken|unrestricted|"
        r"unfiltered|no restrictions|different ai|another ai)\b",
        re.I)),
    ("delimiter_injection", re.compile(
        r"(</?(system|assistant|user)>|\[/?(system|inst)\]|###\s*(system|instruction))", re.I)),
    ("policy_evasion", re.compile(
        r"\b(without|bypass|ignore|disable|turn off|skip)\b[^.]{0,30}"
        r"\b(restriction|restrictions|filter|filters|guardrail|guardrails|safety|"
        r"rules|policy|policies)\b",
        re.I)),
    ("data_exfiltration", re.compile(
        r"\b(list|dump|show|export|send|email)\b[^.]{0,40}"
        r"\b(all (the )?(customer|user|client|policyholder)s?|other (customers|users|clients)|"
        r"database|records|ssn|social security numbers)\b",
        re.I)),
]


def find_injections(text: str) -> list[Detection]:
    """Find prompt-injection attempts — first hit per pattern, enough to block on."""
    return [
        Detection(name, m.group(), m.start(), m.end())
        for name, pattern in INJECTION_PATTERNS
        for m in [pattern.search(text)]
        if m
    ]


def find_all_injections(text: str) -> list[Detection]:
    """Every injection span, ordered by position.

    `find_injections` stops at the first hit per pattern because a single hit is enough
    to reject a user message. Neutralising injected instructions inside a document needs
    all of them, so this variant scans exhaustively.
    """
    found = [
        Detection(name, m.group(), m.start(), m.end())
        for name, pattern in INJECTION_PATTERNS
        for m in pattern.finditer(text)
    ]
    # Overlapping patterns can match the same sentence; keep the earliest, longest span.
    found.sort(key=lambda d: (d.start, -(d.end - d.start)))
    kept: list[Detection] = []
    for d in found:
        if any(d.start < k.end and k.start < d.end for k in kept):
            continue
        kept.append(d)
    return kept
