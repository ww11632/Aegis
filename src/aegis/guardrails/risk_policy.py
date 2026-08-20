"""Domain risk policy for outbound replies.

Toxicity and PII are generic. In insurance the expensive failure is different: a reply
that reads as a promise. The policy this module enforces is:

    Recommendation and FAQ outputs must not imply guaranteed eligibility, underwriting
    approval, claim acceptance, investment returns, or coverage outcome. Responses are
    informational guidance and must surface material uncertainty where relevant.

Two tiers, because the failures are not equally severe:
- A stated guarantee is a compliance failure — the reply is replaced, not annotated.
- A product recommendation with no uncertainty language is incomplete rather than wrong,
  so a standard disclosure is appended.

Insurance vocabulary makes naive matching wrong: "guaranteed issue" and "guaranteed
renewable" are genuine product features, not promises to the customer, so they are
allow-listed and covered by negative evaluation cases.
"""

import re
from typing import NamedTuple


class RiskFinding(NamedTuple):
    """One policy violation found in a reply."""

    type: str
    match: str


# Product features that legitimately contain "guaranteed" — these describe the contract,
# not an outcome promised to this customer.
PRODUCT_TERM_RE = re.compile(
    r"guaranteed\s+(issue|renewable|renewability|death benefit|cash value|"
    r"premium|premiums|level premium|minimum)",
    re.I,
)

RISK_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("guaranteed_underwriting", re.compile(
        r"\b(guarantee|guaranteed|guarantees)\b[^.]{0,40}"
        r"\b(approval|approved|acceptance|accepted|eligibility|eligible|qualify|"
        r"underwriting|insurable|coverage|covered)\b",
        re.I)),
    ("guaranteed_underwriting", re.compile(
        r"\byou\b[^.]{0,20}\b(will|are) (definitely |certainly )?"
        r"(be )?(approved|accepted|eligible|insurable)\b|"
        r"\byou (will|definitely) qualify\b|"
        r"\bwe (will|can) approve\b",
        re.I)),
    ("guaranteed_claim", re.compile(
        r"\b(guarantee|guaranteed|guarantees)\b[^.]{0,40}"
        r"\b(claim|claims|payout|payment|reimbursement|paid out|settlement)\b|"
        r"\byour claim will (be paid|be approved|always be)\b|"
        r"\ball claims are (paid|approved|covered)\b|"
        r"\bwill (always|certainly|definitely) be (paid|covered|reimbursed)\b",
        re.I)),
    ("guaranteed_return", re.compile(
        r"\b(guarantee|guaranteed|guarantees|assured)\b[^.]{0,30}"
        r"\b(return|returns|profit|profits|gain|gains|income|yield|growth)\b|"
        r"\b(risk[- ]free|no risk|zero risk|cannot lose|can't lose)\b|"
        r"\byou (will|are guaranteed to) (make|earn) money\b",
        re.I)),
    ("absolute_coverage", re.compile(
        r"\b(100%|fully|always|never) (covered|reimbursed|protected)\b|"
        r"\bcovers (everything|all damage|any claim)\b|"
        r"\bno exclusions\b|\bthere are no exceptions\b",
        re.I)),
]

# Any of these means the reply already carries uncertainty; no disclosure is appended.
# A hedge in front of the match flips its meaning: "I can't confirm whether your claim
# will be approved" is the model behaving correctly, and blocking it would be worse than
# useless. Only the span immediately before the match is considered.
HEDGE_RE = re.compile(
    r"\b(whether|if|cannot|can't|can not|unable to|don't know|do not know|no|not|never|"
    r"unlikely|depends on|subject to)\b[^.]{0,40}$",
    re.I,
)
HEDGE_WINDOW = 60


DISCLOSURE_MARKER_RE = re.compile(
    r"\b(underwriting|eligibility|subject to|depends on|may vary|not a quote|"
    r"exclusion|exclusions|waiting period|terms and conditions|policy terms|"
    r"confirm with|check your policy|general information)\b",
    re.I,
)

DISCLOSURE = (
    "This is general product information, not an offer of cover — eligibility, pricing, "
    "and claim outcomes depend on underwriting and the policy terms."
)

GUARANTEE_REPLY = (
    "I can share what our products cover and how they generally work, but I can't promise "
    "an underwriting decision, a claim outcome, or a financial return — those depend on "
    "underwriting and the policy terms. Our advisors can confirm your specific situation "
    "at 1-800-AEGIS."
)


def _masked_product_terms(text: str) -> str:
    """Blank out legitimate product features so they cannot trigger a violation."""
    return PRODUCT_TERM_RE.sub("PRODUCT_FEATURE", text)


def find_risk_violations(text: str) -> list[RiskFinding]:
    """Find statements that promise an outcome the business cannot promise."""
    scrubbed = _masked_product_terms(text)
    findings: list[RiskFinding] = []
    seen: set[str] = set()
    for name, pattern in RISK_PATTERNS:
        for match in pattern.finditer(scrubbed):
            preceding = scrubbed[max(0, match.start() - HEDGE_WINDOW) : match.start()]
            if HEDGE_RE.search(preceding):
                continue
            if name not in seen:
                seen.add(name)
                findings.append(RiskFinding(name, match.group()))
            break
    return findings


def needs_disclosure(text: str) -> bool:
    """True when a reply presents products without any uncertainty language."""
    return not DISCLOSURE_MARKER_RE.search(text)
