"""Domain risk policy: replies must not promise outcomes the business cannot promise."""

import pytest

from aegis.guardrails.output_guard import check_output
from aegis.guardrails.risk_policy import DISCLOSURE, find_risk_violations, needs_disclosure


@pytest.mark.parametrize(
    "reply,expected_type",
    [
        ("Yes, we guarantee your claim will be paid.", "guaranteed_claim"),
        ("Your claim will be paid, no matter what happened.", "guaranteed_claim"),
        ("All claims are approved within ten business days.", "guaranteed_claim"),
        ("You will definitely be approved for this policy.", "guaranteed_underwriting"),
        ("This plan guarantees acceptance regardless of history.", "guaranteed_underwriting"),
        ("We can approve you today for FamilyShield Term 20.", "guaranteed_underwriting"),
        ("This policy offers guaranteed returns on your premiums.", "guaranteed_return"),
        ("It's a risk-free way to grow your savings.", "guaranteed_return"),
        ("With HomeSecure Complete you are always covered.", "absolute_coverage"),
        ("AutoGuard Premium covers everything, no exclusions.", "absolute_coverage"),
    ],
)
def test_guarantee_language_is_detected(reply, expected_type):
    findings = find_risk_violations(reply)

    assert [f.type for f in findings][:1] == [expected_type]


@pytest.mark.parametrize(
    "reply",
    [
        "FamilyShield Term 20 is guaranteed renewable for the full 20-year term.",
        "The plan has guaranteed level premiums and a guaranteed death benefit.",
        "This is a guaranteed issue policy, so no medical exam is required.",
        "Claims are usually paid within 10 business days once documents are received.",
        "Eligibility and pricing depend on underwriting and your medical history.",
        "Travel insurance generally covers trip cancellation, subject to the policy exclusions.",
    ],
)
def test_legitimate_insurance_language_is_not_flagged(reply):
    assert find_risk_violations(reply) == []


def test_guarantee_replaces_the_reply_rather_than_annotating_it():
    result = check_output("We guarantee you will be approved and your claim will be paid.")

    assert result.blocked
    assert result.flag_types[0] == "risk_policy"
    assert "guarantee" not in result.text.lower() or "can't promise" in result.text
    assert "underwriting" in result.text


def test_recommendation_without_uncertainty_gets_the_disclosure():
    result = check_output("SafeTravel Plus costs $45 a month.", agent="recommendation")

    assert not result.blocked
    assert result.flag_types == ["disclosure"]
    assert result.text.endswith(DISCLOSURE)


def test_recommendation_that_already_hedges_is_left_alone():
    reply = "SafeTravel Plus costs $45 a month; final pricing depends on underwriting."
    result = check_output(reply, agent="recommendation")

    assert result.text == reply
    assert result.flags == []


def test_faq_replies_do_not_get_the_product_disclosure():
    reply = "Travel insurance usually covers trip cancellation and medical emergencies [1]."
    result = check_output(reply, agent="faq")

    assert result.text == reply
    assert result.flags == []


def test_disclosure_marker_detection():
    assert needs_disclosure("SafeTravel Plus costs $45 a month.")
    assert not needs_disclosure("Cover is subject to the policy exclusions.")


def test_risk_policy_outranks_pii_masking():
    """A guaranteed-outcome reply is replaced, so there is no PII left to mask."""
    result = check_output("We guarantee approval — email us at agent@example.com.")

    assert result.blocked
    assert result.flag_types == ["risk_policy"]
    assert "agent@example.com" not in result.text
