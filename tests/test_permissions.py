"""Permission tiers and the approval path for external side effects."""

import pytest

from aegis.agents.recommendation import RecommendationAgent
from aegis.harness.context import AgentContext
from aegis.harness.permissions import (
    ApprovalStore,
    Level,
    PermissionSet,
    check,
    describe_call,
)
from aegis.llm import FakeLLMClient
from tests.test_agents import RecordingLLM, StubCatalog


class CallbackSeekingLLM(RecordingLLM):
    """A planner that always wants to raise a callback — the external tool."""

    async def generate_structured(self, prompt, *, schema, system="", temperature=0.0):
        if "action" in schema.model_fields:
            return schema.model_validate(
                {
                    "action": "request_advisor_callback",
                    "product_id": "prod-001",
                    "reason": "wants to talk it through",
                    "thought": "the customer asked for a person",
                }
            )
        return await FakeLLMClient().generate_structured(
            prompt, schema=schema, system=system, temperature=temperature
        )


def test_tools_are_classified_by_what_they_can_damage():
    perms = PermissionSet.default()

    assert check("search_products", perms).level is Level.READ
    assert check("get_product_details", perms).level is Level.READ
    assert check("request_advisor_callback", perms).level is Level.EXTERNAL


def test_reads_run_without_asking():
    verdict = check("search_products", PermissionSet.default())

    assert verdict.allowed is True
    assert verdict.requires_approval is False


def test_external_side_effects_require_approval():
    verdict = check("request_advisor_callback", PermissionSet.default())

    assert verdict.allowed is True
    assert verdict.requires_approval is True


def test_an_unregistered_tool_is_denied():
    verdict = check("drop_all_policies", PermissionSet.default())

    assert verdict.denied is True
    assert "not in the permission registry" in verdict.reason


def test_a_read_only_permission_set_refuses_everything_else():
    perms = PermissionSet.read_only()

    assert check("search_products", perms).allowed is True
    assert check("request_advisor_callback", perms).denied is True


def test_describe_call_renders_the_call_for_a_human():
    assert describe_call("search_products", {"query": "travel", "limit": 3}) == (
        "search_products(limit=3, query='travel')"
    )


def test_approvals_can_only_be_resolved_once():
    store = ApprovalStore()
    approval = store.create("request_advisor_callback", {"product_id": "prod-001"})

    assert store.get(approval.id) is not None
    assert store.resolve(approval.id).id == approval.id
    assert store.resolve(approval.id) is None
    assert len(store) == 0


def test_expired_approvals_disappear(monkeypatch):
    from aegis.config import settings

    monkeypatch.setattr(settings, "approval_ttl_seconds", -1)
    store = ApprovalStore()
    approval = store.create("request_advisor_callback", {})

    assert approval.expired is True
    assert store.get(approval.id) is None


async def test_the_agent_stops_and_previews_instead_of_acting():
    catalog = StubCatalog([{"id": "prod-001", "name": "SafeTravel Plus"}])
    ctx = AgentContext(approvals=ApprovalStore())

    result = await RecommendationAgent(CallbackSeekingLLM(), catalog=catalog).run(
        "can someone call me about prod-001?", ctx=ctx
    )

    assert result.stop_reason == "awaiting_approval"
    assert result.pending_approval is not None
    assert result.pending_approval.tool == "request_advisor_callback"
    assert "prod-001" in result.pending_approval.preview
    assert "approval_required" in [s.action for s in result.trace]
    assert catalog.calls == []  # nothing left the system
    assert len(ctx.approvals) == 1


async def test_a_tainted_request_is_refused_rather_than_queued_for_approval():
    catalog = StubCatalog([{"id": "prod-001", "name": "SafeTravel Plus"}])
    ctx = AgentContext(approvals=ApprovalStore())
    ctx.taint("product_catalog: neutralised 1 span(s)")

    result = await RecommendationAgent(CallbackSeekingLLM(), catalog=catalog).run(
        "can someone call me?", ctx=ctx
    )

    actions = [s.action for s in result.trace]
    assert "tool_denied" in actions
    assert "approval_required" not in actions
    assert len(ctx.approvals) == 0  # a human is never asked to approve tainted work
    assert result.stop_reason != "awaiting_approval"


async def test_approval_is_refused_when_no_store_is_wired():
    """Without somewhere to record the request, the call is dropped, not run."""
    catalog = StubCatalog([{"id": "prod-001"}])

    result = await RecommendationAgent(CallbackSeekingLLM(), catalog=catalog).run(
        "call me", ctx=AgentContext(approvals=None)
    )

    assert "tool_denied" in [s.action for s in result.trace]
    assert catalog.calls == []
    assert result.pending_approval is None


@pytest.mark.parametrize("tool", ["search_products", "get_product_details"])
def test_read_tools_survive_tainting(tool):
    assert check(tool, PermissionSet.default(), tainted=True).allowed is True


async def test_an_external_call_runs_directly_when_approval_is_switched_off(monkeypatch):
    """The permission level still classifies it; only the approval requirement is off."""
    from aegis.config import settings

    monkeypatch.setattr(settings, "require_approval_for_external", False)

    class CallbackCatalog(StubCatalog):
        def __init__(self):
            super().__init__([])
            self.callbacks = []

        async def request_advisor_callback(self, product_id="", reason="", contact_hint=""):
            self.callbacks.append((product_id, reason))
            return {"ticket_id": "cb-123", "status": "queued"}

    catalog = CallbackCatalog()
    ctx = AgentContext(approvals=ApprovalStore(), max_steps=2)

    result = await RecommendationAgent(CallbackSeekingLLM(), catalog=catalog).run(
        "call me about prod-001", ctx=ctx
    )

    assert catalog.callbacks == [("prod-001", "wants to talk it through")]
    assert result.stop_reason != "awaiting_approval"
    assert len(ctx.approvals) == 0
