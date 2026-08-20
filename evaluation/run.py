"""Run the Aegis evaluation suites and write results.

    python -m evaluation.run                      # every suite the environment supports
    python -m evaluation.run --suite pii          # one suite
    LLM_PROVIDER=fake python -m evaluation.run    # offline baseline, no API key needed

Suites that need a resource the environment does not have (an API key, a database) are
reported as skipped with the reason, so a partial run still produces a readable report.
"""

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from aegis.agents.base import AgentResult
from aegis.agents.supervisor import Supervisor
from aegis.config import settings
from aegis.guardrails.input_guard import check_input_async
from aegis.guardrails.patterns import find_injections, find_pii
from aegis.guardrails.risk_policy import find_risk_violations
from aegis.llm import build_llm_client
from aegis.rag.embeddings import build_embedder
from aegis.rag.ingest import ingest
from aegis.rag.retriever import Retriever
from aegis.rag.vector_store import PgVectorStore
from evaluation.metrics import binary_metrics, mean, recall_at_k, reciprocal_rank

logger = logging.getLogger("evaluation")

CASES_DIR = Path(__file__).parent / "cases"
RESULTS_DIR = Path(__file__).parent / "results"
SUITES = ("routing", "retrieval", "pii", "injection", "risk")


def load_cases(name: str) -> list[dict]:
    return json.loads((CASES_DIR / f"{name}.json").read_text())


@dataclass
class SuiteResult:
    """Outcome of one evaluation suite."""

    name: str
    status: str = "ok"  # "ok" | "skipped"
    n: int = 0
    headline: str = ""  # the single number this suite reports
    metrics: dict = field(default_factory=dict)
    failures: list[dict] = field(default_factory=list)
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "cases": self.n,
            "headline": self.headline,
            "metrics": self.metrics,
            "failures": self.failures,
            "reason": self.reason,
        }


class _NullAgent:
    """Stands in for a sub-agent: routing is measured without running the agents."""

    def __init__(self, name: str):
        self.name = name

    async def run(self, message: str, *, ctx=None) -> AgentResult:
        return AgentResult(reply="", trace=[])


# --- Suites -----------------------------------------------------------------------

async def run_routing(concurrency: int = 4) -> SuiteResult:
    """Does the supervisor send each message to the right agent?"""
    cases = load_cases("routing")
    try:
        llm = build_llm_client()
    except Exception as exc:
        return SuiteResult("routing", status="skipped", reason=str(exc))

    supervisor = Supervisor(
        llm, {"faq": _NullAgent("faq"), "recommendation": _NullAgent("recommendation")}
    )
    limit = asyncio.Semaphore(concurrency)

    async def classify(case: dict) -> tuple[dict, str, float]:
        async with limit:
            decision = await supervisor.route(case["message"])
        return case, decision.intent, decision.confidence

    try:
        rows = await asyncio.gather(*(classify(c) for c in cases))
    except Exception as exc:
        return SuiteResult("routing", status="skipped", reason=f"routing calls failed: {exc}")

    correct = [c for c, predicted, _ in rows if predicted == c["expected_agent"]]
    failures = [
        {
            "id": c["id"],
            "message": c["message"],
            "expected": c["expected_agent"],
            "predicted": predicted,
            "confidence": round(confidence, 3),
        }
        for c, predicted, confidence in rows
        if predicted != c["expected_agent"]
    ]
    accuracy = len(correct) / len(cases)
    per_class = {}
    for label in ("faq", "recommendation"):
        subset = [(c, p) for c, p, _ in rows if c["expected_agent"] == label]
        hits = sum(1 for c, p in subset if p == label)
        per_class[label] = round(hits / len(subset), 4) if subset else 0.0

    return SuiteResult(
        "routing",
        n=len(cases),
        headline=f"{accuracy:.1%}",
        metrics={
            "accuracy": round(accuracy, 4),
            "per_class_accuracy": per_class,
            "mean_confidence": round(mean([conf for _, _, conf in rows]), 4),
        },
        failures=failures,
    )


async def run_retrieval(top_k: int = 3) -> SuiteResult:
    """Do the top-k retrieved chunks contain the FAQ entry that answers the query?"""
    cases = load_cases("retrieval")
    try:
        embedder = build_embedder()
        store = await PgVectorStore.connect(dim=embedder.dim)
    except Exception as exc:
        return SuiteResult("retrieval", status="skipped", reason=f"vector store unavailable: {exc}")

    try:
        await ingest(store, embedder)
        retriever = Retriever(embedder, store)
        recalls, ranks, failures = [], [], []
        for case in cases:
            chunks = await retriever.retrieve(case["query"], top_k=top_k)
            retrieved = [c.faq_id for c in chunks]
            recall = recall_at_k(case["expected_faq_ids"], retrieved, top_k)
            recalls.append(recall)
            ranks.append(reciprocal_rank(case["expected_faq_ids"], retrieved))
            if recall < 1.0:
                failures.append(
                    {
                        "id": case["id"],
                        "query": case["query"],
                        "expected": case["expected_faq_ids"],
                        "retrieved": retrieved,
                        "scores": [round(c.score, 4) for c in chunks],
                    }
                )
    except Exception as exc:
        return SuiteResult("retrieval", status="skipped", reason=f"retrieval failed: {exc}")
    finally:
        await store.close()

    return SuiteResult(
        "retrieval",
        n=len(cases),
        headline=f"{mean(recalls):.1%}",
        metrics={
            f"recall_at_{top_k}": round(mean(recalls), 4),
            "mrr": round(mean(ranks), 4),
            "top_k": top_k,
            "embedder": embedder.name,
        },
        failures=failures,
    )


def run_pii() -> SuiteResult:
    """Does the guardrail find PII, and does it leave clean messages alone?"""
    cases = load_cases("pii")
    outcomes, failures, type_hits, type_total = [], [], 0, 0

    for case in cases:
        expected_types = sorted(set(case["expected_types"]))
        found_types = sorted({d.type for d in find_pii(case["text"])})
        outcomes.append((bool(expected_types), bool(found_types)))
        type_total += len(expected_types)
        type_hits += len(set(expected_types) & set(found_types))
        if expected_types != found_types:
            failures.append(
                {
                    "id": case["id"],
                    "text": case["text"],
                    "expected": expected_types,
                    "found": found_types,
                }
            )

    metrics = binary_metrics(outcomes)
    return SuiteResult(
        "pii",
        n=len(cases),
        headline=f"{metrics.recall:.1%} recall / {metrics.precision:.1%} precision",
        metrics={
            **metrics.as_dict(),
            "type_level_recall": round(type_hits / type_total, 4) if type_total else 0.0,
        },
        failures=failures,
    )


async def run_injection(with_classifier: bool = False) -> SuiteResult:
    """Does the guardrail block attacks without blocking real customers?

    Defaults to the pattern stage alone, which is deterministic and needs no API key.
    With `--with-classifier` the full two-stage guardrail is measured instead.
    """
    cases = load_cases("injection")
    llm = None
    if with_classifier:
        try:
            llm = build_llm_client()
        except Exception as exc:
            return SuiteResult("injection", status="skipped", reason=str(exc))

    outcomes, failures = [], []
    for case in cases:
        if llm is None:
            matched = [d.type for d in find_injections(case["text"])]
            predicted = bool(matched)
        else:
            result = await check_input_async(case["text"], llm=llm)
            predicted = result.blocked
            matched = [f.detail[:60] for f in result.flags if f.type == "injection"]
        outcomes.append((case["is_injection"], predicted))
        if predicted != case["is_injection"]:
            failures.append(
                {
                    "id": case["id"],
                    "text": case["text"],
                    "expected": case["is_injection"],
                    "predicted": predicted,
                    "matched": matched,
                }
            )

    metrics = binary_metrics(outcomes)
    return SuiteResult(
        "injection",
        n=len(cases),
        headline=f"{metrics.recall:.1%} recall / {metrics.precision:.1%} precision",
        metrics={
            **metrics.as_dict(),
            "stage": "patterns+classifier" if llm is not None else "patterns",
        },
        failures=failures,
    )


def run_risk() -> SuiteResult:
    """Does the output policy catch outcome promises without blocking correct replies?

    Cases are agent replies rather than customer messages: this measures the policy layer
    that sits between the model and the customer, not the model's own wording.
    """
    cases = load_cases("risk")
    outcomes, failures, type_hits, type_total = [], [], 0, 0

    for case in cases:
        expected = case["expected_violation"]
        found = [v.type for v in find_risk_violations(case["reply"])]
        outcomes.append((expected is not None, bool(found)))
        if expected is not None:
            type_total += 1
            type_hits += expected in found
        if (expected is not None) != bool(found) or (expected and expected not in found):
            failures.append(
                {
                    "id": case["id"],
                    "reply": case["reply"],
                    "expected": expected,
                    "found": found,
                }
            )

    metrics = binary_metrics(outcomes)
    return SuiteResult(
        "risk",
        n=len(cases),
        headline=f"{metrics.recall:.1%} recall / {metrics.precision:.1%} precision",
        metrics={
            **metrics.as_dict(),
            "correct_violation_type": round(type_hits / type_total, 4) if type_total else 0.0,
        },
        failures=failures,
    )


# --- Reporting --------------------------------------------------------------------

def _config() -> dict:
    return {
        "llm_provider": settings.llm_provider,
        "gemini_model": settings.gemini_model if settings.llm_provider == "gemini" else None,
        "embedding_model": (
            settings.embedding_model if settings.llm_provider == "gemini" else "lexical (offline)"
        ),
        "rag_top_k": settings.rag_top_k,
    }


def to_markdown(report: dict) -> str:
    config = report["config"]
    offline = config["llm_provider"] != "gemini"
    lines = [
        "# Aegis Evaluation Results",
        "",
        f"Run at {report['timestamp']} · provider `{config['llm_provider']}` · "
        f"embeddings `{config['embedding_model']}`",
        "",
    ]
    if offline:
        lines += [
            "> **Offline baseline.** The LLM is the deterministic fake client and embeddings are "
            "lexical, so routing and retrieval numbers are a floor, not a measurement of Gemini. "
            "The guardrail suites are rule-based and provider-independent — those numbers are "
            "the real ones.",
            "",
        ]
    lines += ["| Suite | Cases | Result | Detail |", "|---|---|---|---|"]
    for name, suite in report["suites"].items():
        if suite["status"] == "skipped":
            lines.append(f"| {name} | — | skipped | {suite['reason']} |")
            continue
        detail = ", ".join(
            f"{k}={v}" for k, v in suite["metrics"].items() if not isinstance(v, dict)
        )
        lines.append(f"| {name} | {suite['cases']} | {suite['headline']} | {detail} |")

    for name, suite in report["suites"].items():
        if suite.get("failures"):
            lines += ["", f"## {name} failures ({len(suite['failures'])})", ""]
            for failure in suite["failures"]:
                lines.append(f"- `{failure['id']}` " + json.dumps(
                    {k: v for k, v in failure.items() if k != "id"}, ensure_ascii=False
                ))
    return "\n".join(lines) + "\n"


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Aegis evaluation suites.")
    parser.add_argument("--suite", choices=(*SUITES, "all"), default="all")
    parser.add_argument("--top-k", type=int, default=settings.rag_top_k)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--with-classifier",
        action="store_true",
        help="measure the injection guardrail with its LLM second stage (needs an API key)",
    )
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "latest")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s | %(name)s | %(message)s")
    wanted = SUITES if args.suite == "all" else (args.suite,)

    results: list[SuiteResult] = []
    if "routing" in wanted:
        results.append(await run_routing(args.concurrency))
    if "retrieval" in wanted:
        results.append(await run_retrieval(args.top_k))
    if "pii" in wanted:
        results.append(run_pii())
    if "injection" in wanted:
        results.append(await run_injection(args.with_classifier))
    if "risk" in wanted:
        results.append(run_risk())

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": _config(),
        "suites": {r.name: r.as_dict() for r in results},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    args.out.with_suffix(".md").write_text(to_markdown(report))

    print(to_markdown(report))
    print(f"Wrote {args.out.with_suffix('.json')} and {args.out.with_suffix('.md')}")
    return 0 if all(r.status == "ok" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
