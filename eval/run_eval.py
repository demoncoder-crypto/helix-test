"""
Eval harness — Extension E7.

Reads ``eval/golden.jsonl``, hits a running Helix SROP server, and prints:

* **routing accuracy** — per-route and overall (knowledge / account / refusal)
* **retrieval hit-rate@k** — for knowledge queries, fraction of cases where
  at least one retrieved ``chunk_id`` belongs to the expected source doc.
* **recall@1, recall@3, recall@5** — same definition, restricted to the
  top-1 / top-3 / top-5 retrieved chunks (in trace order, which is rank
  order). Useful to demonstrate the E4 reranker's lift.

The chunk-id → source mapping is computed by reading the live Chroma
collection directly (``coll.get(ids=...)``), so the harness is decoupled
from any internal schema and only needs the public REST surface.

Usage::

    python eval/run_eval.py
    python eval/run_eval.py --base http://127.0.0.1:8765
    python eval/run_eval.py --tag baseline --out eval/results/baseline.json

Two-condition demo (reranker off vs on)::

    RERANKER_ENABLED=false python eval/run_eval.py --tag baseline
    RERANKER_ENABLED=true  python eval/run_eval.py --tag reranked

Tip: stop and start uvicorn between runs so the new ``RERANKER_ENABLED``
takes effect — pydantic-settings reads ``.env`` at import.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE = os.environ.get("EVAL_BASE_URL", "http://127.0.0.1:8765")
DEFAULT_GOLDEN = Path(__file__).parent / "golden.jsonl"
RECALL_KS = (1, 3, 5)


@dataclass
class GoldenRow:
    id: str
    query: str
    expected_route: str
    expected_source_substr: str | None


@dataclass
class TurnResult:
    row: GoldenRow
    routed_to: str
    trace_id: str | None
    retrieved_chunk_ids: list[str]
    retrieved_sources: list[str]
    latency_ms: int
    error: str | None = None

    @property
    def route_correct(self) -> bool:
        if self.error:
            return False
        return self.routed_to == self.row.expected_route

    @property
    def source_hit(self) -> bool | None:
        """``None`` when the row is not source-checkable (e.g. account/refusal)."""
        if self.row.expected_source_substr is None:
            return None
        if self.error:
            return False
        return any(self.row.expected_source_substr in s for s in self.retrieved_sources)

    def source_hit_at_k(self, k: int) -> bool | None:
        if self.row.expected_source_substr is None:
            return None
        if self.error:
            return False
        return any(
            self.row.expected_source_substr in s for s in self.retrieved_sources[:k]
        )


@dataclass
class EvalReport:
    tag: str
    base_url: str
    total: int = 0
    by_route: dict[str, dict[str, int]] = field(default_factory=dict)
    routing_correct: int = 0
    knowledge_total: int = 0
    knowledge_source_hits: int = 0
    recall_at_k: dict[int, dict[str, int]] = field(default_factory=dict)
    avg_latency_ms: float = 0.0
    errors: int = 0
    rows: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "base_url": self.base_url,
            "total": self.total,
            "routing_accuracy": (
                round(self.routing_correct / self.total, 4) if self.total else 0.0
            ),
            "by_route": self.by_route,
            "knowledge_hit_rate": (
                round(self.knowledge_source_hits / self.knowledge_total, 4)
                if self.knowledge_total
                else 0.0
            ),
            "recall_at_k": {
                f"recall@{k}": (
                    round(v["hits"] / v["total"], 4) if v["total"] else 0.0
                )
                for k, v in sorted(self.recall_at_k.items())
            },
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "errors": self.errors,
            "rows": self.rows,
        }


def load_golden(path: Path) -> list[GoldenRow]:
    rows: list[GoldenRow] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            rows.append(
                GoldenRow(
                    id=obj["id"],
                    query=obj["query"],
                    expected_route=obj["expected_route"],
                    expected_source_substr=obj.get("expected_source_substr"),
                )
            )
    return rows


def lookup_chunk_sources(chunk_ids: list[str]) -> list[str]:
    """Resolve chunk_ids to their source filenames via direct Chroma lookup.

    Returns sources in the **same order** as the input ids (so rank is preserved
    for recall@k). Missing ids resolve to ``""``.
    """
    if not chunk_ids:
        return []
    from app.rag.vector_store import _get_collection_sync

    coll = _get_collection_sync()
    raw = coll.get(ids=chunk_ids, include=["metadatas"])
    got_ids = list(raw.get("ids") or [])
    got_metas = list(raw.get("metadatas") or [])
    by_id = {cid: (meta or {}) for cid, meta in zip(got_ids, got_metas)}
    return [str(by_id.get(cid, {}).get("source", "")) for cid in chunk_ids]


async def run_one_turn(
    client: httpx.AsyncClient,
    session_id: str,
    row: GoldenRow,
    request_timeout: float,
) -> TurnResult:
    t0 = time.perf_counter()
    try:
        chat_resp = await client.post(
            f"/v1/chat/{session_id}",
            json={"content": row.query},
            timeout=request_timeout,
        )
    except httpx.TimeoutException:
        return TurnResult(
            row=row,
            routed_to="<timeout>",
            trace_id=None,
            retrieved_chunk_ids=[],
            retrieved_sources=[],
            latency_ms=int((time.perf_counter() - t0) * 1000),
            error="client_timeout",
        )
    if chat_resp.status_code >= 400:
        return TurnResult(
            row=row,
            routed_to=f"<{chat_resp.status_code}>",
            trace_id=None,
            retrieved_chunk_ids=[],
            retrieved_sources=[],
            latency_ms=int((time.perf_counter() - t0) * 1000),
            error=f"http_{chat_resp.status_code}",
        )
    chat = chat_resp.json()
    trace_id = chat.get("trace_id")
    routed_to = chat.get("routed_to") or "<unknown>"

    retrieved_chunk_ids: list[str] = []
    if trace_id:
        trace_resp = await client.get(f"/v1/traces/{trace_id}", timeout=request_timeout)
        if trace_resp.status_code == 200:
            tr = trace_resp.json()
            retrieved_chunk_ids = list(tr.get("retrieved_chunk_ids") or [])

    retrieved_sources = lookup_chunk_sources(retrieved_chunk_ids)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    return TurnResult(
        row=row,
        routed_to=routed_to,
        trace_id=trace_id,
        retrieved_chunk_ids=retrieved_chunk_ids,
        retrieved_sources=retrieved_sources,
        latency_ms=latency_ms,
    )


def aggregate(results: list[TurnResult], tag: str, base_url: str) -> EvalReport:
    report = EvalReport(tag=tag, base_url=base_url, total=len(results))
    report.recall_at_k = {k: {"hits": 0, "total": 0} for k in RECALL_KS}
    latencies: list[int] = []
    for r in results:
        latencies.append(r.latency_ms)
        if r.error:
            report.errors += 1
        bucket = report.by_route.setdefault(
            r.row.expected_route, {"total": 0, "correct": 0}
        )
        bucket["total"] += 1
        if r.route_correct:
            bucket["correct"] += 1
            report.routing_correct += 1
        if r.row.expected_source_substr is not None:
            report.knowledge_total += 1
            if r.source_hit:
                report.knowledge_source_hits += 1
            for k in RECALL_KS:
                report.recall_at_k[k]["total"] += 1
                if r.source_hit_at_k(k):
                    report.recall_at_k[k]["hits"] += 1
        report.rows.append(
            {
                "id": r.row.id,
                "query": r.row.query,
                "expected_route": r.row.expected_route,
                "routed_to": r.routed_to,
                "route_ok": r.route_correct,
                "expected_source_substr": r.row.expected_source_substr,
                "retrieved_sources": r.retrieved_sources,
                "source_hit": r.source_hit,
                "latency_ms": r.latency_ms,
                "error": r.error,
            }
        )
    report.avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0
    return report


def print_report(report: EvalReport) -> None:
    d = report.to_dict()
    print()
    print("=" * 64)
    print(f" Helix SROP eval — tag: {report.tag}")
    print(f" base url       : {report.base_url}")
    print(f" total queries  : {d['total']}")
    print(f" routing acc    : {d['routing_accuracy'] * 100:5.1f}%  "
          f"({report.routing_correct}/{report.total})")
    for route, b in d["by_route"].items():
        acc = (b["correct"] / b["total"]) if b["total"] else 0.0
        print(f"   - {route:<10} {acc * 100:5.1f}%  ({b['correct']}/{b['total']})")
    print(f" knowledge hit  : {d['knowledge_hit_rate'] * 100:5.1f}%  "
          f"({report.knowledge_source_hits}/{report.knowledge_total})")
    for label, val in d["recall_at_k"].items():
        print(f"   - {label:<10} {val * 100:5.1f}%")
    print(f" avg latency    : {d['avg_latency_ms']:.0f} ms")
    print(f" errors         : {report.errors}")
    print("=" * 64)


async def run_eval(
    base_url: str, golden_path: Path, tag: str, request_timeout: float
) -> EvalReport:
    rows = load_golden(golden_path)
    print(f"[eval] loaded {len(rows)} golden rows from {golden_path}")
    print(f"[eval] target  : {base_url}")
    print(f"[eval] timeout : {request_timeout}s per request")

    async with httpx.AsyncClient(base_url=base_url, timeout=request_timeout) as client:
        sess_resp = await client.post(
            "/v1/sessions",
            json={"user_id": f"u_eval_{int(time.time())}", "plan_tier": "pro"},
        )
        sess_resp.raise_for_status()
        session_id = sess_resp.json()["session_id"]
        print(f"[eval] session : {session_id}")

        results: list[TurnResult] = []
        for i, row in enumerate(rows, 1):
            r = await run_one_turn(client, session_id, row, request_timeout)
            mark = "OK" if r.route_correct else "  "
            hit = ""
            if r.row.expected_source_substr is not None:
                hit = "  [HIT]" if r.source_hit else "  [MISS]"
            print(
                f"[{i:>2}/{len(rows)}] {mark}  {r.row.id}  "
                f"route={r.routed_to:<10}  {r.latency_ms:>5}ms  "
                f"chunks={len(r.retrieved_chunk_ids)}{hit}"
            )
            results.append(r)

    return aggregate(results, tag=tag, base_url=base_url)


def main() -> int:
    ap = argparse.ArgumentParser(description="Helix SROP eval harness")
    ap.add_argument("--base", default=DEFAULT_BASE, help="server base URL")
    ap.add_argument("--golden", default=str(DEFAULT_GOLDEN), help="path to golden.jsonl")
    ap.add_argument("--tag", default="run", help="label for this run")
    ap.add_argument("--out", default=None, help="optional path to write JSON report")
    ap.add_argument(
        "--request-timeout", type=float, default=120.0,
        help="per-request HTTP timeout in seconds",
    )
    args = ap.parse_args()

    report = asyncio.run(
        run_eval(
            base_url=args.base,
            golden_path=Path(args.golden),
            tag=args.tag,
            request_timeout=args.request_timeout,
        )
    )
    print_report(report)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"[eval] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
