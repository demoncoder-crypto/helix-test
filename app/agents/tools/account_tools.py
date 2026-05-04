"""
Account tools — used by ``AccountAgent``.

Mock data is acceptable per the assignment spec: "the wiring is evaluated".
The data is **deterministic per ``user_id``** (we seed a local PRNG with
the user's hash) so the same caller sees consistent results across turns
— much more convincing in a demo than always-random output.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

_PIPELINES = ("backend-ci", "frontend-ci", "release", "nightly", "infra-deploy")
_BRANCHES = ("main", "develop", "feature/auth", "hotfix/login", "release/2026.04")
_STATUSES = ("passed", "failed", "cancelled")
_PLAN_LIMITS = {
    "free": (1, 5.0),
    "pro": (5, 50.0),
    "enterprise": (20, 500.0),
}


@dataclass
class BuildSummary:
    build_id: str
    pipeline: str
    status: str
    branch: str
    started_at: str  # ISO 8601
    duration_seconds: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AccountStatus:
    user_id: str
    plan_tier: str
    concurrent_builds_used: int
    concurrent_builds_limit: int
    storage_used_gb: float
    storage_limit_gb: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _seed_for(user_id: str) -> random.Random:
    h = hashlib.blake2b(user_id.encode("utf-8"), digest_size=8).digest()
    return random.Random(int.from_bytes(h, "big"))


async def get_recent_builds(user_id: str, limit: int = 5) -> list[dict[str, Any]]:
    """Return the most recent builds for a user, newest first.

    Mock data, deterministic per user. Returns a list of dicts (rather than
    dataclass instances) so ADK can serialize the tool result cleanly.
    """
    if limit <= 0:
        return []
    rng = _seed_for(user_id)
    now = datetime.now(UTC)
    out: list[dict[str, Any]] = []
    for i in range(limit):
        started = now - timedelta(minutes=rng.randint(5, 60) * (i + 1))
        out.append(
            BuildSummary(
                build_id=f"bld_{rng.randrange(10**6, 10**7):07d}",
                pipeline=rng.choice(_PIPELINES),
                status=rng.choices(_STATUSES, weights=[6, 3, 1])[0],
                branch=rng.choice(_BRANCHES),
                started_at=started.isoformat(),
                duration_seconds=rng.randint(30, 900),
            ).to_dict()
        )
    return out


async def get_account_status(user_id: str, plan_tier: str = "free") -> dict[str, Any]:
    """Return current account status (plan, usage limits)."""
    rng = _seed_for(user_id + ":status")
    limits = _PLAN_LIMITS.get(plan_tier, _PLAN_LIMITS["free"])
    builds_limit, storage_limit = limits
    return AccountStatus(
        user_id=user_id,
        plan_tier=plan_tier,
        concurrent_builds_used=rng.randint(0, builds_limit),
        concurrent_builds_limit=builds_limit,
        storage_used_gb=round(rng.uniform(0, storage_limit * 0.8), 2),
        storage_limit_gb=storage_limit,
    ).to_dict()
