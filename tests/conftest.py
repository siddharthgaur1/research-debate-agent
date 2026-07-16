"""Shared fixtures. No test in this suite touches the network.

Env vars are set before `src.config` is imported anywhere, because settings are
validated at import time and cached — a test process without them would fail on
collection rather than on assertion.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.setdefault("TAVILY_API_KEY", "tvly-test")
os.environ.setdefault("SERPAPI_KEY", "serp-test")
os.environ.setdefault("SEARCH_PROVIDER", "tavily")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

from src.state.schema import (  # noqa: E402
    BiasReport,
    RunState,
    Source,
    Subtask,
    TokenUsage,
    new_run_state,
)
from src.tools.llm import Budget  # noqa: E402


@pytest.fixture
def fake_llm(monkeypatch):
    """Route Budget through canned responses, keyed by output schema.

    Every agent reaches the model through `Budget`, so patching its two methods
    makes the whole suite offline with no per-agent mocking ceremony.
    """

    def install(text: str = "canned reply", structured: dict | None = None):
        structured = structured or {}

        def _text(self, system, user, **kwargs):
            self.usage.append(TokenUsage(node="fake", model="gpt-4o", cost_usd=0.001))
            return text

        def _structured(self, system, user, schema, **kwargs):
            self.usage.append(TokenUsage(node="fake", model="gpt-4o", cost_usd=0.001))
            if schema not in structured:
                raise AssertionError(f"No canned response for {schema.__name__}")
            return structured[schema]

        monkeypatch.setattr(Budget, "text", _text)
        monkeypatch.setattr(Budget, "structured", _structured)

    return install


def _iso(days_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).date().isoformat()


@pytest.fixture(autouse=True)
def _settings(tmp_path, monkeypatch):
    """Point every run at a temp dir and reset the settings cache per test."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "runs.db"))
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.delenv("CHROMA_HOST", raising=False)

    from src.config import get_settings

    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _no_redis(monkeypatch):
    """Streaming is fire-and-forget; tests must not need a live Redis."""
    monkeypatch.setattr("src.persistence.stream.publish_turn", lambda *a, **k: None)
    monkeypatch.setattr("src.persistence.stream.publish_done", lambda *a, **k: None)


@pytest.fixture
def fixture_sources() -> list[Source]:
    """Three independent, credible sources that broadly agree."""
    return [
        Source(
            id="S1",
            url="https://www.bls.gov/remote-work-productivity",
            title="Remote work and measured output among knowledge workers",
            snippet="Output per worker rose 4% among remote knowledge workers.",
            text="A longitudinal study of remote knowledge workers found output rose 4%.",
            domain="bls.gov",
            published=_iso(120),
            subtask_id="T1",
        ),
        Source(
            id="S2",
            url="https://www.nature.com/articles/remote-productivity",
            title="Measured output among remote knowledge workers rose",
            snippet="Remote knowledge workers showed a 4% rise in output per worker.",
            text="Peer-reviewed analysis reporting a 4% output rise for remote workers.",
            domain="nature.com",
            published=_iso(200),
            subtask_id="T1",
        ),
        Source(
            id="S3",
            url="https://someguy.substack.com/p/remote-is-a-scam",
            title="Remote work destroyed my team's output",
            snippet="Anecdotally our output collapsed after going remote.",
            text="A personal account claiming output fell after going remote.",
            domain="substack.com",
            published=_iso(2600),
            subtask_id="T2",
        ),
    ]


@pytest.fixture
def conflicting_sources() -> list[Source]:
    """Two equally credible sources that flatly disagree — the split-evidence case."""
    return [
        Source(
            id="S1",
            url="https://www.nature.com/articles/remote-up",
            title="Remote work raises productivity",
            snippet="Productivity rose 4% under remote work.",
            text="Study finds a 4% productivity gain.",
            domain="nature.com",
            published=_iso(100),
            credibility_score=0.8,
        ),
        Source(
            id="S2",
            url="https://www.science.org/doi/remote-down",
            title="Remote work lowers productivity",
            snippet="Productivity fell 4% under remote work.",
            text="Study finds a 4% productivity loss.",
            domain="science.org",
            published=_iso(100),
            credibility_score=0.8,
        ),
    ]


@pytest.fixture
def base_state(fixture_sources) -> RunState:
    """A run part-way through: researched, pooled, ready to debate."""
    state = new_run_state("testrun01", "Is remote work more productive than office work?")
    state["hypothesis"] = "Remote work increases measured productivity for knowledge workers."
    state["subtasks"] = [
        Subtask(id="T1", question="Does remote work raise output?", rationale="Core."),
        Subtask(id="T2", question="Where does remote work hurt?", rationale="Refutation."),
    ]
    state["sources"] = fixture_sources
    state["raw_sources"] = fixture_sources
    return state


@pytest.fixture
def bias_report() -> BiasReport:
    return BiasReport(
        outlet_concentration="Spread across three domains.",
        recency_skew="One source is stale.",
        funding_flags=[],
        missing_perspectives=["Employer-side telemetry"],
        weakly_sourced_claims=[],
        summary="Pool is acceptable.",
    )
