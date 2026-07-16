"""The single state object that flows through the whole debate graph.

Researchers run in parallel, so every key they write needs a reducer — without one
LangGraph rejects concurrent updates to the same key. Anything appended (sources,
transcript turns, spend) uses `operator.add`; anything written once by a single
node is a plain field.
"""

from __future__ import annotations

import operator
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStatus(str, Enum):
    """Terminal and in-flight states for a run."""

    PENDING = "pending"
    RESEARCHING = "researching"
    DEBATING = "debating"
    AUDITING = "auditing"
    ARBITRATING = "arbitrating"
    REPORTING = "reporting"
    COMPLETED = "completed"
    FAILED = "failed"


class Stance(str, Enum):
    """Which side of the debate a turn argues."""

    NEUTRAL = "neutral"
    FOR = "for"
    AGAINST = "against"


class Subtask(BaseModel):
    """One research angle the Supervisor carved out of the question."""

    id: str
    question: str
    rationale: str = Field(description="Why this angle matters to the main finding.")


class SourceStub(BaseModel):
    """What a search backend returns before we fetch the page.

    Both the Tavily and SerpAPI adapters normalise to this, so the rest of the
    system never learns which backend it is talking to.
    """

    url: str
    title: str
    snippet: str = ""
    published: str | None = None


class Source(BaseModel):
    """A fetched, scored, deduplicated piece of evidence."""

    id: str
    url: str
    title: str
    snippet: str = ""
    text: str = Field(default="", description="Cleaned page text, truncated to budget.")
    domain: str = ""
    published: str | None = None
    subtask_id: str | None = None

    credibility_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="0=junk, 1=authoritative."
    )
    credibility_reasoning: str = Field(
        default="", description="Why the score is what it is. Stored, not just the number."
    )
    corroborated_by: list[str] = Field(
        default_factory=list, description="Ids of independent sources agreeing."
    )

    embedding_id: str | None = None
    merged_from: list[str] = Field(
        default_factory=list, description="Urls of near-duplicates folded into this one."
    )

    @property
    def is_independent_of(self) -> str:
        """Domain used to decide whether two sources are really independent."""
        return self.domain


class DebateTurn(BaseModel):
    """One turn in the argument. The UI replays these in sequence.

    There is deliberately no `index` field: researchers run as parallel branches
    and would each compute the same "next index". Position in the transcript list
    (and in the Redis replay list) is the order — one source of truth, no ties.
    """

    agent: str
    stance: Stance = Stance.NEUTRAL
    round: int = 0
    content: str
    cited_source_ids: list[str] = Field(default_factory=list)
    at: str = Field(default_factory=_now)


class Claim(BaseModel):
    """A single assertion in the final report, with its evidence and confidence."""

    id: str
    text: str
    supporting_source_ids: list[str] = Field(default_factory=list)
    opposing_source_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    contested: bool = False
    rationale: str = Field(default="", description="How the confidence was reached.")


class BiasReport(BaseModel):
    """The Bias Checker's audit of the source pool itself, not of the argument."""

    outlet_concentration: str = ""
    recency_skew: str = ""
    funding_flags: list[str] = Field(default_factory=list)
    missing_perspectives: list[str] = Field(default_factory=list)
    weakly_sourced_claims: list[str] = Field(
        default_factory=list, description="Claim texts resting on thin/one-sided sourcing."
    )
    summary: str = ""


class Verdict(BaseModel):
    """The Arbitrator's synthesis. May legitimately decline to pick a side."""

    statement: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    uncertainty_mode: bool = Field(
        default=False,
        description="True when evidence is genuinely split and no verdict is forced.",
    )
    reasoning: str = ""
    strongest_for: list[str] = Field(default_factory=list)
    strongest_against: list[str] = Field(default_factory=list)
    contested_points: list[str] = Field(default_factory=list)


class TokenUsage(BaseModel):
    """Per-node LLM spend, accumulated to enforce the per-run cost cap."""

    node: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    at: str = Field(default_factory=_now)


class AgentMessage(BaseModel):
    """One entry in the agent log. Appended by every node, never overwritten."""

    agent: str
    content: str
    level: Literal["info", "warn", "error"] = "info"
    at: str = Field(default_factory=_now)


class RunState(TypedDict, total=False):
    """State passed between every node in the graph."""

    run_id: str
    question: str
    status: RunStatus
    error: str | None

    # Supervisor
    hypothesis: str
    subtasks: list[Subtask]

    # Researchers (parallel — reducers required).
    #
    # Two lists on purpose. `raw_sources` is append-only because parallel branches
    # write it concurrently. `sources` is the deduped, renumbered pool the debate
    # actually cites, written once by the gather node — which it could not do if
    # it shared the `operator.add` reducer, since returning the deduped list would
    # append it to the raw one rather than replace it.
    raw_sources: Annotated[list[Source], operator.add]
    search_count: Annotated[int, operator.add]
    sources: list[Source]

    # Debate
    advocate_case: str
    critic_case: str
    debate_round: int
    bias_report: BiasReport | None

    # Synthesis
    claims: list[Claim]
    verdict: Verdict | None
    report_path: str | None

    # Cross-cutting
    debate_transcript: Annotated[list[DebateTurn], operator.add]
    messages: Annotated[list[AgentMessage], operator.add]
    token_usage: Annotated[list[TokenUsage], operator.add]

    created_at: str


def new_run_state(run_id: str, question: str) -> RunState:
    """Build the initial state for a fresh run."""
    return RunState(
        run_id=run_id,
        question=question,
        status=RunStatus.PENDING,
        error=None,
        hypothesis="",
        subtasks=[],
        raw_sources=[],
        sources=[],
        search_count=0,
        advocate_case="",
        critic_case="",
        debate_round=0,
        bias_report=None,
        claims=[],
        verdict=None,
        report_path=None,
        debate_transcript=[],
        messages=[],
        token_usage=[],
        created_at=_now(),
    )
