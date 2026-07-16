"""Supervisor: turn a question into a hypothesis and research angles.

The hypothesis matters more than the subtasks. "Is remote work more productive?"
is not something you can argue for or against — "Remote work increases measured
productivity for knowledge workers" is. Without a falsifiable main finding the
Advocate and Critic end up talking past each other, so this node's real job is to
give the debate a target.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..state.schema import RunState, RunStatus, Stance, Subtask
from ..tools.llm import Budget
from .common import log, say

NODE = "supervisor"

_SYSTEM = """You are the Supervisor of a research debate team.

Given a user's question you must:
1. State the MAIN FINDING as a single falsifiable hypothesis — a claim that
   evidence could support or refute. Never restate the question as a question.
   Pick the most plausible direction; the Critic exists to attack it.
2. Decompose the question into 3-5 research subtasks that together cover the
   evidence needed to test that hypothesis. Include angles that could REFUTE it,
   not only ones that confirm it — a one-sided search produces a fake debate.

Each subtask must be a self-contained web-search-ready question."""


class _SubtaskSpec(BaseModel):
    question: str = Field(description="A self-contained, searchable research question.")
    rationale: str = Field(description="Why this angle matters to the main finding.")


class SupervisorPlan(BaseModel):
    """Structured output from the Supervisor's decomposition."""

    hypothesis: str = Field(description="The falsifiable main finding to be debated.")
    subtasks: list[_SubtaskSpec] = Field(description="3-5 research subtasks.")


def supervisor_agent(state: RunState) -> RunState:
    """Set the hypothesis and fan-out plan for the run."""
    llm = Budget(state, NODE)
    plan = llm.structured(
        _SYSTEM,
        f"Question: {state['question']}",
        SupervisorPlan,
    )

    subtasks = [
        Subtask(id=f"T{i}", question=s.question, rationale=s.rationale)
        for i, s in enumerate(plan.subtasks[:5], start=1)
    ]

    angles = "\n".join(f"  {t.id}. {t.question} — {t.rationale}" for t in subtasks)
    turn = say(
        state,
        NODE,
        f"Main finding to test:\n  {plan.hypothesis}\n\nResearch angles:\n{angles}",
        stance=Stance.NEUTRAL,
    )

    return {  # type: ignore[return-value]
        "hypothesis": plan.hypothesis,
        "subtasks": subtasks,
        "status": RunStatus.RESEARCHING,
        "debate_transcript": [turn],
        "messages": [log(NODE, f"Planned {len(subtasks)} research subtasks.")],
        "token_usage": llm.usage,
    }
