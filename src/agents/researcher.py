"""Researcher: one instance per subtask, all running concurrently.

Each researcher owns exactly one angle. It searches once, fetches the hits,
scores them, and reports what it found. It does not argue — forming a view is the
Advocate's and Critic's job, and a researcher that editorialises here would bias
the pool before the debate even starts.

Source ids assigned here are provisional (`S1-2`). The gather node renumbers the
survivors to `S1..Sn` after dedup, so the ids the debate cites are dense and real.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict

from ..config import get_settings
from ..state.schema import RunState, Source, Stance, Subtask, TokenUsage
from ..tools import search as search_tool
from ..tools.credibility import score_sources
from ..tools.fetch import domain_of, fetch_text
from ..tools.llm import Budget
from .common import format_sources, log, say

NODE = "researcher"

_SYSTEM = """You are a Researcher on a debate team, assigned ONE narrow question.

Summarise what the sources actually say about your assigned question. Rules:
- Report what is there, including evidence that cuts against the team's hypothesis.
- Note where sources disagree with each other. Disagreement is signal, not noise.
- Say plainly when the evidence is thin or absent. Do not fill gaps with plausible
  reasoning — an invented finding poisons every downstream agent.
- Do not argue for a conclusion. Another agent does that.
Keep it under 200 words."""


class ResearchTask(TypedDict):
    """The payload a fan-out `Send` hands to one researcher branch.

    Deliberately not the full RunState: a branch needs its own subtask and enough
    context to check the spend cap, nothing more.
    """

    run_id: str
    question: str
    hypothesis: str
    subtask: Subtask
    task_number: int
    token_usage: list[TokenUsage]


def _gather_sources(subtask: Subtask, task_number: int) -> tuple[list[Source], int]:
    """Search once, fetch the hits in parallel, and score the result pool."""
    settings = get_settings()
    stubs = search_tool.search(subtask.question, max_results=settings.results_per_subtask)
    searches_used = 1

    sources = [
        Source(
            id=f"S{task_number}-{i}",
            url=stub.url,
            title=stub.title,
            snippet=stub.snippet,
            published=stub.published,
            domain=domain_of(stub.url),
            subtask_id=subtask.id,
        )
        for i, stub in enumerate(stubs, start=1)
    ]
    if not sources:
        return [], searches_used

    # Fetches are pure network waiting; doing them serially is the single biggest
    # avoidable delay in a run.
    with ThreadPoolExecutor(max_workers=min(8, len(sources))) as pool:
        texts = pool.map(fetch_text, [s.url for s in sources])
    for source, text in zip(sources, texts):
        source.text = text

    return score_sources(sources), searches_used


def researcher_agent(task: ResearchTask) -> RunState:
    """Research one subtask and contribute its sources to the shared pool."""
    subtask = task["subtask"]
    sources, searches_used = _gather_sources(subtask, task["task_number"])

    state_view: RunState = {  # type: ignore[assignment]
        "run_id": task["run_id"],
        "token_usage": task.get("token_usage", []),
    }
    llm = Budget(state_view, f"{NODE}:{subtask.id}")

    if sources:
        summary = llm.text(
            _SYSTEM,
            f"Assigned question: {subtask.question}\n\n"
            f"Team hypothesis under test: {task['hypothesis']}\n\n"
            f"Sources:\n{format_sources(sources)}",
            cheap=True,
        )
    else:
        summary = "No usable sources were found for this angle."

    turn = say(
        state_view,
        f"Researcher · {subtask.id}",
        f"**{subtask.question}**\n\n{summary}\n\n"
        f"_{len(sources)} source(s) gathered._",
        stance=Stance.NEUTRAL,
    )

    return {  # type: ignore[return-value]
        "raw_sources": sources,
        "search_count": searches_used,
        "debate_transcript": [turn],
        "messages": [
            log(f"{NODE}:{subtask.id}", f"Gathered {len(sources)} sources.")
        ],
        "token_usage": llm.usage,
    }
