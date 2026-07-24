"""Shared helpers for agent nodes."""

from __future__ import annotations

import re
from collections.abc import Iterable

from ..persistence import stream
from ..state.schema import AgentMessage, DebateTurn, RunState, Source, Stance

_CITE_RE = re.compile(r"\[(S\d+)\]")


def log(agent: str, content: str, level: str = "info") -> AgentMessage:
    """Build one agent-log entry."""
    return AgentMessage(agent=agent, content=content, level=level)  # type: ignore[arg-type]


def say(
    state: RunState,
    agent: str,
    content: str,
    *,
    stance: Stance = Stance.NEUTRAL,
    round: int = 0,
    cited: Iterable[str] = (),
) -> DebateTurn:
    """Build a debate turn and push it to the live stream immediately.

    Publishing here rather than after the node returns is the point: the UI shows
    the argument as it is made, not once the whole run finishes.
    """
    turn = DebateTurn(
        agent=agent,
        stance=stance,
        round=round,
        content=content,
        cited_source_ids=list(cited),
    )
    stream.publish_turn(state["run_id"], turn)
    return turn


def sources_by_id(state: RunState) -> dict[str, Source]:
    """Every source in the run, keyed by id."""
    return {s.id: s for s in state.get("sources", [])}


def cited_ids(text: str, known: Iterable[str]) -> list[str]:
    """Extract `[S3]`-style citations from model prose, keeping only real ids.

    Models cite sources that don't exist. Filtering against the known pool here
    means a hallucinated `[S99]` never reaches the report's citation trail.
    """
    known = set(known)
    seen: list[str] = []
    for match in _CITE_RE.findall(text):
        if match in known and match not in seen:
            seen.append(match)
    return seen


def format_sources(sources: Iterable[Source], *, include_text: bool = True) -> str:
    """Render the source pool for a prompt, with ids the model must cite by.

    Credibility and its reasoning are included so a debater can weigh a source
    rather than treat every url as equal.
    """
    blocks: list[str] = []
    for s in sources:
        header = (
            f"[{s.id}] {s.title}\n"
            f"    url: {s.url}  ({s.domain})\n"
            f"    published: {s.published or 'unknown'}\n"
            f"    credibility: {s.credibility_score:.2f} — {s.credibility_reasoning}"
        )
        if s.merged_from:
            header += f"\n    also republished at: {', '.join(s.merged_from)}"
        body = (s.text or s.snippet).strip()
        if include_text and body:
            header += f"\n    excerpt: {body[:1500]}"
        elif s.snippet:
            header += f"\n    snippet: {s.snippet[:300]}"
        blocks.append(header)
    return "\n\n".join(blocks) if blocks else "(no sources)"
