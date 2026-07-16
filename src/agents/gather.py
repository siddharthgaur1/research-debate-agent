"""Gather: fold the parallel researchers' findings into one citable pool.

This is the join point of the fan-out. It does three things the researchers
cannot do individually, because none of them can see the others' results:

1. Dedup — the same wire story found by three researchers is one piece of
   evidence, not three.
2. Renumber — survivors get dense canonical ids (`S1..Sn`), which are the ids
   every downstream agent cites and the report links back to.
3. Re-score — corroboration is a property of the whole pool, so it can only be
   computed once every source is on the table.
"""

from __future__ import annotations

from ..state.schema import RunState, RunStatus, Stance
from ..tools.credibility import score_sources
from ..tools.dedup import dedup_sources
from .common import log, say

NODE = "gather"


def gather_agent(state: RunState) -> RunState:
    """Dedup, renumber and re-score the pooled sources."""
    raw = state.get("raw_sources", [])
    kept = dedup_sources(state["run_id"], raw)

    for number, source in enumerate(kept, start=1):
        source.id = f"S{number}"

    # Corroboration depends on the final pool, so scoring only settles here.
    score_sources(kept)

    merged = sum(len(s.merged_from) for s in kept)
    usable = [s for s in kept if s.text or s.snippet]

    turn = say(
        state,
        NODE,
        f"Pooled {len(raw)} results into **{len(kept)} distinct sources** "
        f"({merged} near-duplicate(s) merged).\n\n"
        + "\n".join(
            f"- [{s.id}] {s.title} — {s.domain} (credibility {s.credibility_score:.2f})"
            for s in kept
        ),
        stance=Stance.NEUTRAL,
    )

    return {  # type: ignore[return-value]
        "sources": kept,
        "status": RunStatus.DEBATING,
        "debate_transcript": [turn],
        "messages": [
            log(
                NODE,
                f"{len(raw)} raw -> {len(kept)} deduped ({merged} merged), "
                f"{len(usable)} with usable text.",
            )
        ],
    }
