"""Report: assemble the final artifact and enforce the citation guarantee.

No LLM call here on purpose. The Arbitrator already produced every judgement this
report contains; asking a model to "write it up" would only add a chance to
paraphrase a confidence or drop a caveat. Assembly is deterministic.

This node is also where the promise that every claim links back to a real fetched
source stops being an aspiration and becomes an invariant: a claim citing nothing
that survived dedup is dropped, loudly, rather than printed with an empty
evidence trail.
"""

from __future__ import annotations

from ..config import get_settings
from ..persistence import stream
from ..state.schema import Claim, RunState, RunStatus, Stance
from ..tools.confidence import is_uncertain, verdict_confidence
from ..tools.pdf import render_report
from .common import log, say

NODE = "report"


def _grounded(claim: Claim, known: set[str]) -> bool:
    """A claim is reportable only if it cites at least one real surviving source."""
    cited = set(claim.supporting_source_ids) | set(claim.opposing_source_ids)
    return bool(cited & known)


def report_agent(state: RunState) -> RunState:
    """Drop ungrounded claims, re-derive the verdict, and export the PDF."""
    sources = state.get("sources", [])
    known = {s.id for s in sources}
    claims = state.get("claims", [])

    grounded = [c for c in claims if _grounded(c, known)]
    dropped = [c.text for c in claims if not _grounded(c, known)]

    verdict = state.get("verdict")
    if verdict and dropped:
        # Dropping claims changes the evidence base, so the headline numbers have
        # to be recomputed — otherwise the verdict quotes confidence from claims
        # that are no longer in the report.
        verdict = verdict.model_copy(
            update={
                "confidence": verdict_confidence(grounded),
                "uncertainty_mode": is_uncertain(grounded),
                "contested_points": [c.text for c in grounded if c.contested],
            }
        )

    final_state: RunState = {**state, "claims": grounded, "verdict": verdict}  # type: ignore[typeddict-item]

    settings = get_settings()
    path = settings.run_dir(state["run_id"]) / "report.pdf"
    render_report(final_state, path)

    contested = sum(1 for c in grounded if c.contested)
    note = f" Dropped {len(dropped)} uncited claim(s)." if dropped else ""
    turn = say(
        state,
        "Report",
        f"Report ready: **{len(grounded)} claims** ({contested} contested) across "
        f"**{len(sources)} sources**."
        + (
            "\n\nEvidence is split — the report presents both sides rather than a "
            "single verdict."
            if verdict and verdict.uncertainty_mode
            else ""
        )
        + note,
        stance=Stance.NEUTRAL,
    )

    stream.publish_done(state["run_id"])

    return {  # type: ignore[return-value]
        "claims": grounded,
        "verdict": verdict,
        "report_path": str(path),
        "status": RunStatus.COMPLETED,
        "debate_transcript": [turn],
        "messages": [
            log(NODE, f"Exported report to {path}.{note}", "warn" if dropped else "info")
        ],
    }
