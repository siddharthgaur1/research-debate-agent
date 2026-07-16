"""Arbitrator: synthesise the debate into claims, confidences, and a verdict.

The temptation this node exists to resist is false consensus. An LLM handed two
opposing arguments will almost always produce a tidy "on balance, yes" — because
that reads like an answer, and answers feel like competence. When the evidence is
genuinely split, that tidiness is a lie.

So the model's job here is narrow: identify the claims, say which sources support
and oppose each, and report whether the Critic actually landed. The confidence
arithmetic and the uncertainty-mode decision happen in `tools.confidence`, in
code, where they are reproducible and testable — and where "the evidence is split"
is a computed outcome the model cannot smooth over.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..state.schema import RunState, RunStatus, Stance, Verdict
from ..tools.confidence import is_uncertain, score_claims, verdict_confidence
from ..tools.llm import Budget
from .common import format_sources, log, say

NODE = "arbitrator"

_SYSTEM = """You are the Arbitrator of a research debate. You have read the
Advocate's case, the Critic's case, and the Bias Checker's audit.

Extract the 3-6 KEY CLAIMS the question turns on. For each claim:
- supporting_source_ids / opposing_source_ids: only ids from the source list.
  Never invent one. Leave a list empty if nothing supports or opposes it.
- critic_landed_hit: true ONLY if the Critic raised a substantive, evidence-backed
  problem with this specific claim. Rhetorical disagreement is not a hit.

Then write the verdict statement and your reasoning.

CRITICAL — do not manufacture consensus. If the evidence genuinely splits on a
claim, cite both sides and let it be split. A verdict that reads cleanly but
misrepresents divided evidence is the worst output you can produce. You are NOT
scoring confidence — that is computed from your source assignments, so assign
them honestly rather than to reach a conclusion you prefer.

If the evidence does not settle the question, say exactly that in the verdict."""


class _ClaimDraft(BaseModel):
    text: str = Field(description="The claim, as one self-contained sentence.")
    supporting_source_ids: list[str] = Field(default_factory=list)
    opposing_source_ids: list[str] = Field(default_factory=list)
    critic_landed_hit: bool = Field(
        default=False, description="True only for a substantive, evidenced hit."
    )


class ArbitratorOutput(BaseModel):
    """What the Arbitrator's model returns — judgements, not numbers."""

    verdict_statement: str = Field(description="The synthesis, in 1-3 sentences.")
    reasoning: str = Field(description="How the debate led here.")
    claims: list[_ClaimDraft] = Field(description="The 3-6 claims the question turns on.")
    strongest_for: list[str] = Field(
        default_factory=list, description="The Advocate's points that survived."
    )
    strongest_against: list[str] = Field(
        default_factory=list, description="The Critic's points that survived."
    )


def arbitrator_agent(state: RunState) -> RunState:
    """Synthesise the debate, then compute confidence from the evidence."""
    llm = Budget(state, NODE)
    sources = state.get("sources", [])
    bias_report = state.get("bias_report")

    out = llm.structured(
        _SYSTEM,
        f"Question: {state['question']}\n"
        f"Hypothesis under test: {state['hypothesis']}\n\n"
        f"Sources:\n{format_sources(sources, include_text=False)}\n\n"
        f"ADVOCATE'S CASE:\n{state.get('advocate_case', '')}\n\n"
        f"CRITIC'S CASE:\n{state.get('critic_case', '')}\n\n"
        f"BIAS AUDIT:\n{bias_report.model_dump_json(indent=2) if bias_report else 'none'}",
        ArbitratorOutput,
    )

    claims = score_claims(
        [
            (c.text, c.supporting_source_ids, c.opposing_source_ids, c.critic_landed_hit)
            for c in out.claims
        ],
        sources,
        bias_report,
    )

    uncertain = is_uncertain(claims)
    contested = [c.text for c in claims if c.contested]

    verdict = Verdict(
        statement=out.verdict_statement,
        confidence=verdict_confidence(claims),
        uncertainty_mode=uncertain,
        reasoning=out.reasoning,
        strongest_for=out.strongest_for,
        strongest_against=out.strongest_against,
        contested_points=contested,
    )

    headline = (
        "**UNCERTAINTY MODE** — the evidence is genuinely split. Presenting both "
        "sides rather than forcing a verdict.\n\n"
        if uncertain
        else ""
    )
    turn = say(
        state,
        "Arbitrator",
        f"{headline}**Verdict:** {verdict.statement}\n\n"
        f"{verdict.reasoning}\n\n"
        f"**Claims ({len(claims)}, {len(contested)} contested):**\n"
        + "\n".join(
            f"- {'⚖️ ' if c.contested else ''}{c.text} "
            f"(confidence {c.confidence:.2f})"
            for c in claims
        ),
        stance=Stance.NEUTRAL,
    )

    return {  # type: ignore[return-value]
        "claims": claims,
        "verdict": verdict,
        "status": RunStatus.REPORTING,
        "debate_transcript": [turn],
        "messages": [
            log(
                NODE,
                f"{len(claims)} claims, {len(contested)} contested, "
                f"uncertainty_mode={uncertain}.",
            )
        ],
        "token_usage": llm.usage,
    }
