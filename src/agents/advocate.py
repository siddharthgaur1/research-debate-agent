"""Advocate: the strongest evidence-based case FOR the main finding.

Adversarial by design, but not dishonest. The Advocate argues one side as hard as
the evidence allows and no harder — an advocate who overstates gets caught by the
Critic, and the Arbitrator reads both. Every point must carry a `[S#]` citation;
an uncited assertion is exactly the hand-waving this system exists to eliminate.
"""

from __future__ import annotations

from ..state.schema import RunState, Stance
from ..tools.llm import Budget
from .common import cited_ids, format_sources, log, say, sources_by_id

NODE = "advocate"

_SYSTEM = """You are the Advocate in a structured research debate.

Build the STRONGEST case FOR the hypothesis that the evidence can actually carry.

Hard rules:
- Every substantive point MUST cite a source id in square brackets, e.g. [S3].
  A point you cannot cite is a point you must drop.
- Cite only ids that appear in the provided source list. Never invent one.
- Prefer high-credibility and independently corroborated sources; leaning on a
  weak source is a liability the Critic will exploit.
- Do not overstate. If the evidence supports a narrow version of the claim,
  argue the narrow version — it survives contact with the Critic.
- If rebutting the Critic, address their specific points; do not repeat yourself.

Structure: 3-5 numbered points, each one or two sentences with citations.
Under 350 words."""


def advocate_agent(state: RunState) -> RunState:
    """Argue for the hypothesis, optionally rebutting the Critic's last case."""
    llm = Budget(state, NODE)
    sources = state.get("sources", [])
    round_number = state.get("debate_round", 0)

    prompt = (
        f"Hypothesis to defend: {state['hypothesis']}\n\n"
        f"Original question: {state['question']}\n\n"
        f"Sources:\n{format_sources(sources)}"
    )
    critic_case = state.get("critic_case", "")
    if critic_case:
        prompt += (
            f"\n\nThe Critic has argued the following. Rebut their strongest points "
            f"where the evidence lets you, and concede where it does not:\n{critic_case}"
        )

    case = llm.text(_SYSTEM, prompt)
    cites = cited_ids(case, sources_by_id(state))

    turn = say(
        state,
        "Advocate",
        case,
        stance=Stance.FOR,
        round=round_number,
        cited=cites,
    )

    return {  # type: ignore[return-value]
        "advocate_case": case,
        "debate_transcript": [turn],
        "messages": [log(NODE, f"Argued for, citing {len(cites)} source(s).")],
        "token_usage": llm.usage,
    }
