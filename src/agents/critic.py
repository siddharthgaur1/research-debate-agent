"""Critic: the strongest case AGAINST the main finding.

The Critic has two jobs that are easy to confuse. The first is counterevidence —
what the sources say that contradicts the hypothesis. The second is auditing the
Advocate: overreach, cherry-picking, a conclusion resting on one weak source.

A Critic that only does the first is just a second Advocate pointing the other
way. The instruction to concede where the Advocate is right is what stops this
from degenerating into contrarianism — the Arbitrator needs an honest signal about
where the case is genuinely weak, not maximal disagreement.
"""

from __future__ import annotations

from ..state.schema import RunState, Stance
from ..tools.llm import Budget
from .common import cited_ids, format_sources, log, say, sources_by_id

NODE = "critic"

_SYSTEM = """You are the Critic in a structured research debate.

Build the STRONGEST case AGAINST the hypothesis, and audit the Advocate's case.

Cover, where the evidence supports it:
- Direct counterevidence in the sources.
- Methodological weaknesses in what the Advocate leaned on (small samples,
  self-reported data, self-interested publishers, correlation read as causation).
- Where the Advocate OVERREACHED — claimed more than their citation supports.
- Evidence gaps: what would be needed to settle this, and is missing.

Hard rules:
- Every substantive point MUST cite a source id in square brackets, e.g. [S3].
  Cite only ids from the provided list. Never invent one.
- CONCEDE explicitly where the Advocate's point is well-supported. Manufacturing
  disagreement is a failure: the Arbitrator relies on you to distinguish a real
  weakness from a rhetorical one.
- Attacking a source's credibility requires a reason grounded in the source list.

Structure: 3-5 numbered points, each one or two sentences with citations, then a
one-line "Concessions:" noting what genuinely holds up. Under 350 words."""


def critic_agent(state: RunState) -> RunState:
    """Argue against the hypothesis and attack the Advocate's case."""
    llm = Budget(state, NODE)
    sources = state.get("sources", [])
    round_number = state.get("debate_round", 0)

    prompt = (
        f"Hypothesis under attack: {state['hypothesis']}\n\n"
        f"Original question: {state['question']}\n\n"
        f"Sources:\n{format_sources(sources)}\n\n"
        f"The Advocate has argued:\n{state.get('advocate_case', '(nothing yet)')}"
    )

    case = llm.text(_SYSTEM, prompt)
    cites = cited_ids(case, sources_by_id(state))

    turn = say(
        state,
        "Critic",
        case,
        stance=Stance.AGAINST,
        round=round_number,
        cited=cites,
    )

    return {  # type: ignore[return-value]
        "critic_case": case,
        "debate_round": round_number + 1,
        "debate_transcript": [turn],
        "messages": [log(NODE, f"Argued against, citing {len(cites)} source(s).")],
        "token_usage": llm.usage,
    }
