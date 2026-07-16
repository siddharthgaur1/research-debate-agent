"""Bias Checker: audit the evidence pool, not the argument.

The Advocate and Critic both argue from the same sources. If that pool is skewed —
one outlet, one year, one funding interest — then both cases inherit the skew and
the debate looks rigorous while being confidently wrong. Nobody arguing inside the
debate can see this; it takes a node that looks at the sourcing itself.

Pool statistics are computed in code and handed to the model as fact. Asking an
LLM to eyeball "how many sources are from one domain" produces confident bad
arithmetic; asking it to judge what a 60% concentration *means* is what it is
actually good at.
"""

from __future__ import annotations

from collections import Counter

from ..state.schema import BiasReport, RunState, RunStatus, Stance
from ..tools.credibility import parse_date
from ..tools.llm import Budget
from .common import format_sources, log, say

NODE = "bias"

_SYSTEM = """You are the Bias Checker auditing a research debate's SOURCE POOL.

You are not judging who won. You are judging whether the evidence base is sound
enough for anyone to be judged on it.

Assess:
- Outlet concentration: is the pool leaning on one publisher or one syndicated story?
- Recency skew: is it stale, or clustered in one moment that may not generalise?
- Funding/agenda red flags: sources with an obvious interest in the conclusion
  (industry bodies, vendors, advocacy groups) — name them by id.
- Missing perspectives: whose evidence would change this and is absent?
- Weakly sourced claims: which specific claims in the debate rest on thin,
  one-sided, or low-credibility sourcing? Quote the claim text.

Be concrete and cite source ids. If the pool is actually fine, say so plainly —
inventing bias is as damaging as missing it."""


def _pool_stats(sources) -> str:
    """Deterministic facts about the pool, for the model to interpret."""
    if not sources:
        return "The pool is empty."

    total = len(sources)
    domains = Counter(s.domain for s in sources)
    top_domain, top_count = domains.most_common(1)[0]

    years = [d.year for d in (parse_date(s.published) for s in sources) if d]
    undated = total - len(years)
    year_span = f"{min(years)}–{max(years)}" if years else "none dated"

    mean_cred = sum(s.credibility_score for s in sources) / total
    low_cred = [s.id for s in sources if s.credibility_score < 0.4]
    republished = [s.id for s in sources if s.merged_from]

    return (
        f"- {total} distinct sources across {len(domains)} domains.\n"
        f"- Most common domain: {top_domain} with {top_count} "
        f"({top_count / total:.0%} of the pool).\n"
        f"- Domain breakdown: {dict(domains)}\n"
        f"- Publication years: {year_span}; {undated} source(s) undated.\n"
        f"- Mean credibility {mean_cred:.2f}; low-credibility (<0.4): "
        f"{low_cred or 'none'}.\n"
        f"- Sources that were republished elsewhere (near-duplicates merged in): "
        f"{republished or 'none'}."
    )


def bias_agent(state: RunState) -> RunState:
    """Audit the source pool and flag what the debate should not be trusted on."""
    llm = Budget(state, NODE)
    sources = state.get("sources", [])

    report = llm.structured(
        _SYSTEM,
        f"Question: {state['question']}\n"
        f"Hypothesis: {state['hypothesis']}\n\n"
        f"Pool statistics (computed, treat as fact):\n{_pool_stats(sources)}\n\n"
        f"Sources:\n{format_sources(sources, include_text=False)}\n\n"
        f"Advocate's case:\n{state.get('advocate_case', '')}\n\n"
        f"Critic's case:\n{state.get('critic_case', '')}",
        BiasReport,
    )

    flagged = len(report.weakly_sourced_claims)
    turn = say(
        state,
        "Bias Checker",
        f"{report.summary}\n\n"
        f"- **Outlet concentration:** {report.outlet_concentration}\n"
        f"- **Recency skew:** {report.recency_skew}\n"
        f"- **Funding/agenda flags:** {', '.join(report.funding_flags) or 'none'}\n"
        f"- **Missing perspectives:** {', '.join(report.missing_perspectives) or 'none'}\n"
        f"- **Weakly sourced claims:** {flagged}",
        stance=Stance.NEUTRAL,
    )

    return {  # type: ignore[return-value]
        "bias_report": report,
        "status": RunStatus.ARBITRATING,
        "debate_transcript": [turn],
        "messages": [log(NODE, f"Audited pool; flagged {flagged} weak claim(s).")],
        "token_usage": llm.usage,
    }
