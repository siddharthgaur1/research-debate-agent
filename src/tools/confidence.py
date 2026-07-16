"""Turn evidence into a number, reproducibly.

The Arbitrator's model decides *which* sources back a claim and whether the Critic
landed a real hit on it. It does not pick the confidence number. This module does,
from those inputs, in code.

That split is deliberate. A model asked for "confidence: 0.0-1.0" produces a
fluent guess that is stable across reruns only by luck, and that nobody can audit
or unit-test. Splitting it means the judgement calls stay with the model and the
arithmetic stays where it can be tested — every number below is derived from
inputs a reader can check.

Three factors, matching how a careful reader actually weighs evidence:
  balance      — credibility-weighted support vs. opposition
  independence — how many *distinct domains* back it (one outlet is one voice)
  damage       — penalties when the Critic connected, or the pool is thin
"""

from __future__ import annotations

from typing import Iterable, Sequence

from ..config import get_settings
from ..state.schema import BiasReport, Claim, Source

#: Confidence is clamped to this range. Nothing derived from a handful of web
#: sources is ever 0 or 1, and claiming otherwise is the failure mode this whole
#: project is built to avoid.
_FLOOR, _CEILING = 0.02, 0.98

#: Multiplier by number of independent supporting domains. One domain saying
#: something is meaningfully weaker than three, and the curve saturates fast.
_INDEPENDENCE = {0: 0.30, 1: 0.70, 2: 0.85}
_INDEPENDENCE_MAX = 1.0

#: Applied when the Critic landed a substantive, evidenced hit on the claim.
_CRITIC_HIT_PENALTY = 0.75

#: Applied when the Bias Checker flagged this claim's sourcing as thin/one-sided.
_WEAK_SOURCING_PENALTY = 0.80


def _weight(ids: Iterable[str], pool: dict[str, Source]) -> float:
    """Credibility-weighted mass of a set of source ids."""
    return sum(pool[i].credibility_score for i in ids if i in pool)


def _independent_domains(ids: Iterable[str], pool: dict[str, Source]) -> int:
    """How many distinct publishers back this. Five pages from one site count once."""
    return len({pool[i].domain for i in ids if i in pool and pool[i].domain})


def _independence_factor(domains: int) -> float:
    return _INDEPENDENCE.get(domains, _INDEPENDENCE_MAX)


def _flagged_weak(text: str, bias_report: BiasReport | None) -> bool:
    """Did the Bias Checker call out this claim's sourcing?"""
    if not bias_report:
        return False
    needle = text.lower().strip()
    return any(
        needle in flag.lower() or flag.lower() in needle
        for flag in bias_report.weakly_sourced_claims
        if flag.strip()
    )


def score_claim(
    text: str,
    supporting_source_ids: Sequence[str],
    opposing_source_ids: Sequence[str],
    sources: Sequence[Source],
    *,
    critic_landed_hit: bool = False,
    bias_report: BiasReport | None = None,
) -> tuple[float, bool, str]:
    """Return (confidence, contested, rationale) for one claim."""
    settings = get_settings()
    pool = {s.id: s for s in sources}

    support = _weight(supporting_source_ids, pool)
    oppose = _weight(opposing_source_ids, pool)
    domains = _independent_domains(supporting_source_ids, pool)

    if support + oppose == 0:
        balance = 0.5
        balance_note = "no citable evidence either way"
    else:
        balance = support / (support + oppose)
        balance_note = (
            f"credibility-weighted support {support:.2f} vs opposition {oppose:.2f}"
        )

    factor = _independence_factor(domains)
    confidence = balance * factor
    notes = [balance_note, f"{domains} independent domain(s) supporting"]

    if critic_landed_hit:
        confidence *= _CRITIC_HIT_PENALTY
        notes.append("Critic landed a substantive hit")

    if _flagged_weak(text, bias_report):
        confidence *= _WEAK_SOURCING_PENALTY
        notes.append("Bias Checker flagged the sourcing as weak")

    confidence = max(_FLOOR, min(_CEILING, confidence))

    # Contested means the evidence genuinely splits — not that someone objected.
    # A claim only counts as contested if real opposing evidence exists.
    near_even = abs(balance - 0.5) <= settings.contested_margin
    contested = oppose > 0 and (near_even or critic_landed_hit)
    if contested:
        notes.append("contested: real evidence on both sides")

    return round(confidence, 3), contested, "; ".join(notes) + "."


def score_claims(
    drafts: Sequence[tuple[str, Sequence[str], Sequence[str], bool]],
    sources: Sequence[Source],
    bias_report: BiasReport | None = None,
) -> list[Claim]:
    """Score a batch of (text, supporting, opposing, critic_hit) drafts."""
    claims: list[Claim] = []
    for i, (text, supporting, opposing, critic_hit) in enumerate(drafts, start=1):
        known = {s.id for s in sources}
        # Drop invented citations before they reach the report's evidence trail.
        supporting = [s for s in supporting if s in known]
        opposing = [s for s in opposing if s in known]

        confidence, contested, rationale = score_claim(
            text,
            supporting,
            opposing,
            sources,
            critic_landed_hit=critic_hit,
            bias_report=bias_report,
        )
        claims.append(
            Claim(
                id=f"C{i}",
                text=text,
                supporting_source_ids=list(supporting),
                opposing_source_ids=list(opposing),
                confidence=confidence,
                contested=contested,
                rationale=rationale,
            )
        )
    return claims


def is_uncertain(claims: Sequence[Claim]) -> bool:
    """Whether the run should enter uncertainty mode.

    True when enough of the load-bearing claims are genuinely split that naming a
    winner would misrepresent the evidence. An empty claim set is uncertain by
    definition — we found nothing to be confident about.
    """
    if not claims:
        return True
    contested_fraction = sum(1 for c in claims if c.contested) / len(claims)
    return contested_fraction >= get_settings().uncertainty_ratio


def verdict_confidence(claims: Sequence[Claim]) -> float:
    """Overall confidence: the mean of the claims the verdict rests on."""
    if not claims:
        return _FLOOR
    return round(sum(c.confidence for c in claims) / len(claims), 3)
