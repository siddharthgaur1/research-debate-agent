"""Score how much a source deserves to be believed.

Deliberately a heuristic and not an LLM call: this runs on every source of every
run, it must be cheap, and — more to the point — a score you can't reproduce or
test is not a score, it's a vibe. Three inputs: what kind of publisher it is, how
old it is, and whether anyone independent says the same thing.

The reasoning string is part of the output, not a debug aid. The Bias Checker and
the report both quote it, so "0.35" always arrives with its "because".
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from dateutil import parser as date_parser

from .fetch import domain_of

# Institutional publishers: public accountability, corrections policy, or peer review.
_INSTITUTIONAL_SUFFIXES = (".gov", ".edu", ".mil", ".int", ".gov.uk", ".ac.uk", ".edu.au")

# Wire services, papers of record, and peer-reviewed venues.
_MAJOR_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "npr.org",
    "nytimes.com", "wsj.com", "washingtonpost.com", "ft.com", "economist.com",
    "theguardian.com", "bloomberg.com", "nature.com", "science.org",
    "nejm.org", "thelancet.com", "bmj.com", "jamanetwork.com", "pnas.org",
    "sciencedirect.com", "springer.com", "cell.com", "acm.org", "ieee.org",
    "oecd.org", "who.int", "worldbank.org", "imf.org",
}

# Preprints and encyclopedias: useful, but not yet vouched for by anyone.
_UNREVIEWED_DOMAINS = {"arxiv.org", "biorxiv.org", "medrxiv.org", "ssrn.com", "wikipedia.org"}

# Self-published or user-generated: anyone can say anything here.
_LOW_TRUST_DOMAINS = {
    "medium.com", "substack.com", "reddit.com", "quora.com", "blogspot.com",
    "wordpress.com", "tumblr.com", "x.com", "twitter.com", "facebook.com",
    "linkedin.com", "pinterest.com", "answers.com", "ezinearticles.com",
}

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "is",
    "are", "was", "were", "be", "been", "it", "its", "this", "that", "with",
    "as", "at", "by", "from", "has", "have", "had", "not", "more", "than",
    "study", "says", "new", "report", "research",
}

_WORD_RE = re.compile(r"[a-z0-9]+")

#: Jaccard overlap above which two sources are treated as making the same claim.
_CORROBORATION_OVERLAP = 0.12


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2}


def _domain_tier(domain: str) -> tuple[float, str]:
    """Base score from what kind of publisher this is."""
    if domain.endswith(_INSTITUTIONAL_SUFFIXES):
        return 0.90, f"{domain} is an institutional (.gov/.edu-class) publisher"
    if domain in _MAJOR_DOMAINS:
        return 0.78, f"{domain} is a major outlet or peer-reviewed venue"
    if domain in _UNREVIEWED_DOMAINS:
        return 0.55, f"{domain} is credible but unreviewed (preprint/encyclopedia)"
    if domain in _LOW_TRUST_DOMAINS or any(
        domain.endswith("." + d) for d in _LOW_TRUST_DOMAINS
    ):
        return 0.25, f"{domain} is self-published or user-generated"
    return 0.50, f"{domain} is an unrecognised publisher"


def parse_date(published: str | None) -> datetime | None:
    """Best-effort parse of whatever date string a search backend handed us.

    Fuzzy because backends return everything from `2023-05-01` to
    `Published May 1, 2023`. Returns None rather than raising: an unparseable
    date is a scoring input, not an error.
    """
    if not published:
        return None
    try:
        dt = date_parser.parse(published, fuzzy=True)
    except (ValueError, OverflowError, TypeError):
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _recency(published: str | None, now: datetime | None = None) -> tuple[float, str]:
    """Penalty for age. Unknown dates take a small hit — not knowing is itself a risk."""
    now = now or datetime.now(timezone.utc)
    dt = parse_date(published)
    if dt is None:
        return -0.03, "no publication date given"
    years = (now - dt).days / 365.25
    if years < 0:
        return -0.03, f"publication date {dt.date()} is in the future"
    if years <= 2:
        return 0.0, f"recent ({dt.date()})"
    if years <= 5:
        return -0.07, f"somewhat dated ({dt.date()}, ~{years:.0f}y old)"
    return -0.18, f"stale ({dt.date()}, ~{years:.0f}y old)"


def _corroboration(source, peers) -> tuple[float, str, list[str]]:
    """Bonus for independent agreement.

    Independent means a *different domain* — five pages from one outlet repeating
    each other is one source wearing five hats, which is exactly the failure this
    whole project exists to avoid.
    """
    mine = _tokens(f"{source.title} {source.snippet}")
    if not mine:
        return 0.0, "nothing to corroborate", []

    agreeing: list[str] = []
    seen_domains: set[str] = set()
    for peer in peers:
        if peer.id == source.id or peer.domain == source.domain:
            continue
        theirs = _tokens(f"{peer.title} {peer.snippet}")
        if not theirs:
            continue
        overlap = len(mine & theirs) / len(mine | theirs)
        if overlap >= _CORROBORATION_OVERLAP and peer.domain not in seen_domains:
            agreeing.append(peer.id)
            seen_domains.add(peer.domain)

    if not agreeing:
        return 0.0, "no independent corroboration found", []
    bonus = min(0.05 * len(agreeing), 0.15)
    return (
        bonus,
        f"corroborated by {len(agreeing)} independent domain(s)",
        agreeing,
    )


def score_source(source, peers=()) -> tuple[float, str, list[str]]:
    """Return (score, reasoning, corroborating_ids) for one source."""
    domain = source.domain or domain_of(source.url)
    base, why_domain = _domain_tier(domain)
    age_delta, why_age = _recency(source.published)
    corr_delta, why_corr, agreeing = _corroboration(source, peers)

    score = max(0.0, min(1.0, base + age_delta + corr_delta))
    reasoning = f"{why_domain}; {why_age}; {why_corr}."
    return round(score, 3), reasoning, agreeing


def score_sources(sources: list) -> list:
    """Score every source against the rest of the pool, in place, and return it."""
    for source in sources:
        if not source.domain:
            source.domain = domain_of(source.url)
    for source in sources:
        score, reasoning, agreeing = score_source(source, sources)
        source.credibility_score = score
        source.credibility_reasoning = reasoning
        source.corroborated_by = agreeing
    return sources
