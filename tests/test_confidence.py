"""The confidence engine and uncertainty mode.

These are the numbers the whole report hangs on, and they are pure functions —
so they get tested directly rather than through a mocked LLM.
"""

from __future__ import annotations

from src.state.schema import BiasReport, Claim, Source
from src.tools.confidence import (
    is_uncertain,
    score_claim,
    score_claims,
    verdict_confidence,
)


def _claim(cid: str, confidence: float, contested: bool = False) -> Claim:
    return Claim(id=cid, text=f"claim {cid}", confidence=confidence, contested=contested)


def test_more_credible_support_raises_confidence(fixture_sources):
    strong, _, _ = score_claim("c", ["S1", "S2"], [], fixture_sources)
    weak, _, _ = score_claim("c", ["S3"], [], fixture_sources)
    assert strong > weak


def test_independent_domains_beat_a_single_domain():
    same = [
        Source(id="S1", url="https://bls.gov/a", title="A", domain="bls.gov", credibility_score=0.8),
        Source(id="S2", url="https://bls.gov/b", title="B", domain="bls.gov", credibility_score=0.8),
    ]
    across = [
        Source(id="S1", url="https://bls.gov/a", title="A", domain="bls.gov", credibility_score=0.8),
        Source(id="S2", url="https://nature.com/b", title="B", domain="nature.com", credibility_score=0.8),
    ]
    one_voice, _, _ = score_claim("c", ["S1", "S2"], [], same)
    two_voices, _, _ = score_claim("c", ["S1", "S2"], [], across)
    assert two_voices > one_voice


def test_a_critic_hit_lowers_confidence(fixture_sources):
    clean, _, _ = score_claim("c", ["S1", "S2"], ["S3"], fixture_sources)
    hit, _, _ = score_claim("c", ["S1", "S2"], ["S3"], fixture_sources, critic_landed_hit=True)
    assert hit < clean


def test_weak_sourcing_flagged_by_the_bias_checker_lowers_confidence(fixture_sources):
    report = BiasReport(weakly_sourced_claims=["remote work raises output"])
    clean, _, _ = score_claim("remote work raises output", ["S1"], [], fixture_sources)
    flagged, _, _ = score_claim(
        "remote work raises output", ["S1"], [], fixture_sources, bias_report=report
    )
    assert flagged < clean


def test_confidence_never_reaches_certainty(fixture_sources):
    confidence, _, _ = score_claim("c", ["S1", "S2"], [], fixture_sources)
    assert 0.0 < confidence <= 0.98


def test_evenly_split_evidence_marks_a_claim_contested(conflicting_sources):
    _, contested, rationale = score_claim("c", ["S1"], ["S2"], conflicting_sources)
    assert contested
    assert "contested" in rationale


def test_one_sided_evidence_is_not_contested(fixture_sources):
    _, contested, _ = score_claim("c", ["S1", "S2"], [], fixture_sources)
    assert not contested


def test_objection_without_evidence_is_not_contested(fixture_sources):
    """A Critic hit with no opposing source is rhetoric, not a real split."""
    _, contested, _ = score_claim("c", ["S1"], [], fixture_sources, critic_landed_hit=True)
    assert not contested


def test_rationale_explains_the_number(fixture_sources):
    _, _, rationale = score_claim("c", ["S1", "S2"], ["S3"], fixture_sources)
    assert "support" in rationale and "independent domain" in rationale


def test_score_claims_drops_invented_citations(fixture_sources):
    claims = score_claims([("c", ["S1", "S99"], ["S404"], False)], fixture_sources)
    assert claims[0].supporting_source_ids == ["S1"]
    assert claims[0].opposing_source_ids == []


def test_score_claims_numbers_claims_in_order(fixture_sources):
    claims = score_claims(
        [("a", ["S1"], [], False), ("b", ["S2"], [], False)], fixture_sources
    )
    assert [c.id for c in claims] == ["C1", "C2"]


def test_uncertainty_mode_triggers_on_conflicting_evidence(conflicting_sources):
    """The headline feature: split evidence must not resolve into a verdict."""
    claims = score_claims(
        [
            ("remote work raises output", ["S1"], ["S2"], True),
            ("remote work lowers output", ["S2"], ["S1"], True),
        ],
        conflicting_sources,
    )
    assert all(c.contested for c in claims)
    assert is_uncertain(claims)


def test_uncertainty_mode_stays_off_when_evidence_agrees(fixture_sources):
    claims = score_claims(
        [("a", ["S1", "S2"], [], False), ("b", ["S1", "S2"], [], False)], fixture_sources
    )
    assert not any(c.contested for c in claims)
    assert not is_uncertain(claims)


def test_no_claims_is_uncertain_by_definition():
    assert is_uncertain([])


def test_verdict_confidence_is_the_mean_of_its_claims():
    assert verdict_confidence([_claim("C1", 0.8), _claim("C2", 0.4)]) == 0.6
    assert verdict_confidence([]) > 0
