"""Search adapters, credibility scoring, dedup, and page cleaning."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.state.schema import Source, SourceStub
from src.tools import dedup, search
from src.tools.credibility import parse_date, score_source, score_sources
from src.tools.fetch import clean_html, domain_of, truncate


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


# --------------------------------------------------------------------------- search


def test_tavily_adapter_normalises_to_source_stubs(monkeypatch):
    payload = {
        "results": [
            {
                "url": "https://bls.gov/a",
                "title": "A",
                "content": "snippet a",
                "published_date": "2024-01-01",
            },
            {"url": "", "title": "dropped: no url"},
        ]
    }
    monkeypatch.setattr(search.requests, "post", lambda *a, **k: _FakeResponse(payload))

    stubs = search.tavily_search("q")

    assert [type(s) for s in stubs] == [SourceStub]
    assert stubs[0].url == "https://bls.gov/a"
    assert stubs[0].snippet == "snippet a"
    assert stubs[0].published == "2024-01-01"


def test_serpapi_adapter_normalises_to_the_same_shape(monkeypatch):
    payload = {
        "organic_results": [
            {"link": "https://bls.gov/a", "title": "A", "snippet": "snippet a", "date": "2024-01-01"}
        ]
    }
    monkeypatch.setattr(search.requests, "get", lambda *a, **k: _FakeResponse(payload))

    stubs = search.serpapi_search("q")

    assert [type(s) for s in stubs] == [SourceStub]
    assert stubs[0].url == "https://bls.gov/a"
    assert stubs[0].snippet == "snippet a"


def test_both_adapters_satisfy_one_interface(monkeypatch):
    """The point of the abstraction: callers never learn which backend answered."""
    tavily = {"results": [{"url": "https://a.gov/x", "title": "T", "content": "c"}]}
    serp = {"organic_results": [{"link": "https://a.gov/x", "title": "T", "snippet": "c"}]}
    monkeypatch.setattr(search.requests, "post", lambda *a, **k: _FakeResponse(tavily))
    monkeypatch.setattr(search.requests, "get", lambda *a, **k: _FakeResponse(serp))

    from_tavily = search.tavily_search("q")
    from_serp = search.serpapi_search("q")

    assert from_tavily[0].model_dump() == from_serp[0].model_dump()


def test_search_provider_env_selects_the_backend(monkeypatch):
    monkeypatch.setenv("SEARCH_PROVIDER", "serpapi")
    from src.config import get_settings

    get_settings.cache_clear()
    assert search.get_backend() is search.serpapi_search

    monkeypatch.setenv("SEARCH_PROVIDER", "tavily")
    get_settings.cache_clear()
    assert search.get_backend() is search.tavily_search


# ---------------------------------------------------------------------- credibility


def _iso(days_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).date().isoformat()


def test_gov_outranks_a_random_blog():
    gov = Source(id="A", url="https://bls.gov/x", title="T", domain="bls.gov", published=_iso(30))
    blog = Source(
        id="B", url="https://x.substack.com/p", title="T", domain="substack.com", published=_iso(30)
    )

    gov_score, gov_why, _ = score_source(gov)
    blog_score, blog_why, _ = score_source(blog)

    assert gov_score > blog_score
    assert "institutional" in gov_why
    assert "self-published" in blog_why


def test_edu_outranks_an_unknown_domain():
    edu = Source(id="A", url="https://mit.edu/x", title="T", domain="mit.edu", published=_iso(30))
    unknown = Source(
        id="B", url="https://randomsite.io/x", title="T", domain="randomsite.io", published=_iso(30)
    )
    assert score_source(edu)[0] > score_source(unknown)[0]


def test_a_stale_source_is_penalised():
    fresh = Source(id="A", url="https://bls.gov/x", title="T", domain="bls.gov", published=_iso(30))
    stale = Source(id="B", url="https://bls.gov/y", title="T", domain="bls.gov", published=_iso(3000))

    fresh_score, _, _ = score_source(fresh)
    stale_score, stale_why, _ = score_source(stale)

    assert stale_score < fresh_score
    assert "stale" in stale_why


def test_reasoning_is_stored_not_just_the_number():
    source = Source(id="A", url="https://bls.gov/x", title="T", domain="bls.gov", published=_iso(30))
    _, reasoning, _ = score_source(source)
    assert reasoning and "bls.gov" in reasoning


def test_corroboration_only_counts_independent_domains():
    """Same story on one domain twice must not count as corroboration."""
    a = Source(
        id="A", url="https://bls.gov/x", title="remote work output rose four percent",
        snippet="remote knowledge workers output rose", domain="bls.gov", published=_iso(30),
    )
    same_domain = Source(
        id="B", url="https://bls.gov/y", title="remote work output rose four percent",
        snippet="remote knowledge workers output rose", domain="bls.gov", published=_iso(30),
    )
    other_domain = Source(
        id="C", url="https://nature.com/z", title="remote work output rose four percent",
        snippet="remote knowledge workers output rose", domain="nature.com", published=_iso(30),
    )

    _, _, from_same = score_source(a, [a, same_domain])
    _, _, from_other = score_source(a, [a, other_domain])

    assert from_same == []
    assert from_other == ["C"]


def test_score_sources_fills_domain_and_scores_the_pool(fixture_sources):
    scored = score_sources(fixture_sources)
    assert all(s.credibility_reasoning for s in scored)
    by_id = {s.id: s for s in scored}
    assert by_id["S1"].credibility_score > by_id["S3"].credibility_score


def test_parse_date_handles_junk_and_prose():
    assert parse_date(None) is None
    assert parse_date("not a date at all") is None
    assert parse_date("2024-01-01").year == 2024
    assert parse_date("Published May 1, 2023").year == 2023


# ---------------------------------------------------------------------------- fetch


def test_domain_of_strips_www():
    assert domain_of("https://www.bls.gov/a/b") == "bls.gov"
    assert domain_of("https://nature.com/x") == "nature.com"


def test_clean_html_drops_scripts_and_markup():
    text = clean_html("<html><body><script>evil()</script><p>Real content here.</p></body></html>")
    assert "evil" not in text
    assert "Real content" in text


def test_truncate_respects_the_budget():
    assert truncate("word " * 500, budget=50).startswith("word")
    assert len(truncate("word " * 500, budget=50)) < 80
    assert truncate("short", budget=50) == "short"


def test_fetch_text_returns_empty_on_a_dead_link(monkeypatch):
    """A dead link is a normal search outcome, not a run-ending error."""
    import requests

    from src.tools import fetch

    def boom(*a, **k):
        raise requests.RequestException("dead")

    monkeypatch.setattr(fetch.requests, "get", boom)
    assert fetch.fetch_text("https://gone.example") == ""


# ---------------------------------------------------------------------------- dedup


class _FakeCollection:
    """A tiny in-memory stand-in for a Chroma collection using cosine distance."""

    def __init__(self):
        self.ids: list[str] = []
        self.embeddings: list[list[float]] = []

    def add(self, ids, embeddings, metadatas=None):
        self.ids.extend(ids)
        self.embeddings.extend(embeddings)

    def query(self, query_embeddings, n_results, include=None):
        target = query_embeddings[0]
        best_id, best_distance = None, 2.0
        for cid, emb in zip(self.ids, self.embeddings):
            distance = 1.0 - _cosine(target, emb)
            if distance < best_distance:
                best_id, best_distance = cid, distance
        if best_id is None:
            return {"ids": [[]], "distances": [[]]}
        return {"ids": [[best_id]], "distances": [[best_distance]]}


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


@pytest.fixture
def fake_chroma(monkeypatch):
    collection = _FakeCollection()
    monkeypatch.setattr(dedup, "collection_for", lambda run_id: collection)
    return collection


def test_near_duplicate_sources_are_merged(fake_chroma, monkeypatch):
    """The headline dedup guarantee: one story republished is one source."""
    originals = [
        Source(id="S1-1", url="https://reuters.com/a", title="Study finds X", domain="reuters.com"),
        Source(id="S2-1", url="https://yahoo.com/a", title="Study finds X", domain="yahoo.com"),
        Source(id="S3-1", url="https://bls.gov/b", title="Totally different topic", domain="bls.gov"),
    ]
    # First two are near-identical vectors; the third is orthogonal.
    monkeypatch.setattr(
        dedup, "_embed", lambda texts: [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]
    )

    kept = dedup.dedup_sources("run1", originals)

    assert len(kept) == 2
    assert kept[0].id == "S1-1"
    assert kept[0].merged_from == ["https://yahoo.com/a"]
    assert kept[1].id == "S3-1"


def test_distinct_sources_are_all_kept(fake_chroma, monkeypatch):
    originals = [
        Source(id="S1-1", url="https://a.gov/1", title="Alpha", domain="a.gov"),
        Source(id="S2-1", url="https://b.gov/2", title="Beta", domain="b.gov"),
    ]
    monkeypatch.setattr(dedup, "_embed", lambda texts: [[1.0, 0.0], [0.0, 1.0]])

    kept = dedup.dedup_sources("run1", originals)

    assert len(kept) == 2
    assert all(not s.merged_from for s in kept)


def test_dedup_assigns_embedding_ids(fake_chroma, monkeypatch):
    originals = [Source(id="S1-1", url="https://a.gov/1", title="Alpha", domain="a.gov")]
    monkeypatch.setattr(dedup, "_embed", lambda texts: [[1.0, 0.0]])

    kept = dedup.dedup_sources("run1", originals)

    assert kept[0].embedding_id == "S1-1"


def test_dedup_of_an_empty_pool_is_a_noop(fake_chroma):
    assert dedup.dedup_sources("run1", []) == []
