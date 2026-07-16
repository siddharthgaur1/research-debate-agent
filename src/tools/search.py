"""Web search behind one interface.

Tavily and SerpAPI return wildly different JSON. Both adapters normalise to
`SourceStub`, so no agent, test or graph node ever learns which backend answered.
Swapping providers is an env var, not a code change.
"""

from __future__ import annotations

from typing import Protocol

import requests

from ..config import get_settings
from ..state.schema import SourceStub

_TIMEOUT = 30


class SearchBackend(Protocol):
    """The one shape every search provider must satisfy."""

    def __call__(self, query: str, max_results: int = 5) -> list[SourceStub]:
        ...


def tavily_search(query: str, max_results: int = 5) -> list[SourceStub]:
    """Search via Tavily. Returns [] rather than raising on a bad payload."""
    settings = get_settings()
    resp = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": settings.tavily_api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "advanced",
        },
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    results = resp.json().get("results", []) or []
    return [
        SourceStub(
            url=r.get("url", ""),
            title=r.get("title", "") or r.get("url", ""),
            snippet=r.get("content", "") or "",
            published=r.get("published_date"),
        )
        for r in results
        if r.get("url")
    ]


def serpapi_search(query: str, max_results: int = 5) -> list[SourceStub]:
    """Search via SerpAPI's Google engine. Same contract as `tavily_search`."""
    settings = get_settings()
    resp = requests.get(
        "https://serpapi.com/search.json",
        params={
            "q": query,
            "api_key": settings.serpapi_key,
            "engine": "google",
            "num": max_results,
        },
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    results = resp.json().get("organic_results", []) or []
    return [
        SourceStub(
            url=r.get("link", ""),
            title=r.get("title", "") or r.get("link", ""),
            snippet=r.get("snippet", "") or "",
            published=r.get("date"),
        )
        for r in results[:max_results]
        if r.get("link")
    ]


_BACKENDS: dict[str, SearchBackend] = {
    "tavily": tavily_search,
    "serpapi": serpapi_search,
}


def get_backend() -> SearchBackend:
    """The backend named by SEARCH_PROVIDER."""
    return _BACKENDS[get_settings().search_provider]


def search(query: str, max_results: int = 5) -> list[SourceStub]:
    """Search using whichever backend is configured."""
    return get_backend()(query, max_results=max_results)
