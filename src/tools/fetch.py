"""Fetch a page and reduce it to usable text.

Boilerplate removal is trafilatura's whole job and it is very good at it, so we
don't hand-roll a nav/footer stripper. The regex path exists only for when
trafilatura declines to extract (JS-only pages, odd markup) — crude, but better
than dropping the source entirely.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

import requests
import trafilatura

from ..config import get_settings

_TIMEOUT = 20
_UA = "Mozilla/5.0 (compatible; research-debate-agent/1.0)"

_TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
_ANY_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\n{3,}")


def domain_of(url: str) -> str:
    """Registrable-ish host for a url, lowercased, without `www.`."""
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _crude_text(html: str) -> str:
    html = _TAG_RE.sub(" ", html)
    text = _ANY_TAG_RE.sub(" ", html)
    return re.sub(r"[ \t]{2,}", " ", text)


def clean_html(html: str) -> str:
    """Strip boilerplate from raw HTML, falling back to a naive tag strip."""
    extracted = trafilatura.extract(
        html, include_comments=False, include_tables=False, no_fallback=False
    )
    text = extracted or _crude_text(html)
    return _WS_RE.sub("\n\n", text).strip()


def truncate(text: str, budget: int | None = None) -> str:
    """Cut text to the per-source char budget on a word boundary."""
    limit = budget if budget is not None else get_settings().fetch_char_budget
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    return (cut[:space] if space > limit * 0.8 else cut).rstrip() + " …[truncated]"


def fetch_text(url: str) -> str:
    """Fetch `url` and return cleaned, truncated text. Empty string on any failure.

    A dead link is a normal outcome of web search, not a run-ending error — the
    Researcher just keeps the snippet and moves on.
    """
    try:
        resp = requests.get(url, timeout=_TIMEOUT, headers={"User-Agent": _UA})
        resp.raise_for_status()
        if "html" not in resp.headers.get("Content-Type", "") and not resp.text:
            return ""
        return truncate(clean_html(resp.text))
    except (requests.RequestException, ValueError):
        return ""
