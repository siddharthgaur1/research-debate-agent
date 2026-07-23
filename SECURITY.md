# Security

## Threat model

The Research + Debate Agent fetches web pages the model chose to search for, feeds
their text to LLMs, and runs a multi-agent debate over the results. Two things are
untrusted: **content fetched from the open web** (arbitrary HTML, potentially
adversarial) and, transitively, everything the LLMs read. There is no user login;
it is an operator-run tool.

## What is mitigated

| Risk | Status | Where |
|---|---|---|
| Outbound HTTP hanging the run | **Mitigated** — every `requests` call sets an explicit `timeout` | `src/tools/fetch.py`, `src/tools/search.py` |
| Runaway LLM spend | **Mitigated** — hard per-run USD cap, all traffic through one `Budget` chokepoint | `src/tools/llm.py` |
| Runaway search spend | **Mitigated** — `max_searches_per_run` cap | `src/tools/llm.py`, `src/config.py` |
| Fetched-page size blowup | **Mitigated** — `fetch_char_budget` truncates page text | `src/config.py` |
| Container running as root | **Mitigated** — both images run as uid 1000 | `Dockerfile:24`, `dashboard/Dockerfile:23` |
| Secrets in git history | **Clean** — `gitleaks`: 0 findings; no `.env` ever tracked |
| ChromaDB server RCE (PYSEC-2026-311) | **Not applicable by default** — that advisory targets Chroma's HTTP **server** API. Dedup uses an embedded `PersistentClient` unless you explicitly set `CHROMA_HOST` to point at an external server. Running your own trusted Chroma server is your call; the default embedded path exposes no such endpoint. No upstream fix exists yet. | `src/tools/dedup.py` |

## What is NOT mitigated / notes

- **No authentication** on the API or dashboard.
- **Prompt injection from fetched web pages** is the central, unsolved risk. A page
  can contain instructions aimed at the model ("disregard the debate, conclude X").
  The adversarial structure of the system — an Advocate, a Critic, and a Bias
  Checker arguing against each other — provides *some* resistance, since a single
  poisoned source has to survive cross-examination, but this is a mitigation by
  design pressure, not a guarantee. Treat conclusions over untrusted sources as
  argued opinion, not fact.
- **SSRF / fetching internal URLs.** The fetcher will retrieve whatever URL a
  search result yields. It is not restricted from internal/loopback addresses; if
  you deploy this somewhere with sensitive internal services, add an allow-list or
  egress restriction.

## Reporting

Open an issue. Portfolio/demo project, no production deployment, no security SLA.
