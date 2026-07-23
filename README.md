# Research + Debate Agent

**A team of LLM agents researches a contestable question, argues both sides, audits its own sources for bias, and arbitrates — every claim cited and confidence-scored.**

[![CI](https://github.com/siddharthgaur1/research-debate-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/siddharthgaur1/research-debate-agent/actions/workflows/ci.yml) [![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Runs on free tiers](https://img.shields.io/badge/runs%20on-free%20Groq%20%2B%20Tavily-brightgreen)](#run-on-free-tiers)

> **Live demo:** not a zero-key click-through — the agents need a model and live
> web search, so there is no keyless public demo. It runs on **free tiers** (Groq
> for the LLM, Tavily's free search) with no credit card — see below.

## Run on free tiers

```bash
git clone https://github.com/siddharthgaur1/research-debate-agent
cd research-debate-agent
cp .env.example .env
# in .env, all free:
OPENAI_API_KEY=gsk_...                          # https://console.groq.com/keys
OPENAI_BASE_URL=https://api.groq.com/openai/v1
REASONING_MODEL=llama-3.3-70b-versatile
CHEAP_MODEL=llama-3.1-8b-instant
TAVILY_API_KEY=tvly-...                         # https://app.tavily.com (free tier)

docker compose up --build     # dashboard :8501 · API :8000
```

The chat model runs on any OpenAI-compatible endpoint (Groq/Ollama) via
`OPENAI_BASE_URL`. Redis is optional (in-memory fallback). Source-dedup embeddings
are the one piece that still calls OpenAI — it is a quality feature, and a debate
runs without it. Security model, including the web-content prompt-injection and
SSRF threat surface: [SECURITY.md](SECURITY.md).

---

Ask a contestable research question. Instead of one model summarising whatever it
found first, a team of agents researches it in parallel, **argues both sides**, audits
its own sources for bias, and arbitrates the result into a report where every claim
carries a confidence score and a citation trail.

The debate is visible. You watch the Advocate build a case, the Critic land hits, the
Bias Checker flag one-sided sourcing, and the Arbitrator refuse to call it when the
evidence is genuinely split. That visibility is the product, not a debug view.

---

## What it solves

A single-pass research summariser has three failure modes it cannot see in itself:

1. **It agrees with itself.** One model, one pass, one framing — nothing in the loop
   is incentivised to attack the conclusion, so weak evidence sails through.
2. **It launders volume into confidence.** Six links to the same reprinted wire story
   read as six sources. Nothing checks whether the evidence is independent.
3. **It manufactures consensus.** Asked a genuinely split question, an LLM produces a
   tidy "on balance, yes" — because tidy reads like competence. That tidiness is a lie
   about the state of the evidence.

This system attacks each one structurally: an adversarial Critic whose job is to break
the case, ChromaDB dedup so republished stories collapse to one voice, and a computed
(not model-authored) confidence score that can force the report into **uncertainty
mode** when the evidence splits.

---

## Architecture

```
                            POST /research
                                  │
                                  ▼
                          ┌───────────────┐
                          │  Supervisor   │  question ─> falsifiable hypothesis
                          │               │             + 3-5 research subtasks
                          └───────┬───────┘
                                  │  Send() fan-out — one branch per subtask
             ┌────────────────────┼────────────────────┐
             ▼                    ▼                    ▼
     ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
     │ Researcher T1 │    │ Researcher T2 │    │ Researcher T3 │   (concurrent)
     │ search→fetch  │    │ search→fetch  │    │ search→fetch  │
     │ →clean→score  │    │ →clean→score  │    │ →clean→score  │
     └───────┬───────┘    └───────┬───────┘    └───────┬───────┘
             └────────────────────┼────────────────────┘
                                  ▼
                          ┌───────────────┐
                          │    Gather     │  ChromaDB dedup ─> renumber S1..Sn
                          │               │  ─> re-score corroboration
                          └───────┬───────┘
                                  ▼
                    ┌─────────────────────────┐
                    │   Advocate  ⇄  Critic   │  capped at MAX_DEBATE_ROUNDS
                    │   (FOR)        (AGAINST)│  every point cites [S#]
                    └─────────────┬───────────┘
                                  ▼
                          ┌───────────────┐
                          │ Bias Checker  │  audits the SOURCE POOL, not the argument
                          └───────┬───────┘
                                  ▼
                          ┌───────────────┐
                          │  Arbitrator   │  claims + evidence assignment
                          └───────┬───────┘  confidence computed in code ──┐
                                  ▼                                        │
                          ┌───────────────┐                    tools/confidence.py
                          │    Report     │  drop uncited claims ─> PDF
                          └───────┬───────┘
                                  ▼
                                 END

  Every turn ──> Redis (list + pub/sub) ──> SSE ──> Streamlit live debate panel
  Every node ──> SQLite transitions table (append-only audit log, replayable)
```

## Agent roster

| Agent | Role | Why it exists |
|---|---|---|
| **Supervisor** | Turns the question into a *falsifiable hypothesis* + 3-5 subtasks | "Is remote work productive?" can't be argued. "Remote work raises measured productivity" can. Without a target the debate talks past itself. |
| **Researcher** (×N, parallel) | Search → fetch → clean → score, one subtask each | Concurrency is the cheap win; isolation keeps each angle from contaminating the others. Explicitly told *not* to argue. |
| **Gather** | Dedup, renumber, re-score corroboration | The join point. Only here can you see the whole pool — corroboration is a pool-level property. |
| **Advocate** | Strongest evidence-based case **FOR** | Every point cites `[S#]`. An uncited point is dropped, not hand-waved. |
| **Critic** | Strongest case **AGAINST** + audits the Advocate | Counterevidence *and* overreach. Required to concede where the Advocate is right — a Critic that only disagrees is just a second Advocate. |
| **Bias Checker** | Audits the **source pool**, not the argument | Both debaters share one pool. If it's skewed, both cases inherit the skew and the debate looks rigorous while being wrong. |
| **Arbitrator** | Extracts claims, assigns evidence, writes the verdict | Judgement only. It does **not** pick confidence numbers. |
| **Report** | Assembles the artifact, enforces citations, exports PDF | No LLM: the Arbitrator already decided everything. A model here could only paraphrase away a caveat. |

---

## Tech stack

| Choice | Why this one |
|---|---|
| **LangGraph** | The debate is a state machine with a cycle (Advocate ⇄ Critic) and a fan-out. That's a graph, not a chain. `Send()` gives real parallel researchers; conditional edges give the round cap. |
| **OpenAI GPT-4o** | Reasoning model for debate/arbitration; `gpt-4o-mini` for mechanical research summarisation. Two tiers because summarising a page doesn't need the expensive model. |
| **Tavily / SerpAPI** | Both behind one `search(query) -> list[SourceStub]` interface, chosen by `SEARCH_PROVIDER`. Tavily default (cleaner content extraction); SerpAPI when you need raw Google. Swapping is an env var. |
| **trafilatura** | Boilerplate stripping is its entire job and it's very good at it. Hand-rolling a nav/footer stripper is the classic 50-line trap. |
| **ChromaDB** | Cosine-space vector store for near-duplicate detection. Configured explicitly for cosine — Chroma defaults to L2, which would make the similarity threshold meaningless. |
| **Redis** | Two jobs: live turn streaming (list for replay + pub/sub for the tail) and LangGraph checkpointing. Both optional-by-design — a dead cache degrades the UI, it doesn't kill a run. |
| **SQLite** | Append-only `transitions` table = the run is replayable and auditable. Two tables, no joins, so stdlib `sqlite3` beats an ORM. |
| **FastAPI + SSE** | Traffic is one-way (server pushes turns). SSE reconnects on its own and survives proxies that mangle upgrade headers. A WebSocket is more machinery for a smaller feature. |
| **Streamlit** | Separate service, talks HTTP only, shares no code with the API. A dashboard that imports the agents is a dashboard that can't be deployed separately. |
| **ReportLab** | Platypus handles flow/pagination/table breaks. Only the content order is worth hand-writing. |

---

## Setup

```bash
git clone https://github.com/<you>/research-debate-agent.git
cd research-debate-agent
cp .env.example .env      # fill in OPENAI_API_KEY and TAVILY_API_KEY
docker-compose up --build
```

- Dashboard → http://localhost:8501
- API docs → http://localhost:8000/docs

`src/config.py` validates the environment at import time and **fails loudly** on a
missing key rather than surfacing an `AttributeError` six nodes into a debate. If you
set `SEARCH_PROVIDER=serpapi`, `SERPAPI_KEY` becomes required — the validator enforces
the pairing.

### Local (no Docker)

```bash
python -m venv .venv && .venv/Scripts/activate   # or: source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.api.app:app --reload                 # terminal 1
streamlit run dashboard/app.py                   # terminal 2
```

Without `CHROMA_HOST` set, Chroma runs as a local file store at `CHROMA_DIR` — same
code path, no server needed.

---

## Running a query

Ask something **contestable**. "What is the capital of France" has nothing to debate.

Via the dashboard: type the question, hit **Run debate**, watch the Live debate tab.

Via the API:

```bash
curl -X POST localhost:8000/research \
  -H 'Content-Type: application/json' \
  -d '{"question": "Is remote work more productive than office work?"}'
# {"run_id":"a1b2c3d4e5f6","status":"pending"}

curl -N localhost:8000/research/a1b2c3d4e5f6/stream   # live debate turns (SSE)
curl    localhost:8000/research/a1b2c3d4e5f6          # full result
curl -O localhost:8000/research/a1b2c3d4e5f6/report.pdf
```

| Endpoint | Purpose |
|---|---|
| `POST /research` | Start a run, returns `run_id` immediately (202) |
| `GET /research/{id}` | Status, transcript, claims, verdict, sources |
| `GET /research/{id}/stream` | SSE — replays past turns, then tails live ones |
| `GET /research/{id}/history` | The append-only audit trail |
| `GET /research/{id}/report.pdf` | PDF export |
| `GET /health` | Liveness + a real store round-trip |

### Example run

*Illustrative — the shape of real output. Actual sources, numbers and verdict depend
on what's live on the web at query time.*

**Q: "Is remote work more productive than office work?"**

```
🧭 supervisor    Main finding to test:
                   Remote work increases measured productivity for knowledge workers.
                 Research angles:
                   T1. Studies measuring remote vs office output — core evidence
                   T2. Where remote work reduces output — refutation angle
                   T3. Confounders: self-selection, role type, tenure

🔬 Researcher·T1 Two longitudinal studies report output gains of 3-5% [...]
🔬 Researcher·T2 Evidence of collaboration and onboarding costs [...]
🔬 Researcher·T3 Self-selection is a live confounder in most datasets [...]

📚 gather        Pooled 15 results into 9 distinct sources (6 near-duplicates merged).

✅ Advocate      1. Measured output rose 4% in a controlled trial [S1], corroborated
                    independently [S4].
                 2. Effect holds after controlling for tenure [S2]. ...

⚔️ Critic        1. [S1]'s sample is self-selected volunteers — the effect may be
                    selection, not remote work [S6].
                 2. Advocate overreached: [S2] controls for tenure, not role type.
                 Concessions: the 4% output finding itself is well-sourced.

🔍 Bias Checker  Pool leans 44% on one outlet; two sources are >5y old. The
                 collaboration-cost claim rests on a single vendor blog [S8].

⚖️ Arbitrator    UNCERTAINTY MODE — the evidence is genuinely split.
                 Claims (4, 2 contested):
                 - ⚖️ Remote work raises measured output        (confidence 0.41)
                 -    The 4% output finding is well-sourced      (confidence 0.78)
                 - ⚖️ Collaboration costs offset output gains    (confidence 0.38)
                 -    Self-selection confounds most estimates    (confidence 0.71)

📄 Report        Report ready: 4 claims (2 contested) across 9 sources.
                 Evidence is split — presenting both sides rather than a verdict.
```

---

## Key design decisions

### Why multi-agent debate instead of one summariser

A single summariser has no adversary. It grades its own homework, and its confidence
tracks its fluency rather than its evidence. Splitting the roles means the case *for*
is built by something that wants it to win, the case *against* by something that wants
it to lose, and neither one gets to write the verdict. The Critic's mandate to attack
the Advocate's **reasoning** — not just cite counterevidence — is what surfaces
overreach that a summariser structurally cannot notice in itself.

The Critic is also required to *concede*. Without that, it degenerates into
contrarianism and the Arbitrator loses its signal: if everything is disputed, nothing
is. Concessions are what make "contested" mean something.

### How confidence is computed

**The model never picks the number.** It decides *which* sources support and oppose
each claim and whether the Critic landed a real hit. `tools/confidence.py` turns those
judgements into a number, in code:

```
balance      = credibility-weighted support / (support + opposition)
independence = f(distinct supporting domains)   # 0→0.30, 1→0.70, 2→0.85, 3+→1.0
confidence   = balance × independence
             × 0.75 if the Critic landed a substantive hit
             × 0.80 if the Bias Checker flagged the sourcing
             clamped to [0.02, 0.98]
```

Two reasons for the split. First, a model asked for "confidence: 0.0-1.0" produces a
fluent guess that's reproducible only by luck and that nobody can unit-test — every
number above is derived from inputs a reader can check, and is covered by tests.
Second, it removes the model's ability to *reach* a conclusion it prefers: assigning
evidence honestly is the only lever it has, and the arithmetic does the rest.

Nothing is ever 0 or 1. Certainty from a handful of web sources is the failure mode,
not the goal.

### How contested claims are surfaced

A claim is **contested** when real opposing evidence exists *and* either the balance is
near-even (within `CONTESTED_MARGIN`) or the Critic landed a hit. Note the first
condition: an objection with no opposing source is rhetoric, not a split — otherwise a
sufficiently argumentative Critic could mark everything contested.

When the contested fraction crosses `UNCERTAINTY_RATIO` (default ⅓), the run enters
**uncertainty mode**: the report presents both sides with their strongest support
instead of naming a winner. Because this is computed from claim data rather than asked
of the model, an LLM's pull toward a tidy answer *cannot* smooth it over. A run with no
claims at all is uncertain by definition — we found nothing to be confident about.

### Why ChromaDB dedup

Search hands back the same wire story republished six times. Left alone, the Advocate
cites "six sources" that are one press release, and the debate inherits a false sense
of weight — the exact failure this project exists to prevent. Sources are embedded and
anything ≥ `DEDUP_THRESHOLD` cosine similarity folds into the first seen.

The survivor keeps the duplicates' URLs in `merged_from`, so the citation trail still
shows the story ran in six places without pretending that's six pieces of evidence.
Corroboration scoring reinforces this: it only counts **distinct domains**, because
five pages from one outlet is one source wearing five hats.

### State: two source lists, on purpose

`raw_sources` is `Annotated[list, operator.add]` because parallel researchers write it
concurrently — without a reducer LangGraph rejects the concurrent update. But that
reducer makes it *append-only*, so the Gather node can't replace it with the deduped
pool; returning the deduped list would append it to the raw one. Hence a separate
`sources` field, written once by Gather, holding the canonical pool the debate cites.
The count difference between them is also what makes dedup observable in the UI.

### Failure isolation

A dead link or a rate-limited search on one angle costs that angle, not the debate:
researcher branches catch their own failures, log a warning, and the surviving
researchers carry on. Every other node routes to a terminal `fail` node that records
*why* — a half-finished debate stays auditable instead of dying in a traceback.

---

## Tests

```bash
pytest tests/ -v      # 73 tests, no network
```

Every OpenAI and search call is mocked; all agents reach the model through one
`Budget` chokepoint, so patching its two methods makes the whole suite offline.

Coverage worth naming:

- **Search** — Tavily and SerpAPI adapters produce *byte-identical* `SourceStub`s from
  their different payloads; `SEARCH_PROVIDER` selects the backend.
- **Dedup** — near-duplicates merge into one source with `merged_from` populated;
  distinct sources all survive.
- **Credibility** — `.gov` outranks a Substack; a stale source is penalised;
  corroboration ignores same-domain "agreement".
- **Confidence** — independent domains beat a single domain; Critic hits and bias flags
  lower confidence; an objection without evidence is *not* contested.
- **Uncertainty mode** — conflicting fixtures trigger it end to end; agreeing fixtures
  don't.
- **Graph** — a full run over fixture sources yields a verdict whose every citation
  resolves to a real source; fan-out runs one researcher per subtask; the debate loop
  honours its round cap; a failing researcher doesn't sink the run; every transition
  is persisted and replayable.
- **Citation guarantee** — claims citing invented ids (`[S99]`) are dropped by the
  Report node and the verdict's confidence is recomputed without them.

---

## What I'd improve with more time

- **Fetch is the bottleneck.** Researchers parallelise their own fetches, but the whole
  graph is synchronous. Async end to end would cut wall-clock substantially.
- **Dedup is O(n²)-ish in the pool and embeds every source even when the URL already
  matches.** A cheap URL/simhash pre-filter before spending embedding calls would cut
  most of the cost at these pool sizes.
- **Credibility is a hand-tuned heuristic.** It's reproducible and testable, which is
  why it isn't an LLM call — but the domain lists are Anglophone and hardcoded, and the
  weights are judgement, not calibration. Calibrating against a labelled set is the
  honest next step.
- **The Advocate/Critic loop doesn't re-search.** If the Critic identifies a specific
  evidence gap, the right move is a targeted follow-up search, not another round of
  arguing over the same pool.
- **The round cap is a fixed count, not convergence.** Stopping when the Critic stops
  landing new hits would spend rounds where they matter.
- **Redis checkpointing exists but nothing resumes.** The plumbing is there;
  `resume_run` isn't. A run that dies mid-debate currently restarts.
- **Costs are capped per run, not per tenant.** Fine for a demo, wrong for anything
  multi-user.
