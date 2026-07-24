"""FastAPI surface for the debate agent.

The streaming endpoint is SSE rather than WebSockets: the traffic is one-way
(server pushes turns, client just listens), and SSE reconnects on its own and
survives a proxy that mangles upgrade headers. A WebSocket would be strictly more
machinery for a strictly smaller feature.

Runs execute on a background thread. The graph is synchronous and a debate takes
minutes; holding the request open for it would be a timeout waiting to happen.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..config import get_settings
from ..graph.runner import execute_run, prepare_run
from ..persistence import stream
from ..persistence.store import RunStore
from ..state.schema import RunState, RunStatus

app = FastAPI(title="Research + Debate Agent", version="1.0.0")


@lru_cache(maxsize=1)
def store() -> RunStore:
    """The run store, created once on first use."""
    return RunStore(get_settings().db_path)


class ResearchRequest(BaseModel):
    """A question to research and debate."""

    question: str = Field(min_length=8, description="The research question.")


class RunCreated(BaseModel):
    """Acknowledgement that a run has been accepted."""

    run_id: str
    status: str


def _public(state: RunState) -> dict:
    """Shape a run for the API: the debate and its evidence, no raw page dumps."""
    verdict = state.get("verdict")
    return {
        "run_id": state.get("run_id"),
        "question": state.get("question"),
        "status": RunStatus(state.get("status", RunStatus.PENDING)).value,
        "error": state.get("error"),
        "hypothesis": state.get("hypothesis", ""),
        "subtasks": [s.model_dump() for s in state.get("subtasks", [])],
        "sources": [
            s.model_dump(exclude={"text"}) for s in state.get("sources", [])
        ],
        "raw_source_count": len(state.get("raw_sources", [])),
        "search_count": state.get("search_count", 0),
        "claims": [c.model_dump() for c in state.get("claims", [])],
        "verdict": verdict.model_dump() if verdict else None,
        "bias_report": (
            state["bias_report"].model_dump() if state.get("bias_report") else None
        ),
        "debate_transcript": [t.model_dump() for t in state.get("debate_transcript", [])],
        "messages": [m.model_dump() for m in state.get("messages", [])],
        "cost_usd": round(sum(u.cost_usd for u in state.get("token_usage", [])), 4),
        "report_available": bool(state.get("report_path")),
    }


def _load(run_id: str) -> RunState:
    state = store().get_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Unknown run: {run_id}")
    return state


@app.get("/health")
def health() -> dict:
    """Liveness plus a real check that the run store is reachable."""
    try:
        store().list_runs(limit=1)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"store unreachable: {exc}")
    return {"status": "ok"}


@app.post("/research", response_model=RunCreated, status_code=202)
def create_run(request: ResearchRequest) -> RunCreated:
    """Accept a question and start a debate in the background."""
    state = prepare_run(request.question, store())
    threading.Thread(
        target=execute_run, args=(state, store()), daemon=True
    ).start()
    return RunCreated(run_id=state["run_id"], status=RunStatus.PENDING.value)


@app.get("/research")
def list_runs(limit: int = 25) -> list[dict]:
    """Recent runs, newest first."""
    return store().list_runs(limit=limit)


@app.get("/research/{run_id}")
def get_run(run_id: str) -> dict:
    """Status, debate and result for one run."""
    return _public(_load(run_id))


@app.get("/research/{run_id}/history")
def get_history(run_id: str) -> list[dict]:
    """The audit trail: every state transition, in order."""
    _load(run_id)
    return [
        {"seq": seq, "stage": stage, "at": at}
        for seq, stage, at in store().history(run_id)
    ]


@app.get("/research/{run_id}/stream")
def stream_run(run_id: str) -> StreamingResponse:
    """Debate turns as Server-Sent Events.

    Replays the turns already recorded before tailing live ones, so a client that
    connects late — or reconnects — still sees the whole argument.
    """
    _load(run_id)

    def events() -> Iterator[str]:
        for turn in stream.past_turns(run_id):
            yield f"data: {json.dumps(turn)}\n\n"

        state = store().get_run(run_id)
        if state and state.get("status") in (RunStatus.COMPLETED, RunStatus.FAILED):
            yield "event: done\ndata: {}\n\n"
            return

        for payload in stream.tail(run_id):
            if payload:
                yield f"data: {payload}\n\n"
            else:
                yield ": keep-alive\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/research/{run_id}/report.pdf")
def get_report(run_id: str) -> FileResponse:
    """Download the run's PDF report."""
    state = _load(run_id)
    path = state.get("report_path")
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="Report not ready for this run.")
    return FileResponse(
        path, media_type="application/pdf", filename=f"report-{run_id}.pdf"
    )
