"""Start runs.

Redis is a nice-to-have here, not a hard dependency: if the checkpointer can't be
built, the run still executes, it just can't be resumed. Refusing to answer a
question because a cache is down would be the wrong trade.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager

from ..config import get_settings
from ..persistence import stream
from ..persistence.store import RunStore
from ..state.schema import RunState, RunStatus, new_run_state
from .build import build_graph


def new_run_id() -> str:
    """Short, readable, and unique enough for a run directory name."""
    return uuid.uuid4().hex[:12]


@contextmanager
def _checkpointer() -> Iterator[object | None]:
    """Yield a Redis checkpointer, or None if Redis isn't reachable."""
    try:
        from langgraph.checkpoint.redis import RedisSaver
    except ImportError:
        yield None
        return

    try:
        with RedisSaver.from_conn_string(get_settings().redis_url) as saver:
            saver.setup()
            yield saver
    except Exception:  # noqa: BLE001 — Redis down must not block a run
        yield None


def _config(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}, "recursion_limit": 50}


def prepare_run(question: str, store: RunStore | None = None) -> RunState:
    """Register a run without executing it, so the API can return an id immediately."""
    state = new_run_state(new_run_id(), question)
    if store:
        store.create_run(state)
    return state


def execute_run(state: RunState, store: RunStore | None = None) -> RunState:
    """Run the debate graph to completion."""
    try:
        with _checkpointer() as checkpointer:
            graph = build_graph(store=store, checkpointer=checkpointer)
            final = graph.invoke(state, _config(state["run_id"]))
    except Exception as exc:  # noqa: BLE001 — surface as a failed run, not a 500
        final = {**state, "status": RunStatus.FAILED, "error": str(exc)}  # type: ignore[assignment]
        if store:
            store.record_transition(final, stage="run:failed")
        stream.publish_done(state["run_id"])
    return final  # type: ignore[return-value]


def start_run(question: str, store: RunStore | None = None) -> RunState:
    """Prepare and execute a run, blocking until it finishes."""
    return execute_run(prepare_run(question, store), store)
