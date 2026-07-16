"""Wire the agents into the debate graph.

    supervisor
        ├── Send ──> researcher (one parallel branch per subtask)
        └────────────────┴──> gather (dedup + renumber)
                                 └──> advocate ⇄ critic   (capped at N rounds)
                                          └──> bias ──> arbitrator ──> report

Two things here are load-bearing beyond the wiring:

`_instrument` gives every node persistence and a failure net. A node that raises
does not take the run down with an opaque traceback — it records the failure and
routes to a terminal node, so a half-finished debate is still auditable.

Researchers are exempt from the "one failure fails the run" rule. A dead link or a
rate-limited search on one angle should cost that angle, not the debate; the
branch logs the failure, returns nothing, and the remaining researchers carry on.
"""

from __future__ import annotations

from typing import Callable

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ..agents.advocate import advocate_agent
from ..agents.arbitrator import arbitrator_agent
from ..agents.bias import bias_agent
from ..agents.critic import critic_agent
from ..agents.gather import gather_agent
from ..agents.report import report_agent
from ..agents.researcher import researcher_agent
from ..agents.supervisor import supervisor_agent
from ..config import get_settings
from ..persistence import stream
from ..persistence.store import RunStore
from ..state.schema import AgentMessage, RunState, RunStatus


def _merged(state: RunState, update: RunState) -> RunState:
    """Approximate what the graph will hold after this update, for persistence.

    Append-only channels are concatenated the way their reducers would; everything
    else is overwritten. Close enough for an audit row, and it avoids a second
    round-trip through the graph just to learn what we already know.
    """
    merged: RunState = {**state}  # type: ignore[assignment]
    additive = ("raw_sources", "debate_transcript", "messages", "token_usage")
    for key, value in update.items():
        if key in additive:
            merged[key] = list(state.get(key, [])) + list(value)  # type: ignore[literal-required]
        elif key == "search_count":
            merged[key] = state.get(key, 0) + value  # type: ignore[literal-required]
        else:
            merged[key] = value  # type: ignore[literal-required]
    return merged


def _instrument(
    name: str, fn: Callable[[RunState], RunState], store: RunStore | None
) -> Callable[[RunState], RunState]:
    """Wrap a node with persistence and a failure net."""

    def node(state: RunState) -> RunState:
        try:
            update = fn(state)
        except Exception as exc:  # noqa: BLE001 — terminal net; the run records why
            failure: RunState = {  # type: ignore[assignment]
                "status": RunStatus.FAILED,
                "error": f"{name}: {exc}",
                "messages": [AgentMessage(agent=name, content=str(exc), level="error")],
            }
            stream.publish_done(state["run_id"])
            if store:
                store.record_transition(_merged(state, failure), stage=f"{name}:failed")
            return failure

        if store:
            store.record_transition(_merged(state, update), stage=name)
        return update

    return node


def _instrument_researcher(store: RunStore | None) -> Callable:
    """Wrap a researcher branch so one bad angle can't sink the run.

    No persistence here: the branch receives a `Send` payload, not a full RunState,
    so there is no complete snapshot to record. The gather node's transition
    captures the merged pool immediately after, which is the state worth auditing.
    """

    def node(task) -> RunState:
        try:
            return researcher_agent(task)
        except Exception as exc:  # noqa: BLE001
            subtask = task.get("subtask")
            label = getattr(subtask, "id", "?")
            return {  # type: ignore[return-value]
                "raw_sources": [],
                "messages": [
                    AgentMessage(
                        agent=f"researcher:{label}",
                        content=f"Angle failed, continuing without it: {exc}",
                        level="warn",
                    )
                ],
            }

    return node


def _fan_out(state: RunState) -> list[Send] | str:
    """One researcher branch per subtask — the parallel fan-out."""
    if state.get("status") == RunStatus.FAILED:
        return "fail"
    subtasks = state.get("subtasks", [])
    if not subtasks:
        return "fail"
    return [
        Send(
            "researcher",
            {
                "run_id": state["run_id"],
                "question": state["question"],
                "hypothesis": state["hypothesis"],
                "subtask": subtask,
                "task_number": i,
                "token_usage": state.get("token_usage", []),
            },
        )
        for i, subtask in enumerate(subtasks, start=1)
    ]


def _continue_or_stop(next_stage: str) -> Callable[[RunState], str]:
    """A linear step that still respects failure."""

    def route(state: RunState) -> str:
        return "fail" if state.get("status") == RunStatus.FAILED else next_stage

    return route


def _after_critic(state: RunState) -> str:
    """Loop the debate until the round cap, then hand off to the Bias Checker."""
    if state.get("status") == RunStatus.FAILED:
        return "fail"
    if state.get("debate_round", 0) < get_settings().max_debate_rounds:
        return "advocate"
    return "bias"


def _fail(state: RunState) -> RunState:
    """Terminal node for unrecoverable errors."""
    stream.publish_done(state["run_id"])
    return {"status": RunStatus.FAILED}  # type: ignore[return-value]


def build_graph(store: RunStore | None = None, checkpointer=None):
    """Compile the debate graph. `store` is optional so tests can skip persistence."""
    graph = StateGraph(RunState)

    graph.add_node("supervisor", _instrument("supervisor", supervisor_agent, store))
    graph.add_node("researcher", _instrument_researcher(store))
    graph.add_node("gather", _instrument("gather", gather_agent, store))
    graph.add_node("advocate", _instrument("advocate", advocate_agent, store))
    graph.add_node("critic", _instrument("critic", critic_agent, store))
    graph.add_node("bias", _instrument("bias", bias_agent, store))
    graph.add_node("arbitrator", _instrument("arbitrator", arbitrator_agent, store))
    graph.add_node("report", _instrument("report", report_agent, store))
    graph.add_node("fail", _fail)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", _fan_out, ["researcher", "fail"])
    graph.add_edge("researcher", "gather")
    graph.add_conditional_edges("gather", _continue_or_stop("advocate"), ["advocate", "fail"])
    graph.add_conditional_edges("advocate", _continue_or_stop("critic"), ["critic", "fail"])
    graph.add_conditional_edges("critic", _after_critic, ["advocate", "bias", "fail"])
    graph.add_conditional_edges("bias", _continue_or_stop("arbitrator"), ["arbitrator", "fail"])
    graph.add_conditional_edges("arbitrator", _continue_or_stop("report"), ["report", "fail"])
    graph.add_edge("report", END)
    graph.add_edge("fail", END)

    return graph.compile(checkpointer=checkpointer)
