"""SQLite history, state round-tripping, and the Redis stream's failure behaviour."""

from __future__ import annotations

import redis

from src.persistence import stream
from src.persistence.store import RunStore
from src.state.schema import DebateTurn, RunStatus, Stance, new_run_state
from src.state.serde import state_from_json, state_to_json


def test_state_round_trips_through_json(base_state, bias_report):
    from src.state.schema import Verdict
    from src.tools.confidence import score_claims

    base_state["bias_report"] = bias_report
    base_state["claims"] = score_claims(
        [("Output rose.", ["S1"], ["S3"], False)], base_state["sources"]
    )
    base_state["verdict"] = Verdict(statement="Yes.", confidence=0.7, uncertainty_mode=False)
    base_state["debate_transcript"] = [
        DebateTurn(agent="Advocate", stance=Stance.FOR, content="c", cited_source_ids=["S1"])
    ]

    restored = state_from_json(state_to_json(base_state))

    # Pydantic payloads must come back as models, not bare dicts.
    assert restored["sources"][0].credibility_reasoning == base_state["sources"][0].credibility_reasoning
    assert restored["claims"][0].confidence == base_state["claims"][0].confidence
    assert restored["verdict"].statement == "Yes."
    assert restored["bias_report"].summary == bias_report.summary
    assert restored["debate_transcript"][0].stance is Stance.FOR
    assert restored["status"] is RunStatus.PENDING


def test_create_and_get_a_run(tmp_path):
    store = RunStore(tmp_path / "h.db")
    state = new_run_state("r1", "Is remote work more productive?")
    store.create_run(state)

    loaded = store.get_run("r1")

    assert loaded is not None
    assert loaded["question"] == "Is remote work more productive?"
    assert store.get_run("nope") is None


def test_transitions_are_appended_in_order(tmp_path):
    store = RunStore(tmp_path / "h.db")
    state = new_run_state("r1", "Q?")
    store.create_run(state)

    state["status"] = RunStatus.RESEARCHING
    store.record_transition(state, "supervisor")
    state["status"] = RunStatus.COMPLETED
    store.record_transition(state, "report")

    history = store.history("r1")
    assert [stage for _, stage, _ in history] == ["created", "supervisor", "report"]
    assert [seq for seq, _, _ in history] == [0, 1, 2]
    assert store.get_run("r1")["status"] is RunStatus.COMPLETED


def test_replay_reconstructs_each_step(tmp_path):
    store = RunStore(tmp_path / "h.db")
    state = new_run_state("r1", "Q?")
    store.create_run(state)
    state["hypothesis"] = "H"
    store.record_transition(state, "supervisor")

    full = store.replay("r1")
    partial = store.replay("r1", upto_seq=0)

    assert len(full) == 2
    assert full[1]["hypothesis"] == "H"
    assert len(partial) == 1
    assert partial[0]["hypothesis"] == ""


def test_list_runs_is_newest_first(tmp_path):
    store = RunStore(tmp_path / "h.db")
    for i in range(3):
        store.create_run(new_run_state(f"r{i}", f"Q{i}?"))

    rows = store.list_runs(limit=2)

    assert len(rows) == 2
    assert {"run_id", "question", "status"} <= set(rows[0])


def test_streaming_never_raises_when_redis_is_down(monkeypatch):
    """A debate must not die because a cache blinked."""
    def boom(*a, **k):
        raise redis.RedisError("down")

    monkeypatch.setattr(stream, "_client", boom)

    stream.publish_turn("r1", DebateTurn(agent="A", content="c"))  # must not raise
    stream.publish_done("r1")
    assert stream.past_turns("r1") == []
    assert list(stream.tail("r1")) == []
