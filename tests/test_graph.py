"""The graph end to end, fully mocked.

The point of these tests is the wiring the unit tests can't see: that the fan-out
really runs one researcher per subtask, that the debate loop honours its cap, that
a failing branch doesn't sink the run, and that a full pass produces a verdict
whose citations resolve to real sources.
"""

from __future__ import annotations

import pytest

from src.agents.arbitrator import ArbitratorOutput
from src.agents.supervisor import SupervisorPlan
from src.graph.build import build_graph
from src.persistence.store import RunStore
from src.state.schema import BiasReport, RunStatus, SourceStub, new_run_state


@pytest.fixture
def wired(monkeypatch, fake_llm, bias_report):
    """A graph whose every external call is canned."""
    plan = SupervisorPlan(
        hypothesis="Remote work increases measured productivity.",
        subtasks=[
            {"question": "Does output rise?", "rationale": "Core."},
            {"question": "Where does it fall?", "rationale": "Refutation."},
        ],
    )
    arb = ArbitratorOutput(
        verdict_statement="Remote work modestly raises measured output.",
        reasoning="Two independent credible sources agree; one anecdote dissents.",
        claims=[
            {
                "text": "Measured output rose about 4%.",
                "supporting_source_ids": ["S1", "S2"],
                "opposing_source_ids": ["S3"],
                "critic_landed_hit": False,
            }
        ],
        strongest_for=["Two independent studies agree."],
        strongest_against=["One stale anecdote dissents."],
    )
    fake_llm(
        text="Point one [S1]. Point two [S2]. Counterpoint [S3].",
        structured={
            SupervisorPlan: plan,
            BiasReport: bias_report,
            ArbitratorOutput: arb,
        },
    )

    # One distinct hit per subtask; dedup is exercised in its own test.
    hits = {
        "Does output rise?": SourceStub(
            url="https://bls.gov/a", title="Output rose", snippet="rose 4%", published="2024-06-01"
        ),
        "Where does it fall?": SourceStub(
            url="https://x.substack.com/p", title="It collapsed", snippet="fell", published="2018-01-01"
        ),
    }
    monkeypatch.setattr(
        "src.agents.researcher.search_tool.search",
        lambda q, max_results=5: [hits[q]],
    )
    monkeypatch.setattr("src.agents.researcher.fetch_text", lambda url: f"text of {url}")
    monkeypatch.setattr(
        "src.agents.gather.dedup_sources", lambda run_id, sources: list(sources)
    )
    return build_graph()


def _run(graph, question="Is remote work more productive than office work?"):
    return graph.invoke(
        new_run_state("run-test", question),
        {"configurable": {"thread_id": "run-test"}, "recursion_limit": 50},
    )


def test_full_run_produces_a_verdict_with_citations(wired):
    final = _run(wired)

    assert final["status"] == RunStatus.COMPLETED
    verdict = final["verdict"]
    assert verdict is not None
    assert verdict.statement
    assert final["claims"]

    # Every claim resolves to a real source — the headline guarantee.
    known = {s.id for s in final["sources"]}
    assert known
    for claim in final["claims"]:
        cited = set(claim.supporting_source_ids) | set(claim.opposing_source_ids)
        assert cited, f"claim {claim.id} cites nothing"
        assert cited <= known, f"claim {claim.id} cites a source that does not exist"


def test_fan_out_runs_one_researcher_per_subtask(wired):
    final = _run(wired)

    assert final["search_count"] == 2
    assert len(final["raw_sources"]) == 2
    researcher_turns = [
        t for t in final["debate_transcript"] if t.agent.startswith("Researcher")
    ]
    assert len(researcher_turns) == 2


def test_transcript_records_the_debate_in_order(wired):
    final = _run(wired)
    agents = [t.agent for t in final["debate_transcript"]]

    assert agents[0] == "supervisor"
    assert agents.index("Advocate") < agents.index("Bias Checker")
    assert agents.index("Bias Checker") < agents.index("Arbitrator")
    assert agents[-1] == "Report"
    assert {"Advocate", "Critic", "Bias Checker", "Arbitrator"} <= set(agents)


def test_debate_loop_respects_the_round_cap(wired, monkeypatch):
    monkeypatch.setenv("MAX_DEBATE_ROUNDS", "3")
    from src.config import get_settings

    get_settings.cache_clear()

    final = _run(wired)

    assert final["debate_round"] == 3
    assert sum(1 for t in final["debate_transcript"] if t.agent == "Critic") == 3
    assert sum(1 for t in final["debate_transcript"] if t.agent == "Advocate") == 3


def test_a_failing_researcher_does_not_sink_the_run(wired, monkeypatch):
    """One dead angle costs that angle, not the debate."""
    def half_broken(q, max_results=5):
        if "fall" in q:
            raise RuntimeError("search backend exploded")
        return [SourceStub(url="https://bls.gov/a", title="Output rose", snippet="rose 4%")]

    monkeypatch.setattr("src.agents.researcher.search_tool.search", half_broken)

    final = _run(wired)

    assert final["status"] == RunStatus.COMPLETED
    assert len(final["raw_sources"]) == 1
    assert any(m.level == "warn" for m in final["messages"])


def test_a_failing_supervisor_fails_the_run_cleanly(monkeypatch, fake_llm):
    fake_llm()  # no canned SupervisorPlan -> the node raises

    final = _run(build_graph())

    assert final["status"] == RunStatus.FAILED
    assert final["error"]
    assert any(m.level == "error" for m in final["messages"])


def test_every_transition_is_persisted(wired, tmp_path):
    store = RunStore(tmp_path / "history.db")
    state = new_run_state("run-persist", "Is remote work more productive?")
    store.create_run(state)

    graph = build_graph(store=store)
    graph.invoke(state, {"configurable": {"thread_id": "run-persist"}, "recursion_limit": 50})

    stages = [stage for _, stage, _ in store.history("run-persist")]
    assert stages[0] == "created"
    for expected in ("supervisor", "gather", "advocate", "critic", "bias", "arbitrator", "report"):
        assert expected in stages

    replayed = store.replay("run-persist")
    assert replayed[-1]["status"] == RunStatus.COMPLETED
    assert replayed[-1]["verdict"] is not None


def test_uncertainty_mode_survives_a_full_run(monkeypatch, fake_llm, bias_report):
    """Conflicting fixtures must reach the report as split, not resolved."""
    plan = SupervisorPlan(
        hypothesis="Remote work increases productivity.",
        subtasks=[{"question": "Does output rise?", "rationale": "Core."}],
    )
    arb = ArbitratorOutput(
        verdict_statement="The evidence does not settle this.",
        reasoning="Two equally credible studies disagree.",
        claims=[
            {
                "text": "Remote work raises output.",
                "supporting_source_ids": ["S1"],
                "opposing_source_ids": ["S2"],
                "critic_landed_hit": True,
            },
            {
                "text": "Remote work lowers output.",
                "supporting_source_ids": ["S2"],
                "opposing_source_ids": ["S1"],
                "critic_landed_hit": True,
            },
        ],
    )
    fake_llm(
        text="Case [S1] versus [S2].",
        structured={SupervisorPlan: plan, BiasReport: bias_report, ArbitratorOutput: arb},
    )
    monkeypatch.setattr(
        "src.agents.researcher.search_tool.search",
        lambda q, max_results=5: [
            SourceStub(url="https://nature.com/up", title="Remote work raises productivity",
                       snippet="rose 4%", published="2024-06-01"),
            SourceStub(url="https://science.org/down", title="Remote work lowers productivity",
                       snippet="fell 4%", published="2024-06-01"),
        ],
    )
    monkeypatch.setattr("src.agents.researcher.fetch_text", lambda url: f"text of {url}")
    monkeypatch.setattr("src.agents.gather.dedup_sources", lambda run_id, sources: list(sources))

    final = _run(build_graph())

    assert final["status"] == RunStatus.COMPLETED
    assert final["verdict"].uncertainty_mode is True
    assert len(final["verdict"].contested_points) == 2
    assert all(c.contested for c in final["claims"])
