"""Each agent against a fixed fake LLM response and fixture sources.

Every agent reaches the model through `Budget`, so patching its two methods is
enough to make the whole suite offline — no per-agent mocking ceremony.
"""

from __future__ import annotations

from src.agents.advocate import advocate_agent
from src.agents.arbitrator import ArbitratorOutput, arbitrator_agent
from src.agents.bias import bias_agent
from src.agents.critic import critic_agent
from src.agents.gather import gather_agent
from src.agents.report import report_agent
from src.agents.researcher import researcher_agent
from src.agents.supervisor import SupervisorPlan, supervisor_agent
from src.state.schema import (
    BiasReport,
    RunStatus,
    Source,
    SourceStub,
    Stance,
    Subtask,
    Verdict,
)
# ----------------------------------------------------------------------- supervisor


def test_supervisor_sets_a_hypothesis_and_subtasks(base_state, fake_llm):
    plan = SupervisorPlan(
        hypothesis="Remote work increases measured productivity.",
        subtasks=[
            {"question": "Does output rise?", "rationale": "Core."},
            {"question": "Where does it fall?", "rationale": "Refutation."},
        ],
    )
    fake_llm(structured={SupervisorPlan: plan})

    update = supervisor_agent(base_state)

    assert update["hypothesis"] == "Remote work increases measured productivity."
    assert [t.id for t in update["subtasks"]] == ["T1", "T2"]
    assert update["status"] == RunStatus.RESEARCHING
    assert len(update["debate_transcript"]) == 1
    assert update["token_usage"]


def test_supervisor_caps_subtasks_at_five(base_state, fake_llm):
    plan = SupervisorPlan(
        hypothesis="H",
        subtasks=[{"question": f"q{i}", "rationale": "r"} for i in range(9)],
    )
    fake_llm(structured={SupervisorPlan: plan})
    assert len(supervisor_agent(base_state)["subtasks"]) == 5


# ----------------------------------------------------------------------- researcher


def test_researcher_gathers_scores_and_reports(monkeypatch, fake_llm):
    fake_llm(text="Output rose in two studies.")
    monkeypatch.setattr(
        "src.agents.researcher.search_tool.search",
        lambda q, max_results=5: [
            SourceStub(url="https://bls.gov/a", title="A", snippet="s", published="2024-01-01")
        ],
    )
    monkeypatch.setattr("src.agents.researcher.fetch_text", lambda url: "page text")

    update = researcher_agent(
        {
            "run_id": "r1",
            "question": "Q",
            "hypothesis": "H",
            "subtask": Subtask(id="T1", question="q", rationale="r"),
            "task_number": 1,
            "token_usage": [],
        }
    )

    assert update["search_count"] == 1
    assert len(update["raw_sources"]) == 1
    source = update["raw_sources"][0]
    assert source.text == "page text"
    assert source.credibility_score > 0.5  # bls.gov, recent
    assert source.credibility_reasoning
    assert update["debate_transcript"][0].agent == "Researcher · T1"


def test_researcher_survives_zero_results(monkeypatch, fake_llm):
    fake_llm()
    monkeypatch.setattr("src.agents.researcher.search_tool.search", lambda q, max_results=5: [])

    update = researcher_agent(
        {
            "run_id": "r1",
            "question": "Q",
            "hypothesis": "H",
            "subtask": Subtask(id="T1", question="q", rationale="r"),
            "task_number": 1,
            "token_usage": [],
        }
    )

    assert update["raw_sources"] == []
    assert "No usable sources" in update["debate_transcript"][0].content


# --------------------------------------------------------------------------- gather


def test_gather_dedups_and_renumbers(base_state, monkeypatch):
    """Survivors get dense canonical ids — the ids the debate cites."""
    monkeypatch.setattr(
        "src.agents.gather.dedup_sources",
        lambda run_id, sources: list(sources)[:2],
    )
    base_state["raw_sources"] = base_state["sources"]

    update = gather_agent(base_state)

    assert [s.id for s in update["sources"]] == ["S1", "S2"]
    assert update["status"] == RunStatus.DEBATING


# ------------------------------------------------------------------ advocate/critic


def test_advocate_argues_for_and_records_citations(base_state, fake_llm):
    fake_llm(text="1. Output rose [S1]. 2. Confirmed independently [S2].")

    update = advocate_agent(base_state)

    assert update["advocate_case"].startswith("1.")
    turn = update["debate_transcript"][0]
    assert turn.stance == Stance.FOR
    assert turn.cited_source_ids == ["S1", "S2"]


def test_critic_argues_against_and_advances_the_round(base_state, fake_llm):
    fake_llm(text="1. Counterevidence [S3]. Concessions: S1 holds.")
    base_state["advocate_case"] = "Output rose [S1]."

    update = critic_agent(base_state)

    turn = update["debate_transcript"][0]
    assert turn.stance == Stance.AGAINST
    assert turn.cited_source_ids == ["S3"]
    assert update["debate_round"] == 1


def test_hallucinated_citations_never_reach_the_transcript(base_state, fake_llm):
    """A model citing [S99] must not put a dead id in the evidence trail."""
    fake_llm(text="Output rose [S1] and definitely [S99] and also [S404].")

    update = advocate_agent(base_state)

    assert update["debate_transcript"][0].cited_source_ids == ["S1"]


# ----------------------------------------------------------------------------- bias


def test_bias_agent_audits_the_pool(base_state, fake_llm, bias_report):
    fake_llm(structured={BiasReport: bias_report})

    update = bias_agent(base_state)

    assert update["bias_report"] is bias_report
    assert update["status"] == RunStatus.ARBITRATING
    assert update["debate_transcript"][0].agent == "Bias Checker"


def test_pool_stats_are_computed_not_guessed(base_state):
    from src.agents.bias import _pool_stats

    stats = _pool_stats(base_state["sources"])

    assert "3 distinct sources across 3 domains" in stats
    assert "Most common domain" in stats


# ----------------------------------------------------------------------- arbitrator


def test_arbitrator_computes_confidence_from_evidence(base_state, fake_llm, bias_report):
    out = ArbitratorOutput(
        verdict_statement="Remote work modestly raises output.",
        reasoning="Two independent credible sources agree.",
        claims=[
            {
                "text": "Output rose 4%.",
                "supporting_source_ids": ["S1", "S2"],
                "opposing_source_ids": [],
                "critic_landed_hit": False,
            }
        ],
        strongest_for=["Two independent studies agree."],
        strongest_against=["One anecdotal account disagrees."],
    )
    fake_llm(structured={ArbitratorOutput: out})
    base_state["bias_report"] = bias_report

    update = arbitrator_agent(base_state)

    claim = update["claims"][0]
    assert claim.id == "C1"
    assert claim.confidence > 0.5
    assert claim.rationale  # the number arrives with its reasoning
    assert update["verdict"].uncertainty_mode is False
    assert update["status"] == RunStatus.REPORTING


def test_arbitrator_enters_uncertainty_mode_on_split_evidence(
    base_state, fake_llm, conflicting_sources, bias_report
):
    base_state["sources"] = conflicting_sources
    base_state["bias_report"] = bias_report
    out = ArbitratorOutput(
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
    fake_llm(structured={ArbitratorOutput: out})

    update = arbitrator_agent(base_state)

    assert update["verdict"].uncertainty_mode is True
    assert len(update["verdict"].contested_points) == 2


def test_arbitrator_cannot_fabricate_confidence(base_state, fake_llm, bias_report):
    """Confidence is computed, so a claim citing nothing cannot score high."""
    out = ArbitratorOutput(
        verdict_statement="Certainly true.",
        reasoning="Trust me.",
        claims=[
            {
                "text": "Unsupported assertion.",
                "supporting_source_ids": [],
                "opposing_source_ids": [],
                "critic_landed_hit": False,
            }
        ],
    )
    fake_llm(structured={ArbitratorOutput: out})
    base_state["bias_report"] = bias_report

    update = arbitrator_agent(base_state)

    assert update["claims"][0].confidence < 0.35


# --------------------------------------------------------------------------- report


def _verdict() -> Verdict:
    return Verdict(
        statement="Remote work modestly raises output.",
        confidence=0.7,
        uncertainty_mode=False,
        reasoning="Because.",
        strongest_for=["a"],
        strongest_against=["b"],
    )


def test_report_exports_a_pdf(base_state, bias_report):
    from src.tools.confidence import score_claims

    base_state["claims"] = score_claims(
        [("Output rose.", ["S1", "S2"], [], False)], base_state["sources"]
    )
    base_state["verdict"] = _verdict()
    base_state["bias_report"] = bias_report

    update = report_agent(base_state)

    assert update["status"] == RunStatus.COMPLETED
    from pathlib import Path

    path = Path(update["report_path"])
    assert path.exists()
    assert path.read_bytes().startswith(b"%PDF")


def test_report_drops_claims_that_cite_nothing_real(base_state, bias_report):
    """The citation guarantee, enforced rather than hoped for."""
    from src.state.schema import Claim

    base_state["claims"] = [
        Claim(id="C1", text="Grounded.", supporting_source_ids=["S1"], confidence=0.7),
        Claim(id="C2", text="Invented.", supporting_source_ids=["S99"], confidence=0.9),
        Claim(id="C3", text="Cites nothing.", supporting_source_ids=[], confidence=0.9),
    ]
    base_state["verdict"] = _verdict()
    base_state["bias_report"] = bias_report

    update = report_agent(base_state)

    assert [c.id for c in update["claims"]] == ["C1"]
    known = {s.id for s in base_state["sources"]}
    for claim in update["claims"]:
        assert set(claim.supporting_source_ids) <= known


def test_report_recomputes_the_verdict_after_dropping_claims(base_state, bias_report):
    from src.state.schema import Claim

    base_state["claims"] = [
        Claim(id="C1", text="Grounded.", supporting_source_ids=["S1"], confidence=0.2),
        Claim(id="C2", text="Invented.", supporting_source_ids=["S99"], confidence=0.9),
    ]
    base_state["verdict"] = _verdict()
    base_state["bias_report"] = bias_report

    update = report_agent(base_state)

    # 0.7 was the mean including the dropped claim; it must not survive unchanged.
    assert update["verdict"].confidence == 0.2
