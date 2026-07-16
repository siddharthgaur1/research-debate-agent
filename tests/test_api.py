"""The FastAPI surface, with the graph stubbed out."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import app as api_module
from src.persistence.store import RunStore
from src.state.schema import RunStatus, Verdict, new_run_state


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A client whose store is per-test and whose runs never actually execute."""
    store = RunStore(tmp_path / "api.db")
    monkeypatch.setattr(api_module, "store", lambda: store)
    monkeypatch.setattr(api_module, "execute_run", lambda state, s=None: state)
    return TestClient(api_module.app), store


def test_health_reports_ok(client):
    http, _ = client
    assert http.get("/health").json() == {"status": "ok"}


def test_post_research_returns_a_run_id(client):
    http, store = client

    resp = http.post("/research", json={"question": "Is remote work more productive?"})

    assert resp.status_code == 202
    run_id = resp.json()["run_id"]
    assert store.get_run(run_id) is not None


def test_post_research_rejects_a_trivial_question(client):
    http, _ = client
    assert http.post("/research", json={"question": "hi"}).status_code == 422


def test_get_run_shapes_the_debate_for_the_client(client, base_state, bias_report):
    http, store = client
    from src.tools.confidence import score_claims

    base_state["claims"] = score_claims(
        [("Output rose.", ["S1", "S2"], [], False)], base_state["sources"]
    )
    base_state["verdict"] = Verdict(statement="Yes.", confidence=0.7, uncertainty_mode=False)
    base_state["bias_report"] = bias_report
    base_state["status"] = RunStatus.COMPLETED
    store.create_run(base_state)

    body = http.get(f"/research/{base_state['run_id']}").json()

    assert body["status"] == "completed"
    assert body["verdict"]["statement"] == "Yes."
    assert len(body["claims"]) == 1
    assert body["sources"] and "text" not in body["sources"][0]  # no raw page dumps
    assert body["hypothesis"]


def test_get_unknown_run_is_404(client):
    http, _ = client
    assert http.get("/research/nope").status_code == 404


def test_history_exposes_the_audit_trail(client):
    http, store = client
    state = new_run_state("r1", "Is remote work more productive?")
    store.create_run(state)
    state["status"] = RunStatus.COMPLETED
    store.record_transition(state, "report")

    body = http.get("/research/r1/history").json()

    assert [row["stage"] for row in body] == ["created", "report"]


def test_report_pdf_is_404_before_it_exists(client):
    http, store = client
    store.create_run(new_run_state("r1", "Is remote work more productive?"))
    assert http.get("/research/r1/report.pdf").status_code == 404


def test_report_pdf_downloads_when_ready(client, base_state, tmp_path):
    http, store = client
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    base_state["report_path"] = str(pdf)
    base_state["status"] = RunStatus.COMPLETED
    store.create_run(base_state)

    resp = http.get(f"/research/{base_state['run_id']}/report.pdf")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"


def test_stream_replays_recorded_turns(client, base_state, monkeypatch):
    """A client connecting late still sees how the argument started."""
    http, store = client
    base_state["status"] = RunStatus.COMPLETED
    store.create_run(base_state)
    monkeypatch.setattr(
        "src.persistence.stream.past_turns",
        lambda run_id: [{"agent": "Advocate", "content": "Point one [S1]."}],
    )

    with http.stream("GET", f"/research/{base_state['run_id']}/stream") as resp:
        body = "".join(chunk for chunk in resp.iter_text())

    assert resp.status_code == 200
    assert "Advocate" in body
    assert "event: done" in body
