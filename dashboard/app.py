"""Streamlit dashboard: watch the agents argue.

The live debate panel is the point of this app. A finished report is just a
document — what makes the reasoning trustworthy is seeing the Advocate make a
case, the Critic land a hit, and the Bias Checker flag the sourcing, in order, as
it happens. So turns render as they arrive off the SSE stream rather than after
the run completes.
"""

from __future__ import annotations

import json
import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
TIMEOUT = 10

st.set_page_config(page_title="Research + Debate Agent", page_icon="⚖️", layout="wide")

#: Each role gets a colour and an icon so the argument is readable at a glance.
ROLES: dict[str, tuple[str, str]] = {
    "supervisor": ("#6366F1", "🧭"),
    "gather": ("#0EA5E9", "📚"),
    "Advocate": ("#16A34A", "✅"),
    "Critic": ("#DC2626", "⚔️"),
    "Bias Checker": ("#D97706", "🔍"),
    "Arbitrator": ("#7C3AED", "⚖️"),
    "Report": ("#334155", "📄"),
}


def role_style(agent: str) -> tuple[str, str]:
    if agent.startswith("Researcher"):
        return "#0EA5E9", "🔬"
    return ROLES.get(agent, ("#64748B", "•"))


def render_turn(container, turn: dict) -> None:
    """Render one debate turn as a colour-coded card."""
    agent = turn.get("agent", "?")
    colour, icon = role_style(agent)
    round_number = turn.get("round", 0)
    badge = f" · round {round_number + 1}" if agent in ("Advocate", "Critic") else ""
    cites = turn.get("cited_source_ids") or []

    with container:
        st.markdown(
            f"<div style='border-left:4px solid {colour};padding:.5rem .9rem;"
            f"margin:.45rem 0;background:rgba(148,163,184,.08);border-radius:4px'>"
            f"<div style='color:{colour};font-weight:700;font-size:.85rem'>"
            f"{icon} {agent}{badge}</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown(turn.get("content", ""))
        if cites:
            st.caption(f"Cites: {', '.join(cites)}")


def api_healthy() -> bool:
    try:
        return requests.get(f"{API_BASE_URL}/health", timeout=3).status_code == 200
    except requests.RequestException:
        return False


def stream_debate(run_id: str, container) -> None:
    """Tail the SSE endpoint, rendering each turn as it arrives."""
    try:
        with requests.get(
            f"{API_BASE_URL}/research/{run_id}/stream", stream=True, timeout=900
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line or line.startswith(":"):
                    continue
                if line.startswith("event: done"):
                    return
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload.strip() in ("", "{}"):
                        continue
                    render_turn(container, json.loads(payload))
    except requests.RequestException as exc:
        st.warning(f"Live stream dropped ({exc}). The run continues; reload to catch up.")


def confidence_badge(confidence: float, contested: bool) -> str:
    if contested:
        return f"⚖️ CONTESTED · {confidence:.2f}"
    if confidence >= 0.7:
        return f"🟢 {confidence:.2f}"
    if confidence >= 0.45:
        return f"🟡 {confidence:.2f}"
    return f"🔴 {confidence:.2f}"


def render_report(run_id: str) -> None:
    """The final report: verdict, per-claim confidence, and the citation trail."""
    try:
        data = requests.get(f"{API_BASE_URL}/research/{run_id}", timeout=TIMEOUT).json()
    except requests.RequestException as exc:
        st.error(f"Could not load the report: {exc}")
        return

    if data.get("status") == "failed":
        st.error(f"Run failed: {data.get('error')}")
        return

    verdict = data.get("verdict")
    sources = {s["id"]: s for s in data.get("sources", [])}

    if verdict and verdict.get("uncertainty_mode"):
        st.warning(
            "**Uncertainty mode** — the evidence is genuinely split. Both sides are "
            "presented below rather than resolved into a verdict the evidence does "
            "not justify."
        )

    st.subheader("Verdict")
    if verdict:
        st.markdown(f"### {verdict['statement']}")
        st.caption(f"Overall confidence: {verdict['confidence']:.2f}")
        st.markdown(verdict.get("reasoning", ""))
    else:
        st.info("No verdict yet.")

    st.subheader("Key claims")
    for claim in data.get("claims", []):
        with st.container(border=True):
            head, badge = st.columns([5, 1])
            head.markdown(f"**{claim['text']}**")
            badge.markdown(confidence_badge(claim["confidence"], claim["contested"]))
            st.caption(claim.get("rationale", ""))

            cited = claim["supporting_source_ids"] + claim["opposing_source_ids"]
            if cited:
                with st.expander(f"Evidence ({len(cited)} source(s))"):
                    for sid in claim["supporting_source_ids"]:
                        _citation(sources, sid, "Supports")
                    for sid in claim["opposing_source_ids"]:
                        _citation(sources, sid, "Opposes")

    if verdict:
        left, right = st.columns(2)
        with left:
            st.subheader("Strongest FOR")
            for point in verdict.get("strongest_for", []) or ["—"]:
                st.markdown(f"- {point}")
        with right:
            st.subheader("Strongest AGAINST")
            for point in verdict.get("strongest_against", []) or ["—"]:
                st.markdown(f"- {point}")

        contested = verdict.get("contested_points") or []
        if contested:
            st.subheader("Contested points")
            for point in contested:
                st.markdown(f"- ⚖️ {point}")

    bias = data.get("bias_report")
    if bias:
        with st.expander("Source-pool bias audit"):
            st.markdown(bias.get("summary", ""))
            st.markdown(f"- **Outlet concentration:** {bias.get('outlet_concentration')}")
            st.markdown(f"- **Recency skew:** {bias.get('recency_skew')}")
            st.markdown(f"- **Funding flags:** {', '.join(bias.get('funding_flags') or []) or 'none'}")
            st.markdown(
                f"- **Missing perspectives:** "
                f"{', '.join(bias.get('missing_perspectives') or []) or 'none'}"
            )

    with st.expander(f"Citation trail ({len(sources)} sources)"):
        raw = data.get("raw_source_count", 0)
        if raw > len(sources):
            st.caption(f"{raw} results found → {len(sources)} distinct after dedup.")
        for source in sources.values():
            st.markdown(
                f"**[{source['id']}]** [{source['title']}]({source['url']}) — "
                f"`{source['domain']}` · credibility **{source['credibility_score']:.2f}**"
            )
            st.caption(source.get("credibility_reasoning", ""))
            if source.get("merged_from"):
                st.caption(f"Also republished at {len(source['merged_from'])} other url(s).")

    if data.get("report_available"):
        try:
            pdf = requests.get(
                f"{API_BASE_URL}/research/{run_id}/report.pdf", timeout=30
            ).content
            st.download_button(
                "⬇️ Download PDF report",
                pdf,
                file_name=f"report-{run_id}.pdf",
                mime="application/pdf",
            )
        except requests.RequestException:
            st.caption("PDF not retrievable right now.")

    st.caption(f"Run {run_id} · {data.get('search_count', 0)} searches · ${data.get('cost_usd', 0):.4f}")


def _citation(sources: dict, sid: str, relation: str) -> None:
    source = sources.get(sid)
    if not source:
        return
    st.markdown(
        f"- *{relation}* **[{sid}]** [{source['title']}]({source['url']}) "
        f"— credibility {source['credibility_score']:.2f}"
    )


# ------------------------------------------------------------------------------ UI

st.title("⚖️ Research + Debate Agent")
st.caption("Agents research a question, argue both sides, audit the sources, and arbitrate.")

with st.sidebar:
    st.header("Connection")
    if api_healthy():
        st.success(f"API reachable\n\n`{API_BASE_URL}`")
    else:
        st.error(f"API unreachable\n\n`{API_BASE_URL}`")
        st.caption("Start it with `docker-compose up`, or set API_BASE_URL.")

    st.header("Recent runs")
    try:
        for row in requests.get(f"{API_BASE_URL}/research?limit=8", timeout=TIMEOUT).json():
            if st.button(
                f"{row['question'][:40]}… · {row['status']}",
                key=row["run_id"],
                use_container_width=True,
            ):
                st.session_state.run_id = row["run_id"]
                st.session_state.live = False
                st.rerun()
    except requests.RequestException:
        st.caption("No run history available.")

question = st.text_input(
    "Research question",
    placeholder="Is remote work more productive than office work?",
)

if st.button("Run debate", type="primary", disabled=not question.strip()):
    try:
        resp = requests.post(
            f"{API_BASE_URL}/research", json={"question": question}, timeout=TIMEOUT
        )
        resp.raise_for_status()
        st.session_state.run_id = resp.json()["run_id"]
        st.session_state.live = True
    except requests.RequestException as exc:
        st.error(f"Could not start the run: {exc}")

run_id = st.session_state.get("run_id")
if run_id:
    debate_tab, report_tab = st.tabs(["🎙️ Live debate", "📄 Report"])

    with debate_tab:
        st.caption(f"Run `{run_id}`")
        turns = st.container()
        if st.session_state.get("live"):
            with st.spinner("Agents are working…"):
                stream_debate(run_id, turns)
            st.session_state.live = False
            st.success("Debate complete — see the Report tab.")
        else:
            try:
                data = requests.get(f"{API_BASE_URL}/research/{run_id}", timeout=TIMEOUT).json()
                for turn in data.get("debate_transcript", []):
                    render_turn(turns, turn)
            except requests.RequestException as exc:
                st.error(f"Could not load the transcript: {exc}")

    with report_tab:
        render_report(run_id)
else:
    st.info("Ask a contestable question above to start a debate.")
