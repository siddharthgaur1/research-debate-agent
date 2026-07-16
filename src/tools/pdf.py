"""Render a finished run to PDF.

Platypus handles flowing, pagination and table breaks; the only thing worth
hand-writing here is what goes on the page and in what order. The layout mirrors
the report's priorities: the verdict and its honesty about uncertainty first, then
per-claim confidence, then the debate, then the full citation trail — so a reader
who stops after page one has still seen the caveats.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ..state.schema import RunState

_CONTESTED = colors.HexColor("#B45309")
_MUTED = colors.HexColor("#6B7280")


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontSize=20, leading=24, alignment=TA_LEFT
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontSize=13, spaceBefore=14, spaceAfter=6
        ),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontSize=9.5, leading=13),
        "muted": ParagraphStyle(
            "muted", parent=base["BodyText"], fontSize=8, textColor=_MUTED, leading=11
        ),
        "cell": ParagraphStyle("cell", parent=base["BodyText"], fontSize=8, leading=10),
    }


def _p(text: str, style: ParagraphStyle) -> Paragraph:
    """Escape user/model text before it reaches Platypus' mini-HTML parser."""
    return Paragraph(escape(text or ""), style)


def _bullets(items: list[str], style: ParagraphStyle) -> ListFlowable | Paragraph:
    if not items:
        return _p("None recorded.", style)
    return ListFlowable(
        [ListItem(_p(i, style), leftIndent=12) for i in items],
        bulletType="bullet",
        start="•",
        leftIndent=12,
    )


def _claims_table(state: RunState, styles) -> Table:
    rows = [
        [
            _p("Claim", styles["cell"]),
            _p("Conf.", styles["cell"]),
            _p("Status", styles["cell"]),
            _p("Sources", styles["cell"]),
        ]
    ]
    for claim in state.get("claims", []):
        rows.append(
            [
                _p(claim.text, styles["cell"]),
                _p(f"{claim.confidence:.2f}", styles["cell"]),
                _p("CONTESTED" if claim.contested else "settled", styles["cell"]),
                _p(", ".join(claim.supporting_source_ids) or "—", styles["cell"]),
            ]
        )

    table = Table(rows, colWidths=[3.4 * inch, 0.6 * inch, 0.9 * inch, 1.6 * inch])
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D1D5DB")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]
    for i, claim in enumerate(state.get("claims", []), start=1):
        if claim.contested:
            style.append(("TEXTCOLOR", (2, i), (2, i), _CONTESTED))
    table.setStyle(TableStyle(style))
    return table


def _citation_rows(state: RunState, styles) -> Table:
    rows = [
        [
            _p("Id", styles["cell"]),
            _p("Source", styles["cell"]),
            _p("Cred.", styles["cell"]),
            _p("Why", styles["cell"]),
        ]
    ]
    for source in state.get("sources", []):
        title = f"{source.title}\n{source.url}"
        if source.merged_from:
            title += f"\n(also republished at {len(source.merged_from)} other url(s))"
        rows.append(
            [
                _p(source.id, styles["cell"]),
                _p(title, styles["cell"]),
                _p(f"{source.credibility_score:.2f}", styles["cell"]),
                _p(source.credibility_reasoning, styles["cell"]),
            ]
        )
    table = Table(rows, colWidths=[0.4 * inch, 2.9 * inch, 0.5 * inch, 2.7 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D1D5DB")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    return table


def render_report(state: RunState, path: Path) -> Path:
    """Write the run's report to `path` and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    verdict = state.get("verdict")
    story: list = []

    story.append(_p(state.get("question", "Research report"), styles["title"]))
    story.append(
        _p(
            f"Run {state.get('run_id', '?')} · "
            f"{len(state.get('sources', []))} sources · "
            f"{len(state.get('claims', []))} claims",
            styles["muted"],
        )
    )
    story.append(HRFlowable(width="100%", color=colors.HexColor("#D1D5DB")))

    if verdict and verdict.uncertainty_mode:
        story.append(_p("UNCERTAINTY MODE", styles["h2"]))
        story.append(
            _p(
                "The evidence is genuinely split. Both sides are presented below "
                "with their strongest support rather than resolved into a verdict "
                "the evidence does not justify.",
                styles["body"],
            )
        )

    story.append(_p("Verdict", styles["h2"]))
    if verdict:
        story.append(_p(verdict.statement, styles["body"]))
        story.append(Spacer(1, 6))
        story.append(_p(f"Overall confidence: {verdict.confidence:.2f}", styles["muted"]))
        story.append(Spacer(1, 6))
        story.append(_p(verdict.reasoning, styles["body"]))
    else:
        story.append(_p("No verdict was reached.", styles["body"]))

    story.append(_p("Key claims and confidence", styles["h2"]))
    story.append(_claims_table(state, styles))

    if verdict:
        story.append(_p("Strongest points FOR", styles["h2"]))
        story.append(_bullets(verdict.strongest_for, styles["body"]))
        story.append(_p("Strongest points AGAINST", styles["h2"]))
        story.append(_bullets(verdict.strongest_against, styles["body"]))
        story.append(_p("Contested points", styles["h2"]))
        story.append(_bullets(verdict.contested_points, styles["body"]))

    bias = state.get("bias_report")
    if bias:
        story.append(_p("Source-pool bias audit", styles["h2"]))
        story.append(_p(bias.summary, styles["body"]))
        story.append(Spacer(1, 4))
        story.append(
            _bullets(
                [
                    f"Outlet concentration: {bias.outlet_concentration}",
                    f"Recency skew: {bias.recency_skew}",
                    f"Funding flags: {', '.join(bias.funding_flags) or 'none'}",
                    f"Missing perspectives: {', '.join(bias.missing_perspectives) or 'none'}",
                ],
                styles["body"],
            )
        )

    story.append(PageBreak())
    story.append(_p("Citation trail", styles["h2"]))
    story.append(_citation_rows(state, styles))

    SimpleDocTemplate(
        str(path),
        pagesize=LETTER,
        title=state.get("question", "Research report"),
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
    ).build(story)
    return path
