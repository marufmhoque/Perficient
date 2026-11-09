"""Plotly charts + Graphviz line-of-therapy flowchart.

Colour rules (dataviz reference palette): categorical hues in fixed order, colour follows the
entity (never its rank), at most 8 categories (rare ones fold into "Other"), single-series charts
use one hue, one axis per chart, recessive grid, hover tooltips on every mark.
"""

from __future__ import annotations

from collections import Counter

import plotly.graph_objects as go

from ..schemas import FailureRow, Module2, SponsorRank, TrialRecord

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MUTED = "#898781"
GRID = "rgba(137,135,129,0.25)"

# Colour follows the comparator class (entity), fixed regardless of which classes are present.
COMPARATOR_ORDER = [
    "Active – current SoC",
    "Active – other",
    "Active (unclassified)",
    "Placebo + background therapy",
    "Placebo",
    "Dose-comparison (no control)",
    "Single-arm (no comparator)",
    "Other",
]
COMPARATOR_COLORS = dict(zip(COMPARATOR_ORDER, SERIES))
_FOLD_TO_OTHER = {"Sham", "No treatment", "Unclear"}

PHASE_ORDER = ["Early Ph1", "Ph1", "Ph1/Ph2", "Ph2", "Ph2/Ph3", "Ph3", "Ph4", "N/A"]


def _layout(fig: go.Figure, title: str, height: int = 360, legend: bool = True) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 15}},
        height=height,
        margin={"l": 8, "r": 36, "t": 48, "b": 8},
        bargap=0.35,
        barcornerradius=4,
        showlegend=legend,
        # legend below the plot: wrapping entries can never collide with the title
        legend={"orientation": "h", "yanchor": "top", "y": -0.22, "xanchor": "left", "x": 0, "title": None, "traceorder": "normal"},
        hoverlabel={"namelength": -1},
        font={"family": 'system-ui, -apple-system, "Segoe UI", sans-serif'},
    )
    fig.update_xaxes(showgrid=False, linecolor=MUTED, tickfont={"color": MUTED})
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont={"color": MUTED})
    return fig


def comparator_mix_by_phase(trials: list[TrialRecord]) -> go.Figure:
    counts: Counter[tuple[str, str]] = Counter()
    for t in trials:
        cls = t.comparator_class
        counts[(t.phase_label, "Other" if cls in _FOLD_TO_OTHER else cls)] += 1
    phases = [p for p in PHASE_ORDER if any(k[0] == p for k in counts)] + sorted({k[0] for k in counts} - set(PHASE_ORDER))
    fig = go.Figure()
    for cls in COMPARATOR_ORDER:
        ys = [counts.get((p, cls), 0) for p in phases]
        if not any(ys):
            continue
        fig.add_bar(
            name=cls, x=phases, y=ys, marker={"color": COMPARATOR_COLORS[cls], "line": {"width": 2, "color": "rgba(0,0,0,0)"}},
            hovertemplate=f"<b>{cls}</b><br>%{{x}}: %{{y}} trials<extra></extra>",
        )
    fig.update_layout(barmode="stack")
    tallest = max((sum(counts.get((p, c), 0) for c in COMPARATOR_ORDER) for p in phases), default=0)
    fig.update_yaxes(title=None, dtick=1 if tallest <= 8 else None, rangemode="tozero")
    fig.update_xaxes(automargin=True)  # plotly rotates phase labels itself when they don't fit
    n_series = len(fig.data)
    return _layout(fig, "Comparator arm by phase (active trials)", height=380 + 22 * max(0, n_series - 3))


def _hbar(labels: list[str], values: list[int], title: str, unit: str, color: str = SERIES[0]) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=values[::-1], y=labels[::-1], orientation="h", marker={"color": color},
            width=0.6,  # thin marks: thickness stays constant however few categories there are
            text=values[::-1], textposition="outside", textfont={"color": MUTED},
            hovertemplate=f"<b>%{{y}}</b><br>%{{x}} {unit}<extra></extra>", cliponaxis=False,
        )
    )
    fig = _layout(fig, title, height=90 + 30 * max(len(labels), 3), legend=False)
    fig.update_xaxes(showticklabels=False, gridcolor=GRID, showgrid=False, range=[0, max(values or [1]) * 1.15])
    fig.update_yaxes(showgrid=False, tickfont={"color": "#52514e"}, automargin=True)
    return fig


def modality_mix(trials: list[TrialRecord]) -> go.Figure:
    counts = Counter((t.modality or "Unclassified").split(" + ")[0] for t in trials)
    top = counts.most_common(10)
    return _hbar([k for k, _ in top], [v for _, v in top], "Active trials by modality", "trials")


def sponsor_leaderboard(sponsors: list[SponsorRank], top: int = 12) -> go.Figure:
    rows = [s for s in sponsors if s.active_ph2_3 > 0][:top] or sponsors[:top]
    labels = [s.sponsor if len(s.sponsor) <= 42 else s.sponsor[:40] + "…" for s in rows]
    return _hbar(labels, [s.active_ph2_3 for s in rows], "Sponsors by active Phase 2/3 trials", "active Phase 2/3 trials")


def failure_categories(rows: list[FailureRow]) -> go.Figure:
    counts = Counter(r.failure_category for r in rows).most_common()
    return _hbar([k for k, _ in counts], [v for _, v in counts], "Discontinued / failed trials by cause", "trials", color=SERIES[1])


def therapy_flowchart_dot(m2: Module2) -> str:
    """Graphviz DOT for the line-of-therapy algorithm (rendered client-side by st.graphviz_chart)."""

    def q(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')

    lines_order = ["First-line", "Second-line", "Third-line / Refractory"]
    out = [
        "digraph G {",
        '  rankdir=LR; bgcolor="transparent"; nodesep=0.25; ranksep=0.5;',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, color="#c3c2b7", fillcolor="#f0efec", fontcolor="#0b0b0b"];',
        '  edge [color="#898781", arrowsize=0.7, fontname="Helvetica", fontsize=9, fontcolor="#52514e"];',
    ]
    prev_heads: list[str] = []
    for i, line in enumerate(lines_order):
        rows = [r for r in m2.lines_of_therapy if r.line == line]
        if not rows:
            continue
        head = f"L{i}"
        out.append(f'  {head} [label="{q(line)}", fillcolor="#2a78d6", fontcolor="#ffffff", color="#2a78d6"];')
        for j, r in enumerate(rows):
            node = f"L{i}_{j}"
            therapies = "\\n".join(q(t) for t in r.therapies[:5]) + ("\\n…" if len(r.therapies) > 5 else "")
            out.append(f'  {node} [label="{q(r.setting[:60])}\\n—\\n{therapies}"];')
            out.append(f"  {head} -> {node};")
        for p in prev_heads[-1:]:
            out.append(f'  {p} -> {head} [label="inadequate response /\\nintolerance", style=dashed];')
        prev_heads.append(head)
    out.append("}")
    return "\n".join(out)
