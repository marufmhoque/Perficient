"""Streamlit rendering of a Briefing: one tab per module, inline citations, per-module sources."""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from ..citations import cited_keys
from ..factcheck import claim_key
from ..history import report_to_json
from ..schemas import Briefing, CitedStatement, ClaimCheck, Source
from . import charts
from .markdown import briefing_to_markdown

NCT_LINK = st.column_config.LinkColumn("NCT", display_text=r"https://clinicaltrials\.gov/study/(NCT\d+)", width=110)
KIND_LABEL = {"pubmed": "PubMed", "trial": "ClinicalTrials.gov", "sec": "SEC EDGAR"}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "briefing"


def _cite(ids: list[str], smap: dict[str, Source]) -> str:
    links = [f"[\\[{k}\\]]({smap[k].url} \"{smap[k].title[:90].replace(chr(34), '')}\")" for k in ids if k in smap]
    return " ".join(links) if links else ":orange[*(unsourced)*]"


def _md_escape(text: str) -> str:
    return text.replace("$", "\\$")


# Fact-check verdicts: status colour always paired with an icon + label (never colour alone)
VERDICT_ICON = {"Supported": "✅", "Partially supported": "⚠️", "Not supported": "❌", "Cannot verify": "❔"}
VERDICT_BADGE = {
    "Partially supported": ":orange[⚠️ *partially supported*]",
    "Not supported": ":red[❌ *not supported by cited source*]",
    "Cannot verify": ":gray[❔ *could not verify*]",
}
_FC: dict[str, ClaimCheck] = {}  # claim_key -> check, for the report being rendered


def _verdict(obj) -> ClaimCheck | None:
    key = claim_key(obj)
    return _FC.get(key) if key else None


def _badge(obj) -> str:
    chk = _verdict(obj)
    return f" {VERDICT_BADGE[chk.verdict]}" if chk and chk.verdict in VERDICT_BADGE else ""


def _fc_cell(obj) -> str:
    chk = _verdict(obj)
    return f"{VERDICT_ICON[chk.verdict]} {chk.verdict}" if chk else ""


def statements(items: list[CitedStatement], smap: dict[str, Source], empty: str = "No supported statements.") -> None:
    if not items:
        st.caption(empty)
        return
    st.markdown("\n".join(f"- {_md_escape(s.text)} {_cite(s.source_ids, smap)}{_badge(s)}" for s in items))


def fact_check_caption(module) -> None:
    """One-line verdict tally for the claims in a module."""
    if not _FC or module is None:
        return
    from collections import Counter

    keys = {c.key for c in iter_claims_for(module)}
    tally = Counter(_FC[k].verdict for k in keys if k in _FC)
    if tally:
        parts = [f"{VERDICT_ICON[v]} {tally[v]} {v.lower()}" for v in VERDICT_ICON if tally[v]]
        st.caption("Fact check: " + " · ".join(parts) + " — details in the Fact check & Sources tab.")


def iter_claims_for(module):
    from ..factcheck import Claim

    out = []

    def walk(o):
        from pydantic import BaseModel

        if isinstance(o, BaseModel):
            if getattr(type(o), "SKIP_TEXT_SCAN", False):
                return
            k = claim_key(o)
            if k:
                out.append(Claim(key=k, location="", module="", text="", source_ids=[]))
                return
            for name in type(o).model_fields:
                walk(getattr(o, name))
        elif isinstance(o, list):
            for i in o:
                walk(i)

    walk(module)
    return out


def sources_expander(obj, smap: dict[str, Source], label: str = "Sources for this module") -> None:
    keys = [k for k in _ordered_keys(cited_keys(obj)) if k in smap]
    with st.expander(f"{label} ({len(keys)})"):
        if not keys:
            st.caption("No sources cited.")
            return
        st.markdown(
            "\n".join(f"- **[{k}]** [{smap[k].title}]({smap[k].url}) — {smap[k].detail}" for k in keys)
        )


def _ordered_keys(keys: set[str]) -> list[str]:
    def sort_key(k: str) -> tuple:
        return ("PTS".find(k[0]), int(k[1:]) if k[1:].isdigit() else 0)

    return sorted(keys, key=sort_key)


def _ids(ids: list[str]) -> str:
    return ", ".join(ids)


def _df(rows: list[dict], **config) -> None:
    if not rows:
        st.caption("No rows.")
        return
    df = pd.DataFrame(rows)
    if "Fact check" in df.columns:
        if not df["Fact check"].astype(bool).any():
            df = df.drop(columns=["Fact check"])  # report has no fact check: hide the empty column
        else:  # verdict first so it reads as a row status
            df = df[["Fact check"] + [c for c in df.columns if c != "Fact check"]]
    st.dataframe(df, hide_index=True, width="stretch", column_config=config or None)


# ---------------------------------------------------------------------------
# Header & tabs
# ---------------------------------------------------------------------------


def render_briefing(b: Briefing, on_fact_check=None) -> None:
    """Render a report. ``on_fact_check(briefing)`` is called when the user asks to fact-check a
    report that has no fact check yet (older / imported reports)."""
    global _FC
    _FC = b.fact_check.by_key() if b.fact_check else {}
    smap = b.source_map()
    c = b.condition
    st.header(f"🧬 {c.display_name}")
    chips = [
        f"**MeSH** {c.mesh_heading} ({c.mesh_id})" if c.mesh_heading else "**MeSH** n/a",
        f"**ICD-10** {c.icd10_code} {c.icd10_name or ''}".strip() if c.icd10_code else "**ICD-10** n/a",
        f"**Geographies** {', '.join(b.input.geographies) or 'Global'}",
        f"**Patient segment** {b.input.segment_description()}",
        f"**Generated** {b.created_at:%Y-%m-%d %H:%M} UTC",
    ]
    st.markdown(" · ".join(chips))
    if b.is_example:
        st.info("Example report bundled with the app - opening it makes no API calls.", icon="📦")
    for note in c.notes:
        st.caption(f"ℹ️ {note}")

    cols = st.columns(6)
    cols[0].metric("Active trials", f"{len(b.module3a.trials) if b.module3a else 0}",
                   help=f"{b.module3a.total_found if b.module3a else 0} found on ClinicalTrials.gov")
    cols[1].metric("Failed / discontinued", f"{len(b.module3b.failures) if b.module3b else 0}")
    cols[2].metric("PubMed sources", sum(1 for s in b.sources if s.kind == "pubmed"))
    cols[3].metric("Citations verified", f"{b.validation.ids_valid}/{b.validation.ids_checked}",
                   help="Every cited PMID / NCT ID resolves to a retrieved or live-verified record")
    share = b.fact_check.supported_share if b.fact_check else None
    cols[4].metric("Claims supported", f"{share:.0%}" if share is not None else "–",
                   help="Share of LLM-written claims the fact checker found fully supported by their cited sources")
    cols[5].metric("Run time", f"{b.timings.get('total', 0):.0f} s" if b.timings.get("total") else "–", help="PRD target: < 180 s")

    if b.errors:
        st.warning(
            f"**{len(b.errors)} part(s) of this report are missing or incomplete.** First issue: "
            f"{b.errors[0].module} - {b.errors[0].message}", icon="⚠️",
        )
        with st.expander("All data gaps"):
            for e in b.errors:
                st.markdown(f"- **{e.module}:** {e.message}")
    if b.fact_check is None and on_fact_check is not None:
        f1, f2 = st.columns([1, 3], vertical_alignment="center")
        if f1.button("🔎 Fact-check this report", key=f"fc_{b.id}", width="stretch"):
            on_fact_check(b)
        f2.caption("This report has no fact check yet. Checking re-fetches every cited source and verifies each claim against it.")

    d1, d2, _ = st.columns([1, 1, 3])
    name = f"briefing_{slug(c.display_name)}_{b.created_at:%Y%m%d}"
    d1.download_button("⬇️ Markdown", briefing_to_markdown(b), file_name=f"{name}.md", mime="text/markdown", width="stretch",
                       help="Full brief with tables and a numbered reference list")
    d2.download_button("⬇️ JSON", report_to_json(b), file_name=f"{name}.json", mime="application/json", width="stretch",
                       help="Complete report data - re-import it from the Report library after an app restart")

    tabs = st.tabs(["Executive Summary", "1 · Condition", "2 · SoC & MoA", "3A · Pipeline", "3B · Failures", "4 · Stakeholders",
                    "Fact check & Sources"])
    with tabs[0]:
        fact_check_caption(b.executive_summary)
        _exec_summary(b, smap)
    with tabs[1]:
        fact_check_caption(b.module1)
        _module1(b, smap)
    with tabs[2]:
        fact_check_caption(b.module2)
        _module2(b, smap)
    with tabs[3]:
        fact_check_caption(b.module3a)
        _module3a(b, smap)
    with tabs[4]:
        fact_check_caption(b.module3b)
        _module3b(b, smap)
    with tabs[5]:
        fact_check_caption(b.module4)
        _module4(b, smap)
    with tabs[6]:
        _fact_check(b, smap)
        _sources(b, smap)


def _missing(b: Briefing, module: str) -> None:
    msgs = [e.message for e in b.errors if e.module.startswith(module)]
    st.warning(f"{module} is unavailable for this report." + (f" {msgs[0]}" if msgs else ""))


def _exec_summary(b: Briefing, smap) -> None:
    es = b.executive_summary
    if not es:
        _missing(b, "Executive summary")
        return
    st.subheader(es.headline)
    left, right = st.columns(2)
    with left:
        st.markdown("##### Key takeaways")
        statements(es.key_takeaways, smap)
        st.markdown("##### White space")
        statements(es.white_space, smap)
    with right:
        st.markdown("##### Competitive threats")
        statements(es.competitive_threats, smap)
    st.markdown("##### Near-term catalysts (primary completion ≤ 18 months)")
    _df(
        [{"Fact check": _fc_cell(x), "NCT": f"https://clinicaltrials.gov/study/{x.nct_id}", "Asset": x.asset, "Sponsor": x.sponsor, "Phase": x.phase,
          "Primary completion": x.primary_completion, "Why it matters": x.why_it_matters} for x in es.catalysts],
        NCT=NCT_LINK,
    )
    if b.terminology_map:
        with st.expander(f"Terminology reconciled across modules ({len(b.terminology_map)})"):
            st.markdown("\n".join(f"- {k} → **{v}**" for k, v in b.terminology_map.items()))
    sources_expander(es, smap)


def _module1(b: Briefing, smap) -> None:
    m = b.module1
    if not m:
        _missing(b, "Module 1")
        return
    statements(m.overview, smap)
    st.markdown("#### Epidemiology & patient population sizing")
    _df([{"Fact check": _fc_cell(e), "Metric": e.metric, "Region": e.region, "Value": e.value, "Notes": e.population_note, "Sources": _ids(e.source_ids)} for e in m.epidemiology])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Severity stratification")
        _df([{"Fact check": _fc_cell(s), "Severity": s.severity, "Share": s.share, "Definition": s.definition, "Sources": _ids(s.source_ids)} for s in m.severity_stratification])
    with c2:
        st.markdown("##### Addressable population")
        statements(m.addressable_population, smap)
    st.markdown("#### Pathophysiology & disease manifestation")
    statements(m.pathophysiology, smap)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("##### Clinical manifestations")
        statements(m.clinical_manifestations, smap)
    with c2:
        st.markdown("##### Comorbidities")
        statements(m.comorbidities, smap)
    with c3:
        st.markdown("##### Prognosis")
        statements(m.prognosis, smap)
    st.markdown("#### Unmet needs matrix")
    _df([{"Fact check": _fc_cell(u), "Unmet need domain": u.domain, "Current limitation": u.current_limitation, "Clinical impact": u.clinical_impact,
          "Opportunity for novel assets": u.opportunity, "Sources": _ids(u.source_ids)} for u in m.unmet_needs])
    sources_expander(m, smap)


def _module2(b: Briefing, smap) -> None:
    m = b.module2
    if not m:
        _missing(b, "Module 2")
        return
    statements(m.algorithm_summary, smap)
    st.markdown("#### Line-of-therapy treatment algorithm")
    if m.lines_of_therapy:
        st.graphviz_chart(charts.therapy_flowchart_dot(m), width="stretch")
    _df([{"Fact check": _fc_cell(t), "Line": t.line, "Setting": t.setting, "Therapies": ", ".join(t.therapies), "Rationale": t.rationale, "Sources": _ids(t.source_ids)}
         for t in m.lines_of_therapy])
    st.markdown("#### Pharmacological mechanism of action")
    _df([{"Fact check": _fc_cell(x), "Target": x.target, "Class": x.therapy_class, "Example agents": ", ".join(x.example_agents), "Mechanism": x.mechanism,
          "Downstream effects": x.downstream_effects, "Sources": _ids(x.source_ids)} for x in m.moa])
    st.markdown("#### Clinical benchmark metrics")
    _df([{"Fact check": _fc_cell(x), "Therapy": x.therapy, "Class": x.drug_class, "Modality": x.modality, "Primary target": x.primary_target,
          "Primary efficacy benchmark": x.efficacy_benchmark, "Major safety / boxed warnings": x.safety_warnings, "Sources": _ids(x.source_ids)}
         for x in m.benchmarks])
    sources_expander(m, smap)


def _trial_rows(trials) -> list[dict]:
    return [
        {"NCT": t.url, "Drug / intervention": t.asset_name, "Modality": t.modality or "", "Target / MoA": t.target_moa or "",
         "Sponsor": t.sponsor, "Sponsor type": (t.sponsor_class or "").title(), "Phase": t.phase_label,
         "Status": t.status.replace("_", " ").title(), "Comparator type": t.comparator_class, "Comparator arm(s)": t.comparator_summary,
         "Primary completion": t.primary_completion_date or "", "Enrollment": t.enrollment, "Key": t.source_key}
        for t in trials
    ]


def _module3a(b: Briefing, smap) -> None:
    m = b.module3a
    if not m:
        _missing(b, "Module 3A")
        return
    st.caption(f"{len(m.trials)} active interventional trials (of {m.total_found} found) · started within {b.input.time_horizon_years} years")
    statements(m.pipeline_summary, smap, empty="")
    if m.trials:
        c1, c2 = st.columns(2)
        c1.plotly_chart(charts.comparator_mix_by_phase(m.trials), width="stretch")
        c2.plotly_chart(charts.modality_mix(m.trials), width="stretch")

    st.markdown("#### Active trial inventory")
    f1, f2, f3 = st.columns(3)
    phases = sorted({t.phase_label for t in m.trials})
    comps = sorted({t.comparator_class for t in m.trials})
    sel_phase = f1.multiselect("Phase", phases, key=f"ph_{b.id}")
    sel_comp = f2.multiselect("Comparator type", comps, key=f"cmp_{b.id}")
    industry_only = f3.toggle("Industry sponsors only", key=f"ind_{b.id}")
    trials = [
        t for t in m.trials
        if (not sel_phase or t.phase_label in sel_phase)
        and (not sel_comp or t.comparator_class in sel_comp)
        and (not industry_only or t.sponsor_class == "INDUSTRY")
    ]
    _df(_trial_rows(trials), NCT=NCT_LINK)
    st.caption(
        "Comparator type is derived from ClinicalTrials.gov arm-group types (placebo / active / sham / no-intervention); "
        "active comparators are matched against the Module 2 standard of care, with Claude labelling unmatched agents "
        "(e.g. sponsor code names)."
    )

    if m.differentiation:
        st.markdown("#### Differentiation vs standard of care")
        _df([{"Fact check": _fc_cell(d), "Asset": d.asset, "Modality": d.modality, "Target / MoA": d.target_moa, "Comparator evidence": d.comparator_evidence,
              "Differentiation vs SoC": d.differentiation_vs_soc, "Advantage axes": ", ".join(d.advantage_axes),
              "Trials": ", ".join(d.nct_ids), "Sources": _ids(d.source_ids)} for d in m.differentiation])
    sources_expander(m, smap)


def _module3b(b: Briefing, smap) -> None:
    m = b.module3b
    if not m:
        _missing(b, "Module 3B")
        return
    st.caption(
        f"{len(m.failures)} trials: terminated / suspended / withdrawn (updated within {b.input.time_horizon_years} years) "
        "plus completed trials whose primary-endpoint analyses were all non-significant (p ≥ 0.05)."
    )
    if m.lessons:
        st.markdown("#### Mechanistic lessons learned")
        statements(m.lessons, smap)
    if m.failures:
        st.plotly_chart(charts.failure_categories(m.failures), width="stretch")
    st.markdown("#### Failed / terminated / suspended trial log")
    rows = []
    for f in m.failures:
        sec = [smap[k] for k in f.sec_source_ids if k in smap]
        rows.append({
            "Fact check": _fc_cell(f), "NCT": f"https://clinicaltrials.gov/study/{f.nct_id}", "Asset": f.asset, "Sponsor": f.sponsor, "Phase": f.phase, "Status": f.status,
            "Comparator arm(s)": f.comparator_summary, "Reason for stoppage / failure": f.stop_reason, "Evidence": f.evidence,
            "Category": f.failure_category, "Key R&D takeaway": f.takeaway, "SEC filing": sec[0].url if sec else None,
        })
    _df(rows, NCT=NCT_LINK, **{"SEC filing": st.column_config.LinkColumn("SEC filing", display_text="filing ↗")})
    sources_expander(m, smap)


def _module4(b: Briefing, smap) -> None:
    m = b.module4
    if not m:
        _missing(b, "Module 4")
        return
    statements(m.narrative, smap, empty="")
    c1, c2 = st.columns([3, 2])
    with c1:
        if m.sponsors:
            st.plotly_chart(charts.sponsor_leaderboard(m.sponsors), width="stretch")
    with c2:
        st.markdown("#### Industry leadership board")
        _df([{"Sponsor": s.sponsor, "Type": (s.sponsor_class or "").title(), "Active Ph2/3": s.active_ph2_3, "All active": s.active_total}
             for s in m.sponsors])
    st.markdown("#### Academic centers & trial hubs")
    academic_only = st.toggle("Academic / hospital centers only", value=True, key=f"acad_{b.id}")
    _df([{"Center": x.name, "Country": x.country or "", "Academic": x.academic, "Active trials": x.trial_count}
         for x in m.centers if x.academic or not academic_only])
    st.markdown("#### Key opinion leader network")
    st.caption("Principal investigators / study chairs on registry records, plus first and senior authors of guideline and treatment literature.")
    _df([{"Name": k.name, "Affiliation": k.affiliation or "", "Roles": ", ".join(k.roles), "Trials": k.trial_count,
          "Key publications": k.publication_count, "Sources": _ids(k.source_ids)} for k in m.kols])
    sources_expander(m, smap)


MODULE_LABEL = {"executive_summary": "Executive summary", "module1": "1 · Condition", "module2": "2 · SoC & MoA",
                "module3a": "3A · Pipeline", "module3b": "3B · Failures", "module4": "4 · Stakeholders"}


def _fact_check(b: Briefing, smap) -> None:
    st.markdown("#### Fact check")
    fc = b.fact_check
    if fc is None:
        st.info("This report has no fact check. Use **Fact-check this report** above to verify every claim against its sources.", icon="🔎")
        return
    counts = fc.counts()
    cols = st.columns(4)
    for col, verdict in zip(cols, VERDICT_ICON):
        col.metric(f"{VERDICT_ICON[verdict]} {verdict}", counts[verdict])
    st.caption(
        f"{len(fc.claims)} LLM-written claims were checked against the text of the sources they cite (abstracts, registry records) "
        f"by {fc.model}, using no outside knowledge. A 'Supported' verdict also has to pass two code checks: the supporting quote must "
        "appear in the cited source, and every number in the claim must appear there too. Registry facts copied by code "
        "(trial tables, rankings) are correct by construction and are not re-checked."
    )
    f1, f2 = st.columns([2, 2])
    default = [v for v in VERDICT_ICON if v != "Supported"] if counts["Supported"] < len(fc.claims) else list(VERDICT_ICON)
    verdicts = f1.multiselect("Verdict", list(VERDICT_ICON), default=default, key=f"fcv_{b.id}")
    modules = f2.multiselect("Section", list(MODULE_LABEL.values()), key=f"fcm_{b.id}")
    rows = []
    for c in fc.claims:
        label = MODULE_LABEL.get(c.module, c.module)
        if c.verdict not in verdicts or (modules and label not in modules):
            continue
        quote = f"“{c.evidence_quote}” [{c.evidence_source}]" + (" ✓ found in source" if c.quote_verified else "") if c.evidence_quote else ""
        rows.append({"Verdict": f"{VERDICT_ICON[c.verdict]} {c.verdict}", "Section": label, "Claim": c.claim,
                     "Cited sources": _ids(c.source_ids), "Why": c.rationale, "Evidence": quote,
                     "Numbers not in source": ", ".join(c.unmatched_numbers)})
    _df(rows)

    if fc.source_issues:
        st.markdown("##### Source integrity")
        sev_icon = {"critical": "⛔ Critical", "warning": "⚠️ Warning", "info": "ℹ️ Info"}
        _df([{"Severity": sev_icon[i.severity], "Source": i.key, "Title": smap[i.key].title if i.key in smap else "",
              "Issue": i.issue, "Link": smap[i.key].url if i.key in smap else None} for i in fc.source_issues],
            Link=st.column_config.LinkColumn("Link", display_text="open ↗"))
    else:
        st.success("No retracted, corrected or unreachable sources among those cited.", icon="✅")


def _sources(b: Briefing, smap) -> None:
    v = b.validation
    st.divider()
    st.markdown("#### Citation validation (identifiers)")
    c = st.columns(4)
    c[0].metric("Citations checked", v.ids_checked)
    c[1].metric("Valid", v.ids_valid)
    c[2].metric("Verified live", v.ids_resolved_live)
    c[3].metric("Unsupported items removed", v.statements_dropped)
    if v.issues:
        with st.expander(f"Validation log ({len(v.issues)} entries)"):
            _df([{"Location": i.location, "Issue": i.problem, "Action": i.action} for i in v.issues])
    else:
        st.success("Every citation resolved to a retrieved record; nothing was removed.", icon="✅")

    st.markdown("#### Bibliography")
    cited = cited_keys(b)
    f1, f2 = st.columns([2, 1])
    kinds = f1.multiselect("Source type", list(KIND_LABEL.values()), default=list(KIND_LABEL.values()), key=f"kinds_{b.id}")
    scope = f2.radio("Show", ["All", "Cited", "Consulted only"], horizontal=True, key=f"scope_{b.id}")
    usage: dict[str, list[str]] = {}
    for c in (b.fact_check.claims if b.fact_check else []):
        for k in c.source_ids:
            usage.setdefault(k, []).append(c.verdict)
    rows = []
    for s in b.sources:
        is_cited = s.key in cited
        if KIND_LABEL[s.kind] not in kinds or (scope == "Cited" and not is_cited) or (scope == "Consulted only" and is_cited):
            continue
        row = {"Key": s.key, "Type": KIND_LABEL[s.kind], "ID": s.native_id.split(":")[0], "Title": s.title, "Details": s.detail,
               "Link": s.url, "Cited": is_cited}
        if b.fact_check:
            v_list = usage.get(s.key, [])
            row["Claims citing it"] = len(v_list)
            row["…of which supported"] = sum(1 for x in v_list if x == "Supported")
        row["Retrieved"] = s.retrieved_at.strftime("%Y-%m-%d %H:%M")
        rows.append(row)
    _df(rows, Link=st.column_config.LinkColumn("Link", display_text="open ↗"))

    st.markdown("#### Search strategy")
    _df([{"Source": q.source, "Purpose": q.purpose, "Query": q.query, "Results": q.n_results, "API request": q.url} for q in b.queries],
        **{"API request": st.column_config.LinkColumn("API request", display_text="request ↗")})

    with st.expander("Run details"):
        st.json({"timings_s": b.timings, "models": b.models, "token_usage": b.usage, "report_id": b.id, "input": b.input.model_dump()})
