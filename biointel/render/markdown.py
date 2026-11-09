"""Briefing -> Markdown document (tables per PRD section 3, numbered references at the end)."""

from __future__ import annotations

from ..citations import cited_keys
from ..schemas import Briefing, CitedStatement, Source


def _esc(text: str | None) -> str:
    return (text or "").replace("|", "\\|").replace("\n", " ").strip()


class Refs:
    """Inline citation markers -> numbered references, in first-cited order."""

    def __init__(self, sources: dict[str, Source]) -> None:
        self.sources = sources
        self.order: list[str] = []

    def mark(self, ids: list[str]) -> str:
        out = []
        for key in ids:
            src = self.sources.get(key)
            if not src:
                continue
            if key not in self.order:
                self.order.append(key)
            out.append(f"[{key}]({src.url})")
        return (" " + " ".join(out)) if out else " *(unsourced)*"


def _stmts(items: list[CitedStatement], refs: Refs) -> list[str]:
    return [f"- {_esc(s.text)}{refs.mark(s.source_ids)}" for s in items]


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["_No rows._", ""]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(_esc(c) if not c.startswith("[") else c for c in row) + " |" for row in rows]
    return lines + [""]


def briefing_to_markdown(b: Briefing) -> str:
    refs = Refs(b.source_map())
    c = b.condition
    md: list[str] = [
        f"# Strategic Intelligence Briefing: {c.display_name}",
        "",
        f"*Generated {b.created_at:%Y-%m-%d %H:%M UTC} · MeSH: {c.mesh_heading or 'n/a'} ({c.mesh_id or '-'}) · "
        f"ICD-10: {c.icd10_code or 'n/a'} · Geographies: {', '.join(b.input.geographies) or 'Global'} · "
        f"Segment: {b.input.target_patient_segment or 'All'}*",
        "",
        f"> Citation check: {b.validation.ids_valid}/{b.validation.ids_checked} citations verified against retrieved records; "
        f"{b.validation.statements_dropped} unsupported items removed. Models: {b.models.get('agents', '?')} (agents), "
        f"{b.models.get('aggregator', '?')} (aggregator).",
        "",
    ]

    es = b.executive_summary
    if es:
        md += ["## Executive Summary", "", f"**{_esc(es.headline)}**", "", "### Key takeaways", *_stmts(es.key_takeaways, refs), ""]
        md += ["### White space", *_stmts(es.white_space, refs), "", "### Competitive threats", *_stmts(es.competitive_threats, refs), ""]
        if es.catalysts:
            md += ["### Near-term catalysts", ""]
            md += _table(
                ["NCT", "Asset", "Sponsor", "Phase", "Primary completion", "Why it matters"],
                [[f"[{x.nct_id}](https://clinicaltrials.gov/study/{x.nct_id})", x.asset, x.sponsor, x.phase, x.primary_completion,
                  x.why_it_matters + refs.mark(x.source_ids)] for x in es.catalysts],
            )

    m1 = b.module1
    if m1:
        md += ["## Module 1 · Condition Brief", "", *_stmts(m1.overview, refs), "", "### Epidemiology & patient population", ""]
        md += _table(["Metric", "Region", "Value", "Notes", "Sources"],
                     [[e.metric, e.region, e.value, e.population_note, refs.mark(e.source_ids).strip()] for e in m1.epidemiology])
        if m1.severity_stratification:
            md += _table(["Severity", "Share", "Definition", "Sources"],
                         [[s.severity, s.share, s.definition, refs.mark(s.source_ids).strip()] for s in m1.severity_stratification])
        md += ["**Addressable population**", *_stmts(m1.addressable_population, refs), ""]
        md += ["### Pathophysiology & disease manifestation", *_stmts(m1.pathophysiology, refs), ""]
        md += ["**Clinical manifestations**", *_stmts(m1.clinical_manifestations, refs), ""]
        md += ["**Comorbidities**", *_stmts(m1.comorbidities, refs), "", "**Prognosis**", *_stmts(m1.prognosis, refs), ""]
        md += ["### Unmet needs matrix", ""]
        md += _table(["Unmet need domain", "Current limitation", "Clinical impact", "Opportunity for novel assets", "Sources"],
                     [[u.domain, u.current_limitation, u.clinical_impact, u.opportunity, refs.mark(u.source_ids).strip()] for u in m1.unmet_needs])

    m2 = b.module2
    if m2:
        md += ["## Module 2 · Standard of Care & Mechanism of Action", "", *_stmts(m2.algorithm_summary, refs), ""]
        md += ["### Line-of-therapy algorithm", ""]
        md += _table(["Line", "Setting", "Therapies", "Rationale", "Sources"],
                     [[t.line, t.setting, ", ".join(t.therapies), t.rationale, refs.mark(t.source_ids).strip()] for t in m2.lines_of_therapy])
        md += ["### Mechanism of action breakdown", ""]
        md += _table(["Target", "Class", "Example agents", "Mechanism", "Downstream effects", "Sources"],
                     [[m.target, m.therapy_class, ", ".join(m.example_agents), m.mechanism, m.downstream_effects, refs.mark(m.source_ids).strip()]
                      for m in m2.moa])
        md += ["### Clinical benchmark metrics", ""]
        md += _table(["Therapy", "Class / modality", "Primary target", "Efficacy benchmark", "Safety / boxed warnings", "Sources"],
                     [[x.therapy, f"{x.drug_class} / {x.modality}", x.primary_target, x.efficacy_benchmark, x.safety_warnings,
                       refs.mark(x.source_ids).strip()] for x in m2.benchmarks])

    m3a = b.module3a
    if m3a:
        md += ["## Module 3A · Active Clinical Pipeline", "", f"*{len(m3a.trials)} active interventional trials shown (of {m3a.total_found} found).*", ""]
        md += _table(
            ["NCT", "Drug / intervention", "Modality", "Target / MoA", "Sponsor", "Phase", "Status", "Comparator arm", "Primary completion"],
            [[f"[{t.nct_id}]({t.url})", t.asset_name, t.modality or "", t.target_moa or "", t.sponsor, t.phase_label,
              t.status.replace("_", " ").title(), t.comparator_summary, t.primary_completion_date or ""] for t in m3a.trials],
        )
        if m3a.differentiation:
            md += ["### Differentiation vs standard of care", ""]
            md += _table(["Asset", "Modality", "Target / MoA", "Comparator evidence", "Differentiation vs SoC", "Advantage axes", "Sources"],
                         [[d.asset, d.modality, d.target_moa, d.comparator_evidence, d.differentiation_vs_soc, ", ".join(d.advantage_axes),
                           refs.mark(d.source_ids).strip()] for d in m3a.differentiation])
        if m3a.pipeline_summary:
            md += ["**Pipeline summary**", *_stmts(m3a.pipeline_summary, refs), ""]

    m3b = b.module3b
    if m3b:
        md += ["## Module 3B · Discontinued & Failed Trials Audit", ""]
        md += _table(
            ["NCT", "Asset", "Sponsor", "Phase", "Status", "Comparator arm", "Reason for stoppage / failure", "Category", "Key R&D takeaway", "SEC"],
            [[f"[{f.nct_id}](https://clinicaltrials.gov/study/{f.nct_id})", f.asset, f.sponsor, f.phase, f.status, f.comparator_summary,
              f.stop_reason, f.failure_category, f.takeaway, refs.mark(f.sec_source_ids).strip() if f.sec_source_ids else ""] for f in m3b.failures],
        )
        if m3b.lessons:
            md += ["### Mechanistic lessons learned", *_stmts(m3b.lessons, refs), ""]

    m4 = b.module4
    if m4:
        md += ["## Module 4 · Key Stakeholders & Ecosystem", "", *_stmts(m4.narrative, refs), "", "### Industry leadership board", ""]
        md += _table(["Sponsor", "Class", "Active Ph2/3 trials", "All active trials"],
                     [[s.sponsor, (s.sponsor_class or "").title(), str(s.active_ph2_3), str(s.active_total)] for s in m4.sponsors])
        md += ["### Trial centers", ""]
        md += _table(["Center", "Country", "Academic", "Active trials"],
                     [[x.name, x.country or "", "yes" if x.academic else "", str(x.trial_count)] for x in m4.centers])
        md += ["### Key opinion leaders", ""]
        md += _table(["Name", "Affiliation", "Roles", "Trials", "Key publications", "Sources"],
                     [[k.name, k.affiliation or "", ", ".join(k.roles), str(k.trial_count), str(k.publication_count), refs.mark(k.source_ids).strip()]
                      for k in m4.kols])

    if b.errors:
        md += ["## Data gaps", "", *[f"- **{e.module}:** {_esc(e.message)}" for e in b.errors], ""]

    # References ------------------------------------------------------------
    smap = b.source_map()
    cited = cited_keys(b)
    ordered = refs.order + sorted(k for k in cited if k not in refs.order and k in smap)
    md += ["## References (cited)", ""]
    md += [f"{i}. **[{k}]** {_esc(smap[k].title)}. {_esc(smap[k].detail)}. <{smap[k].url}>" for i, k in enumerate(ordered, 1)]
    consulted = [s for s in b.sources if s.key not in cited]
    if consulted:
        md += ["", "## Sources consulted but not cited", ""]
        md += [f"- **[{s.key}]** {_esc(s.title)}. {_esc(s.detail)}. <{s.url}>" for s in consulted]
    if b.queries:
        md += ["", "## Search strategy", ""]
        md += _table(["Source", "Purpose", "Query", "Results"], [[q.source, q.purpose, q.query, str(q.n_results)] for q in b.queries])
    return "\n".join(md)
