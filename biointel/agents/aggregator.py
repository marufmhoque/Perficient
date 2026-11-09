"""Aggregator Agent - Executive Briefing Synthesizer.

* reconciles target / mechanism terminology across modules (LLM proposes a canonical map, code applies it)
* writes the C-suite executive summary and picks near-term catalysts from a code-built candidate list
"""

from __future__ import annotations

from datetime import date, timedelta

from pydantic import BaseModel, Field

from ..schemas import (
    Catalyst,
    CitedStatement,
    ExecutiveSummary,
    Module1,
    Module2,
    Module3A,
    Module3B,
    Module4,
    ResolvedCondition,
    TrialRecord,
)
from .base import CITATION_RULES, Agent, format_trial_line


class TermMapping(BaseModel):
    variant: str = Field(description="A name exactly as it appears in the TERMS list")
    canonical: str = Field(description="Preferred canonical name for the same target / mechanism")


class CatalystPick(BaseModel):
    nct_id: str = Field(description="NCT ID from the CATALYST CANDIDATES list only")
    why_it_matters: str


class ExecSummaryLLM(BaseModel):
    headline: str = Field(description="One-sentence strategic headline")
    key_takeaways: list[CitedStatement]
    white_space: list[CitedStatement]
    competitive_threats: list[CitedStatement]
    catalysts: list[CatalystPick]
    terminology_map: list[TermMapping]


AGG_SYSTEM = f"""You are the head of competitive intelligence at a biopharma company, synthesizing an
executive briefing for the C-suite from analyst modules. Be decisive and specific: what matters,
why, and what to watch. 4-6 key takeaways, 3-5 white-space opportunities, 3-5 competitive threats,
and up to 6 near-term catalysts chosen only from the candidate list.

Terminology reconciliation: in TERMS, find names that refer to the same molecular target or
mechanism but are written differently (e.g. 'IL-23p19' vs 'IL-23 (p19)', 'TNF-α' vs 'TNF-alpha')
and map each variant to one canonical form. Only include genuine variants.

{CITATION_RULES}"""


def _iso_day(d: str) -> str:
    """CT.gov dates are 'YYYY-MM-DD' or 'YYYY-MM'; use mid-month for the latter."""
    return d if len(d) >= 10 else (f"{d}-15" if len(d) == 7 else f"{d}-07-01")


def catalyst_candidates(trials: list[TrialRecord], months: int = 18) -> list[TrialRecord]:
    today = date.today()
    lo = (today - timedelta(days=90)).isoformat()
    hi = (today + timedelta(days=int(months * 30.5))).isoformat()
    out = [
        t for t in trials
        if t.max_phase >= 2 and t.primary_completion_date and lo <= _iso_day(t.primary_completion_date) <= hi
    ]
    out.sort(key=lambda t: (0 if t.sponsor_class == "INDUSTRY" else 1, -t.max_phase, t.primary_completion_date or ""))
    return out[:25]


def _statements(items: list[CitedStatement], limit: int = 6) -> str:
    return "\n".join(f"  - {s.text} [{', '.join(s.source_ids)}]" for s in items[:limit])


def build_digest(
    cond: ResolvedCondition, m1: Module1 | None, m2: Module2 | None, m3a: Module3A | None, m3b: Module3B | None, m4: Module4 | None
) -> str:
    parts = [f"CONDITION: {cond.display_name} (MeSH: {cond.mesh_heading or 'n/a'}, ICD-10: {cond.icd10_code or 'n/a'})"]
    if m1:
        parts.append("MODULE 1 - CONDITION BRIEF")
        parts.append(_statements(m1.overview, 3))
        parts += [f"  - {e.metric} ({e.region}): {e.value}; {e.population_note} [{', '.join(e.source_ids)}]" for e in m1.epidemiology[:8]]
        parts += [f"  - Unmet need - {u.domain}: {u.current_limitation} -> {u.opportunity} [{', '.join(u.source_ids)}]" for u in m1.unmet_needs]
    if m2:
        parts.append("MODULE 2 - STANDARD OF CARE")
        parts.append(_statements(m2.algorithm_summary, 4))
        parts += [f"  - {b.therapy} ({b.primary_target}): {b.efficacy_benchmark} [{', '.join(b.source_ids)}]" for b in m2.benchmarks[:10]]
    if m3a:
        parts.append(f"MODULE 3A - ACTIVE PIPELINE ({m3a.total_found} active trials found)")
        parts.append(_statements(m3a.pipeline_summary, 5))
        parts += [
            f"  - {d.asset} ({d.modality}; {d.target_moa}): {d.differentiation_vs_soc} | comparator evidence: {d.comparator_evidence} "
            f"[{', '.join(d.source_ids)}]"
            for d in m3a.differentiation[:12]
        ]
    if m3b:
        parts.append(f"MODULE 3B - FAILURES ({len(m3b.failures)} discontinued / failed trials)")
        parts += [f"  - {f.asset} ({f.sponsor}, {f.phase}): {f.failure_category} - {f.takeaway} [{', '.join(f.source_ids)}]" for f in m3b.failures[:12]]
        parts.append(_statements(m3b.lessons, 5))
    if m4:
        parts.append("MODULE 4 - STAKEHOLDERS")
        parts += [f"  - {s.sponsor}: {s.active_ph2_3} active Ph2/3 trials [{', '.join(s.source_ids[:3])}]" for s in m4.sponsors[:8]]
        parts.append(_statements(m4.narrative, 4))
    return "\n".join(p for p in parts if p)


def collect_terms(m2: Module2 | None, m3a: Module3A | None) -> list[str]:
    terms: list[str] = []
    if m2:
        terms += [m.target for m in m2.moa] + [b.primary_target for b in m2.benchmarks]
    if m3a:
        terms += [d.target_moa for d in m3a.differentiation]
        for t in m3a.trials:
            if t.target_moa:
                terms += [p.split(" – ")[0] for p in t.target_moa.split("; ")]
    return list(dict.fromkeys(x.strip() for x in terms if x and x.strip()))[:120]


def apply_terminology(mapping: dict[str, str], m2: Module2 | None, m3a: Module3A | None) -> None:
    lower = {k.lower(): v for k, v in mapping.items()}

    def fix(value: str) -> str:
        return lower.get(value.strip().lower(), value)

    if m2:
        for m in m2.moa:
            m.target = fix(m.target)
        for b in m2.benchmarks:
            b.primary_target = fix(b.primary_target)
    if m3a:
        for d in m3a.differentiation:
            d.target_moa = fix(d.target_moa)
        for t in m3a.trials:
            if t.target_moa:
                parts = []
                for p in t.target_moa.split("; "):
                    head, sep, tail = p.partition(" – ")
                    parts.append(f"{fix(head)}{sep}{tail}")
                t.target_moa = "; ".join(parts)


class AggregatorAgent(Agent):
    name = "Aggregator · Executive Briefing Synthesizer"
    stage = "aggregator"

    async def run(
        self, cond: ResolvedCondition, m1, m2, m3a, m3b, m4, segment: str = "All patients"
    ) -> tuple[ExecutiveSummary, dict[str, str]]:
        candidates = catalyst_candidates(m3a.trials if m3a else [])
        terms = collect_terms(m2, m3a)
        user = (
            f"TARGET PATIENT SEGMENT: {segment}\n"
            + build_digest(cond, m1, m2, m3a, m3b, m4)
            + "\n\nCATALYST CANDIDATES (Phase 2/3, primary completion within ~18 months):\n"
            + ("\n".join(format_trial_line(t) for t in candidates) or "(none)")
            + "\n\nTERMS:\n" + "\n".join(f"- {t}" for t in terms)
            + "\n\nAll source keys above are valid SOURCES. Write the executive summary."
        )
        self.report("Synthesizing executive summary and reconciling terminology with Claude…")
        out: ExecSummaryLLM = await self.llm.parse(
            model=self.settings.aggregator_model, effort=self.settings.aggregator_effort, output_format=ExecSummaryLLM,
            system=AGG_SYSTEM, user=user,
        )
        mapping = {m.variant: m.canonical for m in out.terminology_map if m.variant and m.canonical and m.variant != m.canonical}
        apply_terminology(mapping, m2, m3a)

        by_nct = {t.nct_id.upper(): t for t in candidates}
        catalysts = []
        for pick in out.catalysts:
            t = by_nct.get(pick.nct_id.strip().upper())
            if t is None:  # not a provided candidate -> drop (zero-hallucination rule)
                continue
            catalysts.append(
                Catalyst(
                    nct_id=t.nct_id, asset=t.asset_name, sponsor=t.sponsor, phase=t.phase_label,
                    primary_completion=t.primary_completion_date or "", why_it_matters=pick.why_it_matters,
                    source_ids=[t.source_key],
                )
            )
        summary = ExecutiveSummary(
            headline=out.headline, key_takeaways=out.key_takeaways, white_space=out.white_space,
            competitive_threats=out.competitive_threats, catalysts=catalysts,
        )
        self.report("Executive summary complete.")
        return summary, mapping
