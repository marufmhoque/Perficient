"""Agent 1 - Disease Overview & Standard-of-Care Specialist (Modules 1 and 2).

Pipeline: targeted PubMed searches (guidelines, treatment SR/MA, epidemiology, pathophysiology)
-> efetch abstracts -> two concurrent Claude calls with structured outputs, grounded in the
retrieved abstracts and citing their P# keys.
"""

from __future__ import annotations

import asyncio
from datetime import date

from pydantic import BaseModel, Field

from ..schemas import ConditionInput, Module1, Module2, ModuleError, ResolvedCondition
from ..sources.pubmed import PubMedAdapter, PubMedArticle
from ..sources.resolve import pubmed_condition_clause
from .base import CITATION_RULES, Agent, format_articles

GEO_TERMS: dict[str, list[str]] = {
    "US": ['"United States"', "USA", "American"],
    "EU5": ["Europe", "European", "Germany", "France", "Italy", "Spain", '"United Kingdom"'],
    "EU": ["Europe", "European"],
    "UK": ['"United Kingdom"', "England", "British"],
    "Japan": ["Japan", "Japanese"],
    "China": ["China", "Chinese"],
    "Global": ["global", "worldwide"],
}

M1_SYSTEM = f"""You are a senior biomedical strategy analyst writing Module 1 (Condition Brief) of a
competitive-intelligence briefing for biopharma executives. Write in a crisp executive register:
one idea per statement, quantitative where the sources allow, no filler.

{CITATION_RULES}

Coverage guidance: overview 2-3 statements; epidemiology up to 10 figures (global and the requested
geographies first); severity stratification up to 4 strata with the defining instrument; addressable
population 1-3 statements (state the basis); pathophysiology 3-6 statements on initiating and
amplifying pathways; clinical manifestations 2-4; comorbidities 2-5; prognosis 1-3; unmet needs 3-6
rows covering efficacy/durability, safety/tolerability and administration burden where supported."""

M2_SYSTEM = f"""You are a senior clinical strategy analyst writing Module 2 (Standard of Care & Mechanism
of Action) of a competitive-intelligence briefing for biopharma executives.

{CITATION_RULES}

Coverage guidance: algorithm summary 2-4 statements; lines of therapy - map guideline
recommendations to First-line, Second-line and Third-line / Refractory, with the population/severity
setting for each (several rows per line are fine); MoA - one row per major target class in current
use (4-8), naming approved example agents by generic name and the downstream signalling cascade;
benchmarks - 4-10 approved therapies with the primary efficacy endpoint benchmark and timepoint
exactly as reported in the sources (e.g. response rates at week 12/16 from meta-analyses), plus major
safety and boxed warnings. Prefer guideline and meta-analysis sources for benchmarks."""


class ClinicalSoCResult(BaseModel):
    module1: Module1 | None = None
    module2: Module2 | None = None
    articles: list[PubMedArticle] = Field(default_factory=list)
    errors: list[ModuleError] = Field(default_factory=list)


class ClinicalSoCAgent(Agent):
    name = "Agent 1 · Clinical & SoC Specialist"
    stage = "agent1"

    def __init__(self, llm, settings, pubmed: PubMedAdapter, progress=None) -> None:
        super().__init__(llm, settings, progress)
        self.pubmed = pubmed

    def build_queries(self, cond: ResolvedCondition, inp: ConditionInput) -> list[tuple[str, str, int, int]]:
        """(tag, PubMed query, retmax, min publication year)."""
        s = self.settings
        clause = pubmed_condition_clause(cond)
        year = date.today().year
        queries = [
            ("guidelines", f'{clause} AND ("Practice Guideline"[pt] OR "Guideline"[pt] OR guideline*[ti] OR consensus[ti])',
             s.pubmed_guidelines_max, year - 10),
            ("treatment", f'{clause} AND ("drug therapy"[sh] OR "therapeutic use"[sh] OR treatment[ti] OR therapy[ti]) '
                          f'AND ("Systematic Review"[pt] OR "Meta-Analysis"[pt])',
             s.pubmed_treatment_max, year - 6),
            ("epidemiology", f'{clause} AND ("epidemiology"[sh] OR prevalence[ti] OR incidence[ti] OR burden[ti])',
             s.pubmed_epi_max, year - 12),
            ("pathophysiology", f'{clause} AND (pathogenesis[ti] OR pathophysiology[ti] OR immunopathogenesis[ti] '
                                f'OR mechanism*[ti]) AND Review[pt]',
             s.pubmed_patho_max, year - 6),
        ]
        geo = [t for g in inp.geographies for t in GEO_TERMS.get(g, [f'"{g}"'])]
        if geo:
            geo_clause = " OR ".join(f"{t}[tiab]" for t in dict.fromkeys(geo))
            queries.append(
                ("epidemiology-geo", f"{clause} AND (prevalence[ti] OR incidence[ti] OR epidemiology[ti]) AND ({geo_clause})",
                 6, year - 12)
            )
        return queries

    async def gather_literature(self, cond: ResolvedCondition, inp: ConditionInput) -> list[PubMedArticle]:
        queries = self.build_queries(cond, inp)
        self.report(f"Searching PubMed ({len(queries)} targeted queries)…")
        results = await asyncio.gather(
            *(self.pubmed.search(q, n, purpose=f"Agent 1 · {tag}", mindate=y) for tag, q, n, y in queries),
            return_exceptions=True,
        )
        tags_by_pmid: dict[str, list[str]] = {}
        order: list[str] = []
        for (tag, *_), ids in zip(queries, results):
            if isinstance(ids, BaseException):
                self.report(f"PubMed {tag} search failed: {ids}")
                continue
            for pmid in ids:
                if pmid not in tags_by_pmid:
                    tags_by_pmid[pmid] = []
                    order.append(pmid)
                tags_by_pmid[pmid].append(tag)
        self.report(f"Fetching {len(order)} abstracts…")
        articles = await self.pubmed.fetch(order)
        for a in articles:
            a.query_tags = tags_by_pmid.get(a.pmid, [])
        return articles

    def _context(self, cond: ResolvedCondition, inp: ConditionInput) -> str:
        lines = [f"CONDITION: {cond.display_name}"]
        if cond.mesh_heading:
            lines.append(f"MeSH: {cond.mesh_heading} ({cond.mesh_id}) - {cond.mesh_scope_note or ''}".strip())
        if cond.icd10_code:
            lines.append(f"ICD-10: {cond.icd10_code} {cond.icd10_name or ''}".strip())
        lines.append(f"Geographies of interest: {', '.join(inp.geographies) or 'Global'}")
        segment = inp.segment_description()
        if inp.all_patients:
            lines.append("Target patient segment: all patients - cover the full population and note how it splits by severity, age and line of therapy.")
        else:
            lines.append(f"Target patient segment: {segment} - prioritise evidence for this segment and say when a figure applies to a broader population.")
        return "\n".join(lines)

    async def run(self, cond: ResolvedCondition, inp: ConditionInput) -> ClinicalSoCResult:
        result = ClinicalSoCResult()
        articles = await self.gather_literature(cond, inp)
        result.articles = articles
        if not articles:
            msg = "No PubMed literature retrieved for this condition."
            result.errors += [ModuleError(module="Module 1", message=msg), ModuleError(module="Module 2", message=msg)]
            return result

        lim = self.settings.abstract_char_limit
        m1_docs = [a for a in articles if {"guidelines", "epidemiology", "epidemiology-geo", "pathophysiology"} & set(a.query_tags)]
        m2_docs = [a for a in articles if {"guidelines", "treatment", "pathophysiology"} & set(a.query_tags)]
        ctx = self._context(cond, inp)
        self.report(f"Synthesizing Module 1 ({len(m1_docs)} sources) and Module 2 ({len(m2_docs)} sources) with Claude…")

        m1_task = self.llm.parse(
            model=self.settings.agent_model, effort=self.settings.agent_effort, output_format=Module1, system=M1_SYSTEM,
            user=f"{ctx}\n\nSOURCES:\n{format_articles(m1_docs or articles, lim)}\n\nWrite Module 1 for {cond.display_name}.",
        )
        m2_task = self.llm.parse(
            model=self.settings.agent_model, effort=self.settings.agent_effort, output_format=Module2, system=M2_SYSTEM,
            user=f"{ctx}\n\nSOURCES:\n{format_articles(m2_docs or articles, lim)}\n\nWrite Module 2 for {cond.display_name}.",
        )
        m1, m2 = await asyncio.gather(m1_task, m2_task, return_exceptions=True)
        if isinstance(m1, BaseException):
            result.errors.append(ModuleError(module="Module 1", message=str(m1)))
        else:
            result.module1 = m1
        if isinstance(m2, BaseException):
            result.errors.append(ModuleError(module="Module 2", message=str(m2)))
        else:
            result.module2 = m2
        self.report("Modules 1 & 2 drafted.")
        return result
