"""Agent 2 - Emerging Pipeline & Failure Analysis Specialist (Modules 3A, 3B, 4).

run()       : registry scans (active / stopped / completed-with-results), deterministic parsing,
              comparator classification, failure detection, SEC corroboration, stakeholder rankings,
              and two concurrent Claude calls (asset classification, failure analysis).
finalize()  : needs Agent 1's SoC baseline (Module 2) - labels active comparators as SoC or not,
              writes the differentiation matrix, merges literature authors into the KOL list and
              writes the stakeholder narrative.
"""

from __future__ import annotations

import asyncio
import re
from datetime import date, timedelta
from typing import Literal

from pydantic import BaseModel, Field

from ..schemas import (
    CitedStatement,
    ConditionInput,
    DifferentiationEntry,
    FailureCategory,
    FailureRow,
    Module2,
    Module3A,
    Module3B,
    Module4,
    ModuleError,
    ResolvedCondition,
    TrialRecord,
)
from ..sources.clinicaltrials import ACTIVE_STATUSES, STOPPED_STATUSES, ClinicalTrialsAdapter
from ..sources.pubmed import PubMedArticle
from ..sources.sec_edgar import SecEdgarAdapter
from ..stakeholders import merge_authors, rank_centers, rank_officials, rank_sponsors, top_kols
from ..trials import core_drug_name, failed_primary_endpoint, norm
from .base import CITATION_RULES, Agent, format_trial_line

Modality = Literal[
    "Oral small molecule", "Topical small molecule", "Injectable small molecule", "Monoclonal antibody",
    "Bispecific / multispecific antibody", "Antibody fragment / nanobody", "Fusion protein", "Peptide",
    "Cell therapy", "Gene therapy", "RNA therapeutic", "Vaccine", "Biosimilar", "Device / procedure",
    "Phototherapy", "Behavioral / digital", "Natural product / herbal", "Other / unclear",
]


# ---------------------------------------------------------------------------
# LLM output schemas (no defaults -> structured-output friendly)
# ---------------------------------------------------------------------------


class AssetClass(BaseModel):
    asset: str = Field(description="Asset name exactly as given in the input list")
    modality: Modality
    target: str = Field(description="Primary molecular target, e.g. 'TYK2', 'IL-23 p19'; 'Not disclosed' if unknown")
    mechanism: str = Field(description="Short mechanism, e.g. 'Allosteric TYK2 inhibitor'")
    confidence: Literal["high", "medium", "low"]


class AssetClassification(BaseModel):
    assets: list[AssetClass]


class FailureAssessment(BaseModel):
    nct_id: str
    failure_category: FailureCategory
    takeaway: str = Field(description="One-sentence R&D takeaway grounded in the stop reason / results")


class FailureAnalysis(BaseModel):
    assessments: list[FailureAssessment]
    lessons: list[CitedStatement] = Field(description="3-6 mechanistic / R&D lessons across failures, citing T keys")


class ComparatorSoCLabel(BaseModel):
    agent: str = Field(description="Comparator agent exactly as given")
    is_soc: bool = Field(description="True if this agent (or its class) is part of the current standard of care")


class DifferentiationOutput(BaseModel):
    entries: list[DifferentiationEntry]
    pipeline_summary: list[CitedStatement]
    comparator_labels: list[ComparatorSoCLabel]


class StakeholderNarrative(BaseModel):
    narrative: list[CitedStatement]


CLASSIFY_SYSTEM = """You are a pharmaceutical pipeline analyst. Classify each investigational asset by
modality and molecular target using the trial context provided plus well-established public knowledge
of the compound (e.g. sponsor code names). If the target is not identifiable with confidence, use
target 'Not disclosed' and confidence 'low' - never guess a target. Return one row per input asset,
with the asset name copied exactly."""

FAILURE_SYSTEM = f"""You are an R&D strategy analyst auditing discontinued and failed clinical trials.
For each trial, categorize the failure from the registry evidence (verbatim stop reason and/or
primary-endpoint p-values) and give a one-sentence R&D takeaway. Use 'Other / unspecified' when the
stop reason is vague. Do not speculate beyond the evidence; a business/strategic stop is not an
efficacy failure. Then synthesize 3-6 cross-trial lessons (patterns by target class, modality,
endpoint design or comparator choice).

{CITATION_RULES}"""

DIFF_SYSTEM = f"""You are a competitive-intelligence lead comparing the emerging clinical pipeline with
the current standard of care (SoC). For the most advanced / strategically relevant assets, explain how
each aims to differentiate from SoC (efficacy, safety, route, dosing, selectivity, novel mechanism,
population). Weigh comparator design: beating an active SoC comparator head-to-head is stronger
evidence than beating placebo; single-arm data is weakest. Also label each listed active-comparator
agent as SoC or not (consider drug classes and code names, e.g. a sponsor code name of an approved
drug). Finish with 3-5 pipeline summary statements.

{CITATION_RULES}"""

STAKEHOLDER_SYSTEM = f"""You are a competitive-intelligence analyst. Using the ranked sponsors, trial
centers and key opinion leaders provided, write 3-5 concise statements on who is driving innovation
in this indication (industry concentration, academic hubs, notable investigators and geographic
patterns).

{CITATION_RULES}"""


class PipelineResult(BaseModel):
    active: list[TrialRecord] = Field(default_factory=list)
    active_total: int = 0
    failures: list[TrialRecord] = Field(default_factory=list)
    stopped_total: int = 0
    failure_rows: list[FailureRow] = Field(default_factory=list)
    lessons: list[CitedStatement] = Field(default_factory=list)
    errors: list[ModuleError] = Field(default_factory=list)


def _asset_groups(trials: list[TrialRecord]) -> dict[str, dict]:
    """Group trials by normalized core drug name of their experimental agents."""
    groups: dict[str, dict] = {}
    for t in trials:
        agents = t.experimental_agents or [t.asset_name]
        for a in agents:
            core = core_drug_name(a)
            key = norm(core)
            if not key:
                continue
            g = groups.setdefault(key, {"name": core, "trials": [], "max_phase": -1, "industry": False})
            g["trials"].append(t)
            g["max_phase"] = max(g["max_phase"], t.max_phase)
            g["industry"] = g["industry"] or t.sponsor_class == "INDUSTRY"
    return groups


def _trial_sort_key(t: TrialRecord) -> tuple:
    return (-t.max_phase, 0 if t.sponsor_class == "INDUSTRY" else 1, t.primary_completion_date or "9999")


class PipelineAgent(Agent):
    name = "Agent 2 · Pipeline & Failure Analyst"
    stage = "agent2"

    def __init__(self, llm, settings, trials: ClinicalTrialsAdapter, sec: SecEdgarAdapter, progress=None) -> None:
        super().__init__(llm, settings, progress)
        self.ct = trials
        self.sec = sec

    # ------------------------------------------------------------------ run
    async def run(self, cond: ResolvedCondition, inp: ConditionInput) -> PipelineResult:
        s = self.settings
        res = PipelineResult()
        since = date.today() - timedelta(days=365 * max(1, inp.time_horizon_years))
        phases = inp.development_phases or ["PHASE1", "PHASE2", "PHASE3"]
        late = [p for p in phases if p in ("PHASE2", "PHASE3")] or ["PHASE2", "PHASE3"]
        term = cond.trial_query_term
        ages = inp.std_ages()  # age segments become a registry filter; other segments steer the analysis
        self.segment = inp.segment_description()

        age_note = f" · ages {', '.join(a.replace('_', ' ').lower() for a in ages)}" if ages else ""
        self.report(f'Querying ClinicalTrials.gov for "{term}"{age_note} (active, stopped, completed-with-results)…')
        tasks = [self.ct.search(term, ACTIVE_STATUSES, "Agent 2 · active pipeline", s.ctgov_active_max, phases=phases,
                                start_since=since, std_ages=ages)]
        if inp.include_discontinued:
            tasks.append(self.ct.search(term, STOPPED_STATUSES, "Agent 2 · terminated / suspended / withdrawn", s.ctgov_stopped_max,
                                        phases=phases, updated_since=since, std_ages=ages))
            tasks.append(self.ct.search(term, ["COMPLETED"], "Agent 2 · completed with results (endpoint check)", s.ctgov_results_max,
                                        phases=late, updated_since=since, has_results=True, with_results=True, std_ages=ages))
        results = await asyncio.gather(*tasks, return_exceptions=True)

        active_res = results[0]
        if isinstance(active_res, BaseException):
            res.errors.append(ModuleError(module="Module 3A", message=f"ClinicalTrials.gov active search failed: {active_res}"))
            active, res.active_total = [], 0
        else:
            active, res.active_total = active_res
        stopped: list[TrialRecord] = []
        completed: list[TrialRecord] = []
        if inp.include_discontinued:
            if isinstance(results[1], BaseException):
                res.errors.append(ModuleError(module="Module 3B", message=f"Stopped-trial search failed: {results[1]}"))
            else:
                stopped, res.stopped_total = results[1]
            if not isinstance(results[2], BaseException):
                completed, _ = results[2]

        active.sort(key=_trial_sort_key)
        failed_by_p = [t for t in completed if failed_primary_endpoint(t)]
        seen: set[str] = set()
        failures: list[TrialRecord] = []
        for t in sorted(stopped + failed_by_p, key=_trial_sort_key):
            if t.nct_id not in seen:
                seen.add(t.nct_id)
                failures.append(t)
        res.active, res.failures = active, failures
        self.report(
            f"{len(active)} active trials (of {res.active_total}), {len(stopped)} stopped, "
            f"{len(failed_by_p)} completed trials missing primary endpoint."
        )

        # Concurrent: Claude asset classification, Claude failure analysis, SEC corroboration
        self.report("Classifying assets and auditing failures with Claude; checking SEC filings…")
        cls_res, fail_res, sec_res = await asyncio.gather(
            self._classify_assets(active + failures),
            self._analyze_failures(cond, failures),
            self._sec_corroboration(failures),
            return_exceptions=True,
        )
        if isinstance(cls_res, BaseException):
            res.errors.append(ModuleError(module="Module 3A", message=f"Asset classification unavailable: {cls_res}"))
        if isinstance(sec_res, BaseException):
            sec_res = {}
            self.report("SEC EDGAR search unavailable; continuing without filings.")

        res.failure_rows, res.lessons = self._build_failure_rows(failures, fail_res, sec_res)
        if isinstance(fail_res, BaseException):
            res.errors.append(ModuleError(module="Module 3B", message=f"Failure analysis unavailable: {fail_res}"))
        self.report("Pipeline, failure audit and stakeholder data assembled.")
        return res

    async def _classify_assets(self, trials: list[TrialRecord]) -> None:
        groups = _asset_groups(trials)
        if not groups:
            return
        ranked = sorted(groups.values(), key=lambda g: (-g["max_phase"], not g["industry"], g["name"]))[: self.settings.llm_assets_max]
        lines = []
        for g in ranked:
            t0 = g["trials"][0]
            desc = next((i.description for i in t0.interventions if i.description and norm(core_drug_name(i.name)) == norm(g["name"])), "")
            lines.append(f"- {g['name']} | sponsor: {t0.sponsor} | trial: {t0.title[:140]}" + (f" | description: {desc[:200]}" if desc else ""))
        out: AssetClassification = await self.llm.parse(
            model=self.settings.agent_model, effort="low", output_format=AssetClassification, system=CLASSIFY_SYSTEM,
            user="ASSETS:\n" + "\n".join(lines) + "\n\nClassify every asset.",
        )
        by_key = {norm(a.asset): a for a in out.assets}
        for t in trials:
            mods, moas = [], []
            for a in t.experimental_agents or [t.asset_name]:
                c = by_key.get(norm(core_drug_name(a))) or by_key.get(norm(a))
                if c:
                    mods.append(c.modality)
                    moas.append(c.mechanism if c.target in ("", "Not disclosed") else f"{c.target} – {c.mechanism}")
            if mods:
                t.modality = " + ".join(dict.fromkeys(mods))
                t.target_moa = "; ".join(dict.fromkeys(moas))

    async def _analyze_failures(self, cond: ResolvedCondition, failures: list[TrialRecord]) -> FailureAnalysis | None:
        if not failures:
            return None
        lines = []
        for t in failures:
            evidence = []
            if t.why_stopped:
                evidence.append(f'stop reason: "{t.why_stopped[:300]}"')
            if t.primary_results:
                evidence.append("primary p-values: " + "; ".join(f"{r.comparison}: p={r.p_value_raw}" for r in t.primary_results[:4]))
            lines.append(format_trial_line(t, extra=" | ".join(evidence)))
        return await self.llm.parse(
            model=self.settings.agent_model, effort=self.settings.agent_effort, output_format=FailureAnalysis, system=FAILURE_SYSTEM,
            user=f"CONDITION: {cond.display_name}\n\nSOURCES (failed / discontinued trials):\n" + "\n".join(lines)
                 + "\n\nAssess every trial (use its NCT ID) and synthesize lessons.",
        )

    async def _sec_corroboration(self, failures: list[TrialRecord]) -> dict[str, list[str]]:
        """{normalized asset: [S keys]} for top industry-sponsored discontinued assets."""
        out: dict[str, list[str]] = {}
        candidates: list[str] = []
        for t in failures:
            if t.sponsor_class != "INDUSTRY":
                continue
            for a in t.experimental_agents[:1]:
                core = core_drug_name(a)
                if len(core) >= 4 and norm(core) not in {norm(c) for c in candidates}:
                    candidates.append(core)
        for asset in candidates[: self.settings.sec_assets_max]:
            try:
                hits = await self.sec.search_asset(asset, purpose="Agent 2 · SEC corroboration of discontinued asset")
            except Exception:
                continue
            if hits:
                out[norm(asset)] = [h.key for h in hits]
        return out

    def _build_failure_rows(self, failures, analysis, sec_map) -> tuple[list[FailureRow], list[CitedStatement]]:
        assessments = {}
        lessons: list[CitedStatement] = []
        if isinstance(analysis, FailureAnalysis):
            assessments = {a.nct_id.strip().upper(): a for a in analysis.assessments}
            lessons = analysis.lessons
        rows = []
        for t in failures:
            p_fail = failed_primary_endpoint(t)
            reasons = []
            if t.why_stopped:
                reasons.append(t.why_stopped)
            if p_fail:
                ps = "; ".join(f"{r.comparison}: p={r.p_value_raw}" for r in t.primary_results[:3])
                reasons.append(f"Primary endpoint not met ({ps})")
            evidence = "both" if (t.why_stopped and p_fail) else ("primary endpoint p-value" if p_fail else "whyStopped")
            a = assessments.get(t.nct_id.upper())
            core = norm(core_drug_name(t.experimental_agents[0])) if t.experimental_agents else ""
            rows.append(
                FailureRow(
                    nct_id=t.nct_id, asset=t.asset_name, sponsor=t.sponsor, phase=t.phase_label,
                    status=t.status.replace("_", " ").title(), stop_reason="; ".join(reasons) or "No reason recorded",
                    evidence=evidence,
                    failure_category=a.failure_category if a else ("Efficacy" if p_fail else "Other / unspecified"),
                    takeaway=a.takeaway if a else "",
                    comparator_summary=t.comparator_summary,
                    source_ids=[t.source_key] if t.source_key else [],
                    sec_source_ids=sec_map.get(core, []) if isinstance(sec_map, dict) else [],
                )
            )
        return rows, lessons

    # ------------------------------------------------------------- finalize
    @staticmethod
    def soc_vocabulary(module2: Module2) -> set[str]:
        raw: list[str] = []
        for line in module2.lines_of_therapy:
            raw += line.therapies
        for m in module2.moa:
            raw += m.example_agents
        for b in module2.benchmarks:
            raw.append(b.therapy)
        vocab: set[str] = set()
        for item in raw:
            for piece in re.split(r"[,/;()+]|\band\b|\bor\b", item):
                key = norm(core_drug_name(piece))
                if len(key) >= 5:
                    vocab.add(key)
        return vocab

    @staticmethod
    def label_comparators(trials: list[TrialRecord], vocab: set[str]) -> list[str]:
        """Deterministically mark active comparators that match SoC; return unresolved agent names."""
        unresolved: list[str] = []
        for t in trials:
            for c in t.comparators:
                if c.category != "Active comparator":
                    continue
                keys = [norm(core_drug_name(a)) for a in c.agents]
                if any(k and any(k in v or v in k for v in vocab if len(v) >= 5) for k in keys):
                    c.is_soc = True
                else:
                    for a in c.agents:
                        if a not in unresolved:
                            unresolved.append(a)
        return unresolved

    async def finalize(
        self, res: PipelineResult, module2: Module2 | None, articles: list[PubMedArticle]
    ) -> tuple[Module3A, Module3B, Module4, list[ModuleError]]:
        errors: list[ModuleError] = []
        all_trials = res.active + res.failures

        # Stakeholders (deterministic) - counts use full facility lists, then lists are trimmed
        people = merge_authors(rank_officials(all_trials), articles)
        module4 = Module4(sponsors=rank_sponsors(res.active), centers=rank_centers(res.active), kols=top_kols(people))
        for t in all_trials:
            t.facilities = t.facilities[:25]

        unresolved: list[str] = []
        if module2 is not None:
            unresolved = self.label_comparators(all_trials, self.soc_vocabulary(module2))

        self.report("Differentiating pipeline vs SoC and mapping stakeholders with Claude…")
        diff_task = self._differentiate(res, module2, unresolved) if module2 is not None else _none()
        narr_task = self._stakeholder_narrative(module4)
        diff, narr = await asyncio.gather(diff_task, narr_task, return_exceptions=True)

        module3a = Module3A(trials=res.active, total_found=res.active_total)
        if module2 is None:
            errors.append(ModuleError(module="Module 3A", message="Differentiation vs SoC skipped: Module 2 (SoC baseline) unavailable."))
        elif isinstance(diff, BaseException):
            errors.append(ModuleError(module="Module 3A", message=f"Differentiation analysis unavailable: {diff}"))
        elif diff is not None:
            module3a.differentiation = diff.entries
            module3a.pipeline_summary = diff.pipeline_summary
            labels = {norm(lbl.agent): lbl.is_soc for lbl in diff.comparator_labels}
            for t in all_trials:
                for c in t.comparators:
                    if c.category == "Active comparator" and c.is_soc is None:
                        hits = [labels[norm(a)] for a in c.agents if norm(a) in labels]
                        if hits:
                            c.is_soc = any(hits)

        if isinstance(narr, BaseException):
            errors.append(ModuleError(module="Module 4", message=f"Stakeholder narrative unavailable: {narr}"))
        elif narr is not None:
            module4.narrative = narr.narrative

        # Failure rows: comparator summaries may have gained SoC labels
        by_nct = {t.nct_id: t for t in res.failures}
        for row in res.failure_rows:
            if row.nct_id in by_nct:
                row.comparator_summary = by_nct[row.nct_id].comparator_summary
        module3b = Module3B(trials=res.failures, failures=res.failure_rows, lessons=res.lessons, total_stopped_found=res.stopped_total)
        return module3a, module3b, module4, errors

    async def _differentiate(self, res: PipelineResult, module2: Module2, unresolved: list[str]) -> DifferentiationOutput:
        soc_lines = []
        for m in module2.moa:
            soc_lines.append(f"- MoA: {m.target} ({m.therapy_class}; e.g. {', '.join(m.example_agents)}) [{', '.join(m.source_ids)}]")
        for b in module2.benchmarks:
            soc_lines.append(
                f"- Benchmark: {b.therapy} ({b.modality}, {b.primary_target}): {b.efficacy_benchmark}; safety: {b.safety_warnings} "
                f"[{', '.join(b.source_ids)}]"
            )
        for line in module2.lines_of_therapy:
            soc_lines.append(f"- {line.line} ({line.setting}): {', '.join(line.therapies)} [{', '.join(line.source_ids)}]")

        groups = _asset_groups(res.active)
        ranked = sorted(groups.values(), key=lambda g: (-g["max_phase"], not g["industry"], g["name"]))[:35]
        asset_lines = []
        for g in ranked:
            ts = sorted(g["trials"], key=_trial_sort_key)[:4]
            t0 = ts[0]
            comps = "; ".join(dict.fromkeys(t.comparator_summary for t in ts))
            asset_lines.append(
                f"- {g['name']} | modality: {t0.modality or '?'} | target/MoA: {t0.target_moa or '?'} | "
                f"phases: {', '.join(dict.fromkeys(t.phase_label for t in ts))} | sponsors: {', '.join(dict.fromkeys(t.sponsor for t in ts))} | "
                f"comparators: {comps} | trials: {', '.join(f'{t.nct_id} [{t.source_key}]' for t in ts)}"
            )
        user = (
            f"TARGET PATIENT SEGMENT: {getattr(self, 'segment', 'All patients')} - weigh differentiation for this segment.\n\n"
            "CURRENT STANDARD OF CARE (from Module 2, with source keys):\n" + "\n".join(soc_lines)
            + "\n\nSOURCES - PIPELINE ASSETS (trial keys in brackets):\n" + "\n".join(asset_lines)
            + "\n\nACTIVE-COMPARATOR AGENTS TO LABEL (SoC or not):\n"
            + ("\n".join(f"- {a}" for a in unresolved) if unresolved else "(none)")
            + "\n\nWrite the differentiation matrix (up to 15 most relevant assets), label the comparators, and summarize."
        )
        return await self.llm.parse(
            model=self.settings.agent_model, effort=self.settings.agent_effort, output_format=DifferentiationOutput,
            system=DIFF_SYSTEM, user=user,
        )

    async def _stakeholder_narrative(self, m4: Module4) -> StakeholderNarrative | None:
        if not (m4.sponsors or m4.centers or m4.kols):
            return None
        lines = ["SPONSORS (active Ph2/3 count, total active):"]
        lines += [f"- {s.sponsor} ({(s.sponsor_class or '?').lower()}): {s.active_ph2_3}, {s.active_total} [{', '.join(s.source_ids[:4])}]" for s in m4.sponsors]
        lines.append("TRIAL CENTERS (trial count):")
        lines += [f"- {c.name}, {c.country or '?'}: {c.trial_count} [{', '.join(c.source_ids[:4])}]" for c in m4.centers[:12]]
        lines.append("KEY OPINION LEADERS:")
        lines += [
            f"- {k.name} ({k.affiliation or 'affiliation n/a'}): {k.trial_count} trials, {k.publication_count} key publications; "
            f"roles: {', '.join(k.roles)} [{', '.join(k.source_ids[:4])}]"
            for k in m4.kols[:15]
        ]
        return await self.llm.parse(
            model=self.settings.agent_model, effort="low", output_format=StakeholderNarrative, system=STAKEHOLDER_SYSTEM,
            user="SOURCES:\n" + "\n".join(lines),
        )


async def _none() -> None:
    return None
