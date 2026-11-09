"""Orchestrator: resolve -> (Agent 1 || Agent 2) -> finalize -> Aggregator -> citation validation.

Each stage runs under the remaining time budget (PRD NFR: < 180 s end to end); a failing or
slow stage degrades its own modules and is reported in ``Briefing.errors`` instead of failing the run.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable

from .agents import AggregatorAgent, ClinicalSoCAgent, ClinicalSoCResult, PipelineAgent, PipelineResult
from .citations import SourceRegistry
from .config import Settings, get_settings
from .llm import LLM, ClaudeLLM
from .schemas import Briefing, ConditionInput, ModuleError
from .sources import ClinicalTrialsAdapter, Http, PubMedAdapter, SecEdgarAdapter, resolve_condition
from .citations import cited_keys
from .factcheck import EvidenceStore, FactChecker, rekey
from .llm import LLMAccountError
from .validation import ANALYST_MODULES, CitationValidator

# progress(stage, message, state) where state in {"running", "complete", "error"}
Progress = Callable[[str, str, str], None]

STAGES = {
    "resolve": "Resolve condition (ICD-10 / MeSH) · check Claude account",
    "agent1": "Agent 1 · Clinical & SoC Specialist",
    "agent2": "Agent 2 · Pipeline & Failure Analyst",
    "finalize": "Differentiation vs SoC & stakeholder mapping",
    "validation": "Citation validation",
    "aggregator": "Aggregator · Executive Briefing Synthesizer",
    "factcheck": "Fact check · every claim vs its cited sources",
}


def _raise_if_fatal(llm) -> None:
    """Billing / auth failures make every later call fail: stop instead of saving a hollow report."""
    fatal = getattr(llm, "fatal", None)
    if fatal is not None:
        raise fatal


class BriefingTimeout(RuntimeError):
    pass


async def run_briefing(
    inp: ConditionInput,
    settings: Settings | None = None,
    progress: Progress | None = None,
    llm: LLM | None = None,
    http: Http | None = None,
) -> Briefing:
    settings = settings or get_settings()
    report = progress or (lambda stage, msg, state: None)
    llm = llm or ClaudeLLM(settings)  # raises a clear error if the API key is missing
    own_http = http is None
    http = http or Http(settings)
    registry = SourceRegistry()
    t0 = time.monotonic()
    timings: dict[str, float] = {}
    errors: list[ModuleError] = []

    def remaining(reserve: float = 0.0) -> float:
        return max(5.0, settings.total_budget_s - (time.monotonic() - t0) - reserve)

    def agent_progress(stage: str, message: str) -> None:
        report(stage, message, "running")

    pubmed = PubMedAdapter(http, registry, settings)
    trials = ClinicalTrialsAdapter(http, registry, settings)
    sec = SecEdgarAdapter(http, registry, settings)

    try:
        # 1. Resolve ------------------------------------------------------
        report("resolve", "Normalizing condition and checking the Claude account…", "running")
        s = time.monotonic()
        steps = [asyncio.wait_for(resolve_condition(inp, http, settings), timeout=30)]
        if hasattr(llm, "preflight"):
            steps.append(asyncio.wait_for(llm.preflight(), timeout=30))
        results = await asyncio.gather(*steps, return_exceptions=True)
        for res in results[1:]:  # account problems first: they are the actionable ones
            if isinstance(res, BaseException):
                report("resolve", str(res), "error")
                raise res
        if isinstance(results[0], BaseException):
            raise results[0]
        cond = results[0]
        timings["resolve"] = time.monotonic() - s
        bits = [cond.display_name]
        if cond.mesh_heading:
            bits.append(f"MeSH {cond.mesh_id} “{cond.mesh_heading}”")
        if cond.icd10_code:
            bits.append(f"ICD-10 {cond.icd10_code}")
        report("resolve", " · ".join(bits), "complete")

        # 2. Agents 1 & 2 in parallel ------------------------------------
        a1 = ClinicalSoCAgent(llm, settings, pubmed, agent_progress)
        a2 = PipelineAgent(llm, settings, trials, sec, agent_progress)

        async def timed(name: str, coro):
            start = time.monotonic()
            try:
                return await asyncio.wait_for(coro, timeout=remaining(reserve=45))
            finally:
                timings[name] = time.monotonic() - start

        report("agent1", "Starting…", "running")
        report("agent2", "Starting…", "running")
        r1, r2 = await asyncio.gather(timed("agent1", a1.run(cond, inp)), timed("agent2", a2.run(cond, inp)), return_exceptions=True)
        _raise_if_fatal(llm)

        if isinstance(r1, BaseException):
            msg = "timed out" if isinstance(r1, asyncio.TimeoutError) else str(r1)
            errors += [ModuleError(module="Module 1", message=f"Agent 1 failed: {msg}"), ModuleError(module="Module 2", message=f"Agent 1 failed: {msg}")]
            report("agent1", f"Failed: {msg}", "error")
            r1 = ClinicalSoCResult()
        else:
            errors += r1.errors
            report("agent1", f"{len(r1.articles)} PubMed sources · Modules 1–2 {'ready' if r1.module1 and r1.module2 else 'partial'}",
                   "complete" if not r1.errors else "error")
        if isinstance(r2, BaseException):
            msg = "timed out" if isinstance(r2, asyncio.TimeoutError) else str(r2)
            errors += [ModuleError(module=m, message=f"Agent 2 failed: {msg}") for m in ("Module 3A", "Module 3B", "Module 4")]
            report("agent2", f"Failed: {msg}", "error")
            r2 = None
        else:
            errors += r2.errors
            report("agent2", f"{len(r2.active)} active · {len(r2.failures)} failed/discontinued trials", "complete" if not r2.errors else "error")

        # 3. Finalize Agent 2 with Agent 1's SoC baseline ----------------
        m3a = m3b = m4 = None
        if isinstance(r2, PipelineResult):
            report("finalize", "Comparing pipeline with standard of care…", "running")
            s = time.monotonic()
            try:
                m3a, m3b, m4, fin_errors = await asyncio.wait_for(a2.finalize(r2, r1.module2, r1.articles), timeout=remaining(reserve=30))
                errors += fin_errors
                report("finalize", f"{len(m3a.differentiation)} assets differentiated · {len(m4.kols)} KOLs mapped",
                       "complete" if not fin_errors else "error")
            except Exception as exc:
                msg = "timed out" if isinstance(exc, asyncio.TimeoutError) else str(exc)
                errors.append(ModuleError(module="Module 3A/4", message=f"Finalize failed: {msg}"))
                report("finalize", f"Failed: {msg}", "error")
            timings["finalize"] = time.monotonic() - s
        else:
            report("finalize", "Skipped (no pipeline data)", "error")
        _raise_if_fatal(llm)

        briefing = Briefing(
            id=uuid.uuid4().hex[:12], input=inp, condition=cond,
            module1=r1.module1, module2=r1.module2, module3a=m3a, module3b=m3b, module4=m4,
            models={"agents": settings.agent_model, "aggregator": settings.aggregator_model},
        )

        # 4. Citation gate on analyst modules BEFORE synthesis, so the aggregator never
        #    builds conclusions on unsupported or fabricated claims -------------------
        report("validation", "Verifying module citations before synthesis…", "running")
        s = time.monotonic()
        validator = CitationValidator(registry, pubmed, trials, settings.drop_unsourced)
        await _validate(validator, briefing, ANALYST_MODULES, timeout=min(25.0, remaining(reserve=30)))
        timings["validation"] = time.monotonic() - s

        # 5a. Fact check of analyst modules runs concurrently with the aggregator --------
        checker: FactChecker | None = None
        module_fc: asyncio.Task | None = None
        fc_start = time.monotonic()
        if settings.fact_check:
            evidence = EvidenceStore()
            evidence.add_articles(r1.articles)
            if isinstance(r2, PipelineResult):
                evidence.add_trials(r2.active + r2.failures)
            try:  # sources not held in memory: SEC filing metadata, PMIDs the citation gate verified live
                await asyncio.wait_for(evidence.fill_missing(cited_keys(briefing), registry, http, settings), timeout=15)
            except Exception:
                pass  # missing text only makes the affected claims 'Cannot verify'
            checker = FactChecker(llm, settings, registry, evidence)
            report("factcheck", "Checking module claims against their cited sources…", "running")
            module_fc = asyncio.create_task(checker.check(briefing, ANALYST_MODULES))

        # 5b. Aggregator ---------------------------------------------------
        if any([briefing.module1, briefing.module2, briefing.module3a, briefing.module3b]):
            s = time.monotonic()
            agg = AggregatorAgent(llm, settings, agent_progress)
            try:
                summary, mapping = await asyncio.wait_for(
                    agg.run(cond, briefing.module1, briefing.module2, briefing.module3a, briefing.module3b, briefing.module4,
                            segment=inp.segment_description()),
                    timeout=remaining(reserve=8),
                )
                briefing.executive_summary, briefing.terminology_map = summary, mapping
                report("aggregator", f"Executive summary ready · {len(mapping)} terminology fixes", "complete")
            except Exception as exc:
                msg = "timed out" if isinstance(exc, asyncio.TimeoutError) else str(exc)
                errors.append(ModuleError(module="Executive summary", message=f"Aggregator failed: {msg}"))
                report("aggregator", f"Failed: {msg}", "error")
            timings["aggregator"] = time.monotonic() - s
        else:
            report("aggregator", "Skipped (no module content)", "error")

        if getattr(llm, "fatal", None) is not None and module_fc is not None:
            module_fc.cancel()
        _raise_if_fatal(llm)

        # 6. Citation gate on the executive summary ------------------------
        report("validation", "Verifying executive-summary citations…", "running")
        s = time.monotonic()
        await _validate(validator, briefing, ("executive_summary",), timeout=max(10.0, remaining()))
        timings["validation"] = timings.get("validation", 0.0) + time.monotonic() - s
        briefing.validation = validator.report
        v = briefing.validation
        report("validation", f"{v.ids_valid}/{v.ids_checked} citations valid · {v.statements_dropped} unsupported items removed", "complete")

        # 7. Fact check: executive summary, then merge with the module results -------------
        if checker is not None and module_fc is not None:
            try:
                exec_checks = await asyncio.wait_for(checker.check(briefing, ("executive_summary",)), timeout=remaining(reserve=2))
                module_checks = await asyncio.wait_for(module_fc, timeout=remaining())
                checks = rekey(module_checks + exec_checks, briefing)  # terminology fixes may have edited claim text
                briefing.fact_check = checker.report(checks, cited_keys(briefing))
                c = briefing.fact_check.counts()
                report("factcheck", f"{c['Supported']} supported · {c['Partially supported']} partially · "
                                    f"{c['Not supported']} not supported · {c['Cannot verify']} unverifiable", "complete")
            except Exception as exc:
                module_fc.cancel()
                if isinstance(exc, LLMAccountError):
                    raise
                msg = "timed out" if isinstance(exc, asyncio.TimeoutError) else str(exc)
                errors.append(ModuleError(module="Fact check", message=f"Fact check unavailable: {msg}"))
                report("factcheck", f"Failed: {msg}", "error")
            timings["factcheck"] = time.monotonic() - fc_start

        briefing.sources = registry.all()
        briefing.queries = registry.queries
        briefing.errors = errors
        timings["total"] = time.monotonic() - t0
        briefing.timings = {k: round(v, 1) for k, v in timings.items()}
        briefing.usage = dict(getattr(llm, "usage", {}) or {})
        return briefing
    finally:
        if own_http:
            await http.aclose()


async def _validate(validator: CitationValidator, briefing: Briefing, modules: tuple[str, ...], timeout: float) -> None:
    """Validate with live resolution; if that times out, validate offline (registry only) so
    nothing unverified ships. Live lookups are the only awaited step, so a timeout leaves the
    briefing untouched before the offline pass."""
    try:
        await asyncio.wait_for(validator.validate(briefing, modules), timeout=timeout)
    except asyncio.TimeoutError:
        offline = CitationValidator(validator.registry, None, None, validator.drop_unsourced, report=validator.report)
        await offline.validate(briefing, modules)
