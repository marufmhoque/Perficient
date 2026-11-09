"""Fact checker: verifies every LLM-written claim against the text of the sources it cites.

Layer 1 - source integrity: every cited source resolves (citation gate) and PubMed sources are
          screened for retractions, expressions of concern and errata.
Layer 2 - claim grounding: Claude judges each claim against ONLY its cited sources' text
          (abstracts / registry records - never outside knowledge). Two deterministic checks then
          keep the judge honest:
            * the supporting quote must actually occur in the cited source text, and
            * every number in the claim must appear in the cited source text.
          A 'Supported' verdict that fails these checks is downgraded.

Registry facts copied by code (trial tables, rankings) are correct by construction and are not
re-checked; everything the LLM wrote (statements, table rows, takeaways, catalysts) is.
"""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import re
from collections.abc import Iterable

from pydantic import BaseModel, Field

from .citations import SourceRegistry
from .config import Settings
from .llm import LLM, LLMAccountError
from .schemas import (
    Benchmark,
    Briefing,
    Catalyst,
    CitedStatement,
    ClaimCheck,
    DifferentiationEntry,
    EpiFigure,
    FactCheckReport,
    FailureRow,
    MoAEntry,
    SeverityStratum,
    SourceIssue,
    TherapyLine,
    TrialRecord,
    UnmetNeed,
    Verdict,
)
from .sources.base import Http
from .sources.clinicaltrials import CTGOV, RESULTS_FIELDS
from .sources.pubmed import EUTILS, PubMedArticle, parse_efetch_xml
from .trials import parse_study

CHECKED = ("executive_summary", "module1", "module2", "module3a", "module3b", "module4")


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------


def claim_text(obj) -> str | None:
    """Canonical text of a claim-bearing object (the same function drives checking and rendering)."""
    if isinstance(obj, CitedStatement):
        return obj.text
    if isinstance(obj, EpiFigure):
        return f"{obj.metric} ({obj.region}): {obj.value}. {obj.population_note}".strip()
    if isinstance(obj, SeverityStratum):
        return f"{obj.severity}: {obj.share} of patients (defined by {obj.definition})"
    if isinstance(obj, UnmetNeed):
        return f"{obj.domain}: {obj.current_limitation}; clinical impact: {obj.clinical_impact}"
    if isinstance(obj, TherapyLine):
        return f"{obj.line} for {obj.setting}: {', '.join(obj.therapies)}. {obj.rationale}"
    if isinstance(obj, MoAEntry):
        return f"{obj.target} ({obj.therapy_class}, e.g. {', '.join(obj.example_agents)}): {obj.mechanism}. {obj.downstream_effects}"
    if isinstance(obj, Benchmark):
        return f"{obj.therapy} ({obj.drug_class}; target {obj.primary_target}): {obj.efficacy_benchmark}. Safety: {obj.safety_warnings}"
    if isinstance(obj, DifferentiationEntry):
        return f"{obj.asset} ({obj.modality}; {obj.target_moa}): {obj.comparator_evidence}"
    if isinstance(obj, FailureRow):
        what = ("did not meet its primary efficacy endpoint" if obj.evidence == "primary endpoint p-value"
                else f"was {obj.status.lower()}")
        return f"{obj.asset} ({obj.nct_id}) {what}; cause categorized as {obj.failure_category.lower()}. {obj.takeaway}".strip()
    if isinstance(obj, Catalyst):
        return f"{obj.asset} ({obj.nct_id}, {obj.phase}, primary completion {obj.primary_completion}): {obj.why_it_matters}"
    return None


def claim_key(obj) -> str | None:
    text = claim_text(obj)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12] if text else None


class Claim(BaseModel):
    key: str
    location: str
    module: str
    text: str
    source_ids: list[str]


def iter_claims(briefing: Briefing, modules: Iterable[str] = CHECKED) -> list[Claim]:
    out: list[Claim] = []
    seen: set[str] = set()

    def walk(obj, loc: str, module: str) -> None:
        if isinstance(obj, BaseModel):
            if getattr(type(obj), "SKIP_TEXT_SCAN", False):
                return
            text = claim_text(obj)
            if text is not None:
                key = claim_key(obj)
                if key and key not in seen:
                    seen.add(key)
                    out.append(Claim(key=key, location=loc, module=module, text=text, source_ids=list(getattr(obj, "source_ids", []))))
                return
            for name in type(obj).model_fields:
                walk(getattr(obj, name), f"{loc}.{name}", module)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                walk(item, f"{loc}[{i}]", module)

    for m in modules:
        walk(getattr(briefing, m, None), m, m)
    return out


# ---------------------------------------------------------------------------
# Evidence (text of every source a claim may be checked against)
# ---------------------------------------------------------------------------


def article_evidence(a: PubMedArticle) -> str:
    types = ", ".join(a.pub_types)
    return f"Title: {a.title}\nJournal: {a.journal} {a.year}\nPublication types: {types}\nAbstract: {a.abstract or '(no abstract available)'}"


def trial_evidence(t: TrialRecord) -> str:
    """Registry facts only - never the LLM's modality/target classification."""
    lines = [
        f"ClinicalTrials.gov record {t.nct_id}{f' ({t.acronym})' if t.acronym else ''}: {t.title}",
        f"Status: {t.status.replace('_', ' ').lower()}; Phase: {t.phase_label}; Sponsor: {t.sponsor} ({(t.sponsor_class or '').lower()})",
        f"Conditions: {', '.join(t.conditions)}",
        f"Start date: {t.start_date or 'n/a'}; Primary completion: {t.primary_completion_date or 'n/a'}; Enrollment: {t.enrollment or 'n/a'}",
    ]
    for a in t.arms:
        lines.append(f"Arm [{a.type or '?'}] {a.label}: {', '.join(a.intervention_names)}" + (f" - {a.description}" if a.description else ""))
    if t.primary_outcomes:
        lines.append("Primary outcomes: " + "; ".join(t.primary_outcomes))
    if t.why_stopped:
        lines.append(f"Why stopped: {t.why_stopped}")
    for r in t.primary_results[:6]:
        lines.append(f"Primary endpoint result: {r.outcome} - {r.comparison}: p={r.p_value_raw}")
    if t.officials:
        lines.append("Investigators / officials: " + "; ".join(
            f"{o.name} ({(o.role or '').replace('_', ' ').lower()}{', ' + o.affiliation if o.affiliation else ''})" for o in t.officials[:8]))
    if t.facilities:
        shown = "; ".join(f"{f.name}, {f.country or '?'}" for f in t.facilities[:25])
        lines.append(f"Sites ({t.n_sites or len(t.facilities)} total{', first 25' if (t.n_sites or 0) > 25 else ''}): {shown}")
    return "\n".join(lines)


DERIVED_KEY = "DERIVED"
DERIVED_MODULES = {"module4", "executive_summary", "module3a"}


def derived_tables(briefing: Briefing) -> str:
    """Counts computed by code over ALL retrieved records (rankings, pipeline tallies). Claims about these
    aggregates cite only a few example records, so the judge needs the tables to verify them."""
    from collections import Counter

    lines: list[str] = []
    m3a, m4 = briefing.module3a, briefing.module4
    if m3a and m3a.trials:
        lines.append(f"Active interventional trials retrieved: {len(m3a.trials)} (of {m3a.total_found} found).")
        lines.append("By phase: " + ", ".join(f"{k} {v}" for k, v in Counter(t.phase_label for t in m3a.trials).most_common()))
        lines.append("By comparator: " + ", ".join(f"{k} {v}" for k, v in Counter(t.comparator_class for t in m3a.trials).most_common()))
        lines.append("By sponsor type: " + ", ".join(f"{(k or '?').lower()} {v}" for k, v in Counter(t.sponsor_class for t in m3a.trials).most_common()))
    if m4:
        if m4.sponsors:
            lines.append("Sponsor ranking (active Phase 2/3 trials, all active trials): " + "; ".join(
                f"{s.sponsor} {s.active_ph2_3}, {s.active_total}" for s in m4.sponsors))
        if m4.centers:
            lines.append("Trial centers (number of active trials): " + "; ".join(
                f"{c.name} ({c.country or '?'}) {c.trial_count}" for c in m4.centers))
        if m4.kols:
            lines.append("Key opinion leaders (trials as investigator, key guideline/treatment publications, roles, affiliation): " + "; ".join(
                f"{k.name} - {k.trial_count} trials, {k.publication_count} publications, {', '.join(k.roles)}, {k.affiliation or 'n/a'}"
                for k in m4.kols))
    return "\n".join(lines)


class EvidenceStore:
    def __init__(self) -> None:
        self.text: dict[str, str] = {}
        self.pub_types: dict[str, list[str]] = {}

    def add_articles(self, articles: Iterable[PubMedArticle]) -> None:
        for a in articles:
            if a.source_key:
                self.text[a.source_key] = article_evidence(a)
                self.pub_types[a.source_key] = list(a.pub_types)

    def add_trials(self, trials: Iterable[TrialRecord]) -> None:
        for t in trials:
            if t.source_key and t.source_key not in self.text:
                self.text[t.source_key] = trial_evidence(t)

    async def fill_missing(self, keys: Iterable[str], registry: SourceRegistry, http: Http, settings: Settings) -> None:
        """Fetch text for cited sources not already in memory (e.g. a saved or imported report)."""
        missing = [k for k in dict.fromkeys(keys) if k not in self.text and registry.get(k)]
        pmids = {registry.get(k).native_id: k for k in missing if registry.get(k).kind == "pubmed"}
        ncts = {registry.get(k).native_id.upper(): k for k in missing if registry.get(k).kind == "trial"}
        ids = list(pmids)
        for i in range(0, len(ids), 100):
            params = {"db": "pubmed", "retmode": "xml", "id": ",".join(ids[i : i + 100]), "tool": "biointel-prototype"}
            if settings.ncbi_api_key:
                params["api_key"] = settings.ncbi_api_key
            for a in parse_efetch_xml(await http.get_text(EUTILS + "efetch.fcgi", params)):
                if a.pmid in pmids:
                    a.source_key = pmids[a.pmid]
                    self.add_articles([a])
        nct_ids = list(ncts)
        for i in range(0, len(nct_ids), 50):
            chunk = nct_ids[i : i + 50]
            data = await http.get_json(CTGOV, {"filter.ids": ",".join(chunk), "fields": ",".join(RESULTS_FIELDS), "pageSize": len(chunk)})
            for raw in data.get("studies", []) or []:
                t = parse_study(raw)
                if t.nct_id.upper() in ncts:
                    t.source_key = ncts[t.nct_id.upper()]
                    self.add_trials([t])
        for k in missing:  # SEC filings and anything unfetchable: metadata only
            if k not in self.text and (s := registry.get(k)):
                self.text[k] = f"{s.title}. {s.detail} (metadata only - full text not retrieved)"


# ---------------------------------------------------------------------------
# Deterministic checks
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"(?<![\w.\-/])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?![\w])")


def _norm_text(s: str) -> str:
    s = s.lower().replace("–", "-").replace("—", "-").replace("’", "'").replace("“", '"').replace("”", '"')
    s = s.replace(" ", " ").replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()


_SPACED_THOUSANDS_RE = re.compile(r"(?<=\d)[   \xa0](?=\d{3}(?!\d))")
_DOTTED_THOUSANDS_RE = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d{3})+)(?![\d.])")


def numbers_in(text: str) -> set[str]:
    # "100 000" / "100 000" (thin-space thousands) -> "100000"
    text = _SPACED_THOUSANDS_RE.sub("", text)
    out = set()
    for m in _NUM_RE.finditer(text):
        whole = m.group(1).replace(",", "") + (m.group(2) or "")
        out.add(whole)
    # "100.000" may be a European thousands separator: accept both readings
    for m in _DOTTED_THOUSANDS_RE.finditer(text):
        out.add(m.group(1).replace(".", ""))
    return out


def _rounds_to(claim_num: str, source_nums: set[str]) -> bool:
    """'71' matches a source 70.8 ('near 71%'): rounding at the claim's own precision is not an error."""
    decimals = len(claim_num.split(".")[1]) if "." in claim_num else 0
    target = float(claim_num)
    for s in source_nums:
        try:
            if round(float(s), decimals) == target:
                return True
        except ValueError:
            continue
    return False


def unmatched_numbers(claim: str, evidence: str) -> list[str]:
    """Numbers in the claim (ignoring small integers) that appear in none of the cited sources,
    allowing for rounding."""
    have = numbers_in(evidence)
    missing = []
    for n in sorted(numbers_in(claim)):
        if "." not in n and int(n) < 10:
            continue  # 'IL-17', 'week 4', '2-3%' style small integers are too ambiguous to check
        if n not in have and not _rounds_to(n, have) and n not in missing:
            missing.append(n)
    return missing


def quote_in_text(quote: str, text: str) -> bool:
    """True if the quote occurs in the text, tolerating whitespace / punctuation / casing drift."""
    q, t = _norm_text(quote), _norm_text(text)
    if not q:
        return False
    if q in t:
        return True
    # word tokens without punctuation, so '71.6 %' and '71.6%' or '10–16' and '10-16' compare equal
    qw, tw = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", q), re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", t)
    if not qw:
        return False
    m = difflib.SequenceMatcher(None, qw, tw, autojunk=False).find_longest_match(0, len(qw), 0, len(tw))
    return m.size >= min(8, len(qw))


def source_issues(registry: SourceRegistry, evidence: EvidenceStore, cited: set[str]) -> list[SourceIssue]:
    flags = {
        "Retracted Publication": ("critical", "Publication has been RETRACTED"),
        "Expression of Concern": ("warning", "Journal issued an expression of concern"),
        "Retraction of Publication": ("warning", "Source is a retraction notice, not primary evidence"),
        "Published Erratum": ("info", "Source is an erratum notice"),
    }
    out: list[SourceIssue] = []
    for key in sorted(cited):
        src = registry.get(key)
        if src is None:
            continue
        for pt in evidence.pub_types.get(key, []):
            if pt in flags:
                sev, msg = flags[pt]
                out.append(SourceIssue(key=key, severity=sev, issue=msg))
        if src.kind == "pubmed" and key in evidence.text and "(no abstract available)" in evidence.text[key]:
            out.append(SourceIssue(key=key, severity="info", issue="No abstract available - claims citing it were judged on the title only"))
        if src.kind == "sec":
            out.append(SourceIssue(key=key, severity="info", issue="SEC filing cited by metadata only (full text not retrieved)"))
        if key not in evidence.text:
            out.append(SourceIssue(key=key, severity="warning", issue="Source text could not be retrieved for fact checking"))
    return out


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------


class ClaimJudgement(BaseModel):
    claim_id: str = Field(description="The claim id exactly as given, e.g. C7")
    verdict: Verdict
    rationale: str = Field(description="One sentence; name any discrepancy (number, timepoint, population, drug, direction)")
    evidence_quote: str = Field(description="Verbatim excerpt (max 40 words) copied from ONE cited source; empty if none")
    evidence_source: str = Field(description="Key of the source the quote comes from (e.g. P3); empty if none")


class FactCheckBatch(BaseModel):
    judgements: list[ClaimJudgement]


JUDGE_SYSTEM = """You are a meticulous biomedical fact-checker. For each CLAIM, decide whether the text of
ITS cited sources (only the keys listed for that claim) supports it. Use no outside knowledge - a claim
that is true in general but not stated in its cited sources is not supported by them.

Verdicts:
- Supported: the claim's factual content is stated in the cited source text. Faithful paraphrase,
  reasonable rounding (70.8% -> "about 71%") and omitting secondary details all count as Supported.
- Partially supported: use ONLY when a substantive element - a number, timepoint, population, drug,
  trial design or direction of effect - is missing from or conflicts with the sources, while the core
  assertion is supported. Do not downgrade for minor wording, unstated confidence intervals, or
  analytical framing.
- Not supported: the cited sources contradict the claim or contain nothing that supports it.
- Cannot verify: the available source text is too thin to judge (title only, metadata only).

A [DERIVED] block, when present, holds counts computed by code over ALL retrieved records (sponsor
rankings, site and investigator tallies, pipeline counts). Claims about those aggregates may be judged
against it (use evidence_source "DERIVED"), because the cited records are only examples.

Analytical or forward-looking wording (e.g. "suggests an opportunity") is acceptable when the underlying
facts are supported; judge the facts. Copy evidence_quote character-for-character from the source.
Return one judgement per claim, using its claim id."""


class FactChecker:
    def __init__(self, llm: LLM, settings: Settings, registry: SourceRegistry, evidence: EvidenceStore) -> None:
        self.llm = llm
        self.settings = settings
        self.registry = registry
        self.evidence = evidence
        self.derived = ""  # code-computed aggregate tables (set per check from the briefing)

    def _uses_derived(self, claim: Claim) -> bool:
        return bool(self.derived) and claim.module in DERIVED_MODULES

    def _evidence_for(self, claim: Claim) -> str:
        text = "\n".join(self.evidence.text.get(k, "") for k in claim.source_ids)
        return text + ("\n" + self.derived if self._uses_derived(claim) else "")

    def _source_text(self, claim: Claim, key: str) -> str | None:
        """Text the judge may quote for this claim: its cited sources, plus DERIVED where allowed."""
        if key == DERIVED_KEY:
            return self.derived if self._uses_derived(claim) else None
        return self.evidence.text.get(key, "") if key in claim.source_ids else None

    def _batches(self, claims: list[Claim], max_claims: int = 25, max_chars: int = 60000) -> list[list[Claim]]:
        batches: list[list[Claim]] = []
        cur: list[Claim] = []
        keys: set[str] = set()
        size = 0
        for c in claims:
            new = [k for k in c.source_ids if k not in keys]
            add = sum(len(self.evidence.text.get(k, "")[: self.settings.abstract_char_limit + 600]) for k in new)
            if cur and (len(cur) >= max_claims or size + add > max_chars):
                batches.append(cur)
                cur, keys, size = [], set(), 0
                new = list(c.source_ids)
                add = sum(len(self.evidence.text.get(k, "")[: self.settings.abstract_char_limit + 600]) for k in new)
            cur.append(c)
            keys.update(new)
            size += add
        if cur:
            batches.append(cur)
        return batches

    async def _judge(self, batch: list[Claim]) -> dict[str, ClaimJudgement]:
        keys = list(dict.fromkeys(k for c in batch for k in c.source_ids))
        limit = self.settings.abstract_char_limit + 600
        src_block = "\n\n".join(f"[{k}]\n{self.evidence.text[k][:limit]}" for k in keys if k in self.evidence.text)
        if any(self._uses_derived(c) for c in batch):
            src_block += f"\n\n[{DERIVED_KEY}] Counts computed by code over all retrieved records (context, not a citation):\n{self.derived}"
        ids = {f"C{i + 1}": c for i, c in enumerate(batch)}
        claim_block = "\n".join(
            f"{cid} (cites {', '.join(c.source_ids)}{'; DERIVED counts available' if self._uses_derived(c) else ''}): {c.text}"
            for cid, c in ids.items()
        )
        out: FactCheckBatch = await self.llm.parse(
            model=self.settings.fact_check_model or self.settings.agent_model, effort=self.settings.fact_check_effort,
            output_format=FactCheckBatch, system=JUDGE_SYSTEM,
            user=f"SOURCES:\n{src_block}\n\nCLAIMS:\n{claim_block}\n\nJudge every claim.",
        )
        return {ids[j.claim_id.strip()].key: j for j in out.judgements if j.claim_id.strip() in ids}

    def _finalize(self, c: Claim, j: ClaimJudgement | None) -> ClaimCheck:
        evidence = self._evidence_for(c)
        if j is None:
            why = ("No text is available for the cited sources." if not evidence.strip()
                   else "The fact checker returned no judgement for this claim.")
            return ClaimCheck(claim_key=c.key, location=c.location, module=c.module, claim=c.text, source_ids=c.source_ids,
                              verdict="Cannot verify", rationale=why)
        verdict: Verdict = j.verdict
        notes = [j.rationale.strip()]
        quotable = self._source_text(c, j.evidence_source.strip())
        quote_ok = bool(j.evidence_quote.strip()) and quotable is not None and quote_in_text(j.evidence_quote, quotable)
        missing = unmatched_numbers(c.text, evidence)
        if verdict in ("Supported", "Partially supported") and j.evidence_quote.strip() and not quote_ok:
            verdict = "Cannot verify"
            notes.append("Checker's supporting quote was not found in the cited source, so the verdict was not accepted.")
        if verdict == "Supported" and missing:
            verdict = "Partially supported"
            notes.append(f"Number(s) {', '.join(missing)} not found in the cited source text.")
        return ClaimCheck(
            claim_key=c.key, location=c.location, module=c.module, claim=c.text, source_ids=c.source_ids, verdict=verdict,
            rationale=" ".join(n for n in notes if n), evidence_quote=j.evidence_quote.strip(),
            evidence_source=j.evidence_source.strip(), quote_verified=quote_ok, unmatched_numbers=missing,
        )

    async def check(self, briefing: Briefing, modules: Iterable[str] = CHECKED) -> list[ClaimCheck]:
        claims = iter_claims(briefing, modules)
        if not claims:
            return []
        self.derived = derived_tables(briefing)
        judgeable = [c for c in claims if any(k in self.evidence.text for k in c.source_ids) or self._uses_derived(c)]
        results: dict[str, ClaimJudgement] = {}
        sem = asyncio.Semaphore(6)

        async def run(batch: list[Claim]) -> None:
            async with sem:
                results.update(await self._judge(batch))

        outcomes = await asyncio.gather(*(run(b) for b in self._batches(judgeable)), return_exceptions=True)
        for o in outcomes:
            if isinstance(o, LLMAccountError):
                raise o
        failed = [o for o in outcomes if isinstance(o, BaseException)]
        checks = [self._finalize(c, results.get(c.key)) for c in claims]
        if failed and not results:
            raise RuntimeError(f"Fact check failed: {failed[0]}")
        return checks

    def report(self, checks: list[ClaimCheck], cited: set[str]) -> FactCheckReport:
        return FactCheckReport(
            claims=checks, source_issues=source_issues(self.registry, self.evidence, cited),
            sources_with_text=sum(1 for k in cited if k in self.evidence.text),
            model=self.settings.fact_check_model or self.settings.agent_model,
        )


def rekey(checks: list[ClaimCheck], briefing: Briefing) -> list[ClaimCheck]:
    """Point each result at the claim's *current* text (looked up by location), so renderers find it
    even if a later step (terminology reconciliation) edited the wording."""
    current = {c.location: c for c in iter_claims(briefing)}
    for chk in checks:
        cur = current.get(chk.location)
        if cur is not None:
            chk.claim_key, chk.claim = cur.key, cur.text
    return checks


async def fact_check_briefing(briefing: Briefing, settings: Settings, llm: LLM, http: Http | None = None) -> FactCheckReport:
    """Fact-check a saved / imported report: re-fetch the cited sources' text, then judge every claim."""
    from .citations import cited_keys

    registry = SourceRegistry.from_sources(briefing.sources)
    evidence = EvidenceStore()
    own = http is None
    http = http or Http(settings)
    try:
        cited = {k for k in cited_keys(briefing) if registry.get(k)}
        await evidence.fill_missing(cited, registry, http, settings)
        checker = FactChecker(llm, settings, registry, evidence)
        return checker.report(await checker.check(briefing), cited)
    finally:
        if own:
            await http.aclose()
