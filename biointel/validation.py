"""Citation gate - enforces the PRD's zero-hallucination requirement.

1. Collect every reference: ``source_ids`` lists, NCT-ID fields, and NCT / PMID mentions inside
   LLM-written text (registry facts such as TrialRecord are skipped).
2. Anything not already in the SourceRegistry is resolved live (PubMed esummary /
   ClinicalTrials.gov) - real records are registered, fakes are not.
3. Apply: normalize ids to registry keys, remove unknown ids, drop (or flag) statements left
   without any valid source, and scrub unverifiable ids from free text.
"""

from __future__ import annotations

import asyncio
import re

from pydantic import BaseModel

from .citations import SourceRegistry
from .schemas import Briefing, ValidationIssue, ValidationReport
from .sources.clinicaltrials import ClinicalTrialsAdapter
from .sources.pubmed import PubMedAdapter

NCT_TEXT_RE = re.compile(r"\bNCT\d{8}\b", re.I)
PMID_TEXT_RE = re.compile(r"\bPMID\s*:?\s*(\d{1,9})\b", re.I)
RAW_PMID_RE = re.compile(r"^(?:PMID\s*:?\s*)?(\d{1,9})$", re.I)
ANALYST_MODULES = ("module1", "module2", "module3a", "module3b", "module4")
CHECKED_MODULES = ("executive_summary", *ANALYST_MODULES)


def _skip(obj: BaseModel) -> bool:
    return bool(getattr(type(obj), "SKIP_TEXT_SCAN", False))


class CitationValidator:
    def __init__(
        self,
        registry: SourceRegistry,
        pubmed: PubMedAdapter | None = None,
        trials: ClinicalTrialsAdapter | None = None,
        drop_unsourced: bool = True,
        report: ValidationReport | None = None,
    ) -> None:
        self.registry = registry
        self.pubmed = pubmed
        self.trials = trials
        self.drop_unsourced = drop_unsourced
        self.report = report or ValidationReport()  # shared across passes so counts accumulate

    # ---------------------------------------------------------- collection
    def _collect(self, obj, pmids: set[str], ncts: set[str]) -> None:
        if isinstance(obj, BaseModel):
            if _skip(obj):
                return
            for name in type(obj).model_fields:
                val = getattr(obj, name)
                if name in ("source_ids", "sec_source_ids", "nct_ids") and isinstance(val, list):
                    for ref in val:
                        self._note_ref(ref, pmids, ncts)
                elif name == "nct_id" and isinstance(val, str):
                    self._note_ref(val, pmids, ncts)
                elif isinstance(val, str):
                    ncts.update(m.upper() for m in NCT_TEXT_RE.findall(val) if not self.registry.key_for("trial", m))
                    pmids.update(m for m in PMID_TEXT_RE.findall(val) if not self.registry.key_for("pubmed", m))
                else:
                    self._collect(val, pmids, ncts)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                self._collect(item, pmids, ncts)

    def _note_ref(self, ref: str, pmids: set[str], ncts: set[str]) -> None:
        if not isinstance(ref, str) or self.registry.resolve(ref):
            return
        nct = NCT_TEXT_RE.search(ref)
        if nct:
            ncts.add(nct.group(0).upper())
            return
        m = RAW_PMID_RE.match(ref.strip())
        if m:
            pmids.add(m.group(1))

    async def _resolve_live(self, pmids: set[str], ncts: set[str]) -> None:
        if pmids and self.pubmed:
            try:
                found = await self.pubmed.existing(sorted(pmids))
            except Exception:
                found = {}
            for pmid, doc in found.items():
                title = doc.get("title") or f"PMID {pmid}"
                detail = f"{doc.get('source', '')} {doc.get('pubdate', '')[:4]}. PMID: {pmid}".strip()
                self.registry.add("pubmed", pmid, title, f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", detail)
                self.report.ids_resolved_live += 1
                self.report.issues.append(
                    ValidationIssue(location="(live check)", problem=f"PMID {pmid} was cited but not retrieved; verified on PubMed", action="resolved live")
                )
        if ncts and self.trials:
            sem = asyncio.Semaphore(5)

            async def check(nct: str):
                async with sem:
                    return nct, await self.trials.exists(nct)

            for nct, data in await asyncio.gather(*(check(n) for n in sorted(ncts)[:30])):
                if data:
                    ident = data.get("protocolSection", {}).get("identificationModule", {})
                    self.registry.add("trial", nct, ident.get("briefTitle", nct), f"https://clinicaltrials.gov/study/{nct}", "verified live")
                    self.report.ids_resolved_live += 1
                    self.report.issues.append(
                        ValidationIssue(location="(live check)", problem=f"{nct} was cited but not retrieved; verified on ClinicalTrials.gov", action="resolved live")
                    )

    # --------------------------------------------------------------- apply
    def _fix_ids(self, obj: BaseModel, field: str, location: str) -> list[str]:
        refs = getattr(obj, field) or []
        self.report.ids_checked += len(refs)
        valid, invalid = self.registry.normalize(refs)
        self.report.ids_valid += len(valid)
        for ref in invalid:
            self.report.issues.append(ValidationIssue(location=location, problem=f"Unknown source id '{ref}'", action="removed id"))
        setattr(obj, field, valid)
        return valid

    def _scrub_text(self, text: str, location: str) -> str:
        def repl_nct(m: re.Match) -> str:
            if self.registry.key_for("trial", m.group(0).upper()):
                return m.group(0)
            self.report.issues.append(ValidationIssue(location=location, problem=f"Unverifiable {m.group(0)} in text", action="removed id"))
            return "[unverified trial ID removed]"

        def repl_pmid(m: re.Match) -> str:
            if self.registry.key_for("pubmed", m.group(1)):
                return m.group(0)
            self.report.issues.append(ValidationIssue(location=location, problem=f"Unverifiable PMID {m.group(1)} in text", action="removed id"))
            return "[unverified PMID removed]"

        return PMID_TEXT_RE.sub(repl_pmid, NCT_TEXT_RE.sub(repl_nct, text))

    def _apply(self, obj, location: str) -> bool:
        """Validate obj in place. Returns False if obj should be dropped."""
        if not isinstance(obj, BaseModel) or _skip(obj):
            return True
        fields = type(obj).model_fields
        for name in fields:
            val = getattr(obj, name)
            loc = f"{location}.{name}"
            if name in ("source_ids", "sec_source_ids"):
                continue
            if name == "nct_ids" and isinstance(val, list):
                keep = [n for n in val if self.registry.key_for("trial", str(n).upper())]
                for n in val:
                    if n not in keep:
                        self.report.issues.append(ValidationIssue(location=loc, problem=f"Unknown NCT ID '{n}'", action="removed id"))
                setattr(obj, name, keep)
            elif name == "nct_id" and isinstance(val, str):
                if not self.registry.key_for("trial", val.upper()):
                    self.report.issues.append(ValidationIssue(location=location, problem=f"Row references unknown {val}", action="dropped"))
                    return False
            elif isinstance(val, str):
                setattr(obj, name, self._scrub_text(val, loc))
            elif isinstance(val, list):
                kept = []
                for i, item in enumerate(val):
                    if self._apply(item, f"{loc}[{i}]"):
                        kept.append(item)
                setattr(obj, name, kept)
            elif isinstance(val, BaseModel):
                # claim-bearing models only occur inside lists in our schemas; a nested
                # singleton is validated (ids cleaned, text scrubbed) but never removed
                self._apply(val, loc)

        if "sec_source_ids" in fields:
            self._fix_ids(obj, "sec_source_ids", location)
        if "source_ids" in fields:
            self.report.statements_checked += 1
            valid = self._fix_ids(obj, "source_ids", location)
            if not valid:
                if self.drop_unsourced:
                    self.report.statements_dropped += 1
                    self.report.issues.append(ValidationIssue(location=location, problem="No valid supporting source", action="dropped"))
                    return False
                self.report.issues.append(ValidationIssue(location=location, problem="No valid supporting source", action="flagged"))
        return True

    async def validate(self, briefing: Briefing, modules: tuple[str, ...] = CHECKED_MODULES) -> ValidationReport:
        """Validate the named parts of the briefing in place. Call once per part: the
        orchestrator validates analyst modules before synthesis, then the executive summary."""
        pmids: set[str] = set()
        ncts: set[str] = set()
        for name in modules:
            self._collect(getattr(briefing, name), pmids, ncts)
        await self._resolve_live(pmids, ncts)  # the only await: a timeout here leaves the briefing untouched
        for name in modules:
            module = getattr(briefing, name)
            if module is not None:
                self._apply(module, name)
        return self.report
