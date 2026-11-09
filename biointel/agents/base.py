"""Agent base class + prompt-formatting helpers shared by all agents."""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable

from ..config import Settings
from ..llm import LLM
from ..schemas import TrialRecord
from ..sources.pubmed import PubMedArticle

# progress(stage, message)
ProgressFn = Callable[[str, str], None]


def _noop(stage: str, message: str) -> None:  # pragma: no cover - trivial
    return None


class Agent(ABC):
    name: str = "agent"
    stage: str = "agent"

    def __init__(self, llm: LLM, settings: Settings, progress: ProgressFn | None = None) -> None:
        self.llm = llm
        self.settings = settings
        self._progress = progress or _noop

    def report(self, message: str) -> None:
        self._progress(self.stage, message)


CITATION_RULES = """Citation rules (strict):
- Cite only the source keys shown in the SOURCES list (e.g. "P3", "T12"), exactly as written, in source_ids.
- Every item needs at least one source key that directly supports it. Items without support are deleted downstream.
- Never invent or recall identifiers (PMIDs, NCT numbers, DOIs) - only use the provided keys.
- Report figures exactly as the source states them (with population, region and year when given); do not extrapolate.
- If the sources do not support a field, return fewer items or an empty list."""


def format_articles(articles: list[PubMedArticle], char_limit: int) -> str:
    blocks = []
    for a in articles:
        abstract = a.abstract.strip()
        if len(abstract) > char_limit:
            abstract = abstract[:char_limit].rsplit(" ", 1)[0] + " ..."
        types = ", ".join(t for t in a.pub_types if t not in ("Journal Article",)) or "Journal Article"
        blocks.append(
            f"[{a.source_key}] {a.title}\n"
            f"  {a.journal} {a.year} | {types} | PMID {a.pmid}\n"
            f"  Abstract: {abstract or '(no abstract)'}"
        )
    return "\n\n".join(blocks)


def format_trial_line(t: TrialRecord, extra: str = "") -> str:
    bits = [
        f"[{t.source_key}] {t.nct_id}",
        t.phase_label,
        t.status.replace("_", " ").lower(),
        f"sponsor: {t.sponsor} ({(t.sponsor_class or '?').lower()})",
        f"asset: {t.asset_name}",
        f"comparator: {t.comparator_summary}",
    ]
    if t.target_moa:
        bits.append(f"target/MoA: {t.target_moa}")
    if t.modality:
        bits.append(f"modality: {t.modality}")
    if t.primary_completion_date:
        bits.append(f"primary completion: {t.primary_completion_date}")
    if extra:
        bits.append(extra)
    return " | ".join(bits)
