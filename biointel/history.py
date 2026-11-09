"""Prior searches & saved reports.

``InMemoryHistory`` is process-wide: the Streamlit app wraps it in ``st.cache_resource`` so every
viewer shares one library. It resets when the app process restarts - reports can be downloaded as
JSON and re-imported. ``HistoryStore`` is the seam for a persistent backend later.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Protocol

from pydantic import BaseModel, Field

from .schemas import Briefing, ConditionInput, utcnow


class SearchEntry(BaseModel):
    input: ConditionInput
    at: datetime = Field(default_factory=utcnow)
    report_id: str | None = None

    @property
    def label(self) -> str:
        return self.input.label()


class ReportMeta(BaseModel):
    id: str
    title: str
    icd10: str | None
    mesh: str | None
    created_at: datetime
    duration_s: float | None
    n_trials: int
    n_failures: int
    n_pubmed: int
    n_sources: int
    citations_valid: str
    is_example: bool = False


class HistoryStore(Protocol):
    def add_search(self, inp: ConditionInput, report_id: str | None = None) -> None: ...
    def recent_searches(self, n: int = 10) -> list[SearchEntry]: ...
    def save_report(self, briefing: Briefing) -> None: ...
    def list_reports(self) -> list[ReportMeta]: ...
    def get_report(self, report_id: str) -> Briefing | None: ...
    def delete_report(self, report_id: str) -> bool: ...


def report_meta(b: Briefing) -> ReportMeta:
    return ReportMeta(
        id=b.id,
        title=b.condition.display_name,
        icd10=b.condition.icd10_code,
        mesh=b.condition.mesh_id,
        created_at=b.created_at,
        duration_s=b.timings.get("total"),
        n_trials=len(b.module3a.trials) if b.module3a else 0,
        n_failures=len(b.module3b.failures) if b.module3b else 0,
        n_pubmed=sum(1 for s in b.sources if s.kind == "pubmed"),
        n_sources=len(b.sources),
        citations_valid=f"{b.validation.ids_valid}/{b.validation.ids_checked}",
        is_example=b.is_example,
    )


class InMemoryHistory:
    def __init__(self, max_reports: int = 50, max_searches: int = 25) -> None:
        self.max_reports = max_reports
        self.max_searches = max_searches
        self._lock = threading.Lock()
        self._searches: list[SearchEntry] = []
        self._reports: dict[str, Briefing] = {}  # insertion-ordered; newest last

    # searches ------------------------------------------------------------
    def add_search(self, inp: ConditionInput, report_id: str | None = None) -> None:
        with self._lock:
            key = inp.model_dump_json()
            self._searches = [s for s in self._searches if s.input.model_dump_json() != key]
            self._searches.append(SearchEntry(input=inp, report_id=report_id))
            del self._searches[: max(0, len(self._searches) - self.max_searches)]

    def recent_searches(self, n: int = 10) -> list[SearchEntry]:
        with self._lock:
            return list(reversed(self._searches[-n:]))

    # reports ---------------------------------------------------------------
    def save_report(self, briefing: Briefing) -> None:
        with self._lock:
            self._reports.pop(briefing.id, None)
            self._reports[briefing.id] = briefing
            # evict oldest non-example reports beyond the cap
            while len(self._reports) > self.max_reports:
                victim = next((rid for rid, b in self._reports.items() if not b.is_example), None)
                if victim is None:
                    break
                self._reports.pop(victim)

    def list_reports(self) -> list[ReportMeta]:
        with self._lock:
            reports = list(self._reports.values())
        return [report_meta(b) for b in sorted(reports, key=lambda b: b.created_at, reverse=True)]

    def get_report(self, report_id: str) -> Briefing | None:
        with self._lock:
            return self._reports.get(report_id)

    def delete_report(self, report_id: str) -> bool:
        with self._lock:
            return self._reports.pop(report_id, None) is not None

    def __len__(self) -> int:
        return len(self._reports)


def report_to_json(b: Briefing) -> str:
    return b.model_dump_json(indent=2)


def report_from_json(text: str | bytes) -> Briefing:
    return Briefing.model_validate_json(text)
