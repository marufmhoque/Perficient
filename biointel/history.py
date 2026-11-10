"""Prior searches & saved reports.

The library is process-wide: the Streamlit app keeps its *state* in ``st.cache_resource`` so every
viewer shares one library. It resets when the app process restarts - reports can be downloaded as
JSON and re-imported. ``HistoryStore`` is the seam for a persistent backend later.

Reload safety
-------------
Streamlit Cloud applies code updates by re-executing changed modules *without* restarting the
process, which re-creates every class. State cached across reruns must therefore never hold model
instances or code: it holds only plain data (JSON text, dicts, strings). ``InMemoryHistory`` is a
thin facade built fresh on every run around that state, so it always uses the current classes.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any, Protocol

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


def new_state() -> dict[str, Any]:
    """Plain-data state to cache across reruns (only builtins - no model instances, no code)."""
    return {
        "lock": threading.RLock(),
        "searches": [],  # list of SearchEntry dumps (JSON-mode dicts), oldest first
        "reports": {},  # id -> {"json": str, "meta": dict}; insertion-ordered, oldest first
        "example_files": {},  # bundled sample file name -> mtime already loaded
        "example_errors": {},  # bundled sample file name -> load error
    }


class InMemoryHistory:
    """Facade over a plain-data state dict. Build a new one per run; share the state."""

    def __init__(self, max_reports: int = 50, max_searches: int = 25, state: dict[str, Any] | None = None) -> None:
        self.max_reports = max_reports
        self.max_searches = max_searches
        self.state = state if state is not None else new_state()
        self._lock = self.state["lock"]
        self._parsed: dict[str, Briefing] = {}  # per-facade (per-run) parse cache

    # bundled-example bookkeeping lives in the shared state too
    @property
    def example_files(self) -> dict[str, float]:
        return self.state["example_files"]

    @property
    def example_errors(self) -> dict[str, str]:
        return self.state["example_errors"]

    # searches ------------------------------------------------------------
    def add_search(self, inp: ConditionInput, report_id: str | None = None) -> None:
        entry = SearchEntry(input=inp.model_dump(mode="json"), report_id=report_id).model_dump(mode="json")
        key = entry["input"]
        with self._lock:
            searches = [s for s in self.state["searches"] if s["input"] != key]
            searches.append(entry)
            self.state["searches"] = searches[-self.max_searches :]

    def recent_searches(self, n: int = 10) -> list[SearchEntry]:
        with self._lock:
            raw = list(reversed(self.state["searches"][-n:]))
        out = []
        for s in raw:
            try:
                out.append(SearchEntry.model_validate(s))
            except Exception:  # an entry written by an incompatible older version: skip it
                continue
        return out

    # reports ---------------------------------------------------------------
    def save_report(self, briefing: Briefing) -> None:
        record = {"json": briefing.model_dump_json(), "meta": report_meta(briefing).model_dump(mode="json")}
        with self._lock:
            reports = self.state["reports"]
            reports.pop(briefing.id, None)
            reports[briefing.id] = record
            self._parsed.pop(briefing.id, None)
            # evict oldest non-example reports beyond the cap
            while len(reports) > self.max_reports:
                victim = next((rid for rid, r in reports.items() if not r["meta"].get("is_example")), None)
                if victim is None:
                    break
                reports.pop(victim)

    def list_reports(self) -> list[ReportMeta]:
        with self._lock:
            metas = [r["meta"] for r in self.state["reports"].values()]
        out = []
        for m in metas:
            try:
                out.append(ReportMeta.model_validate(m))
            except Exception:
                continue
        return sorted(out, key=lambda r: r.created_at, reverse=True)

    def get_report(self, report_id: str) -> Briefing | None:
        if report_id in self._parsed:
            return self._parsed[report_id]
        with self._lock:
            record = self.state["reports"].get(report_id)
        if record is None:
            return None
        try:
            briefing = Briefing.model_validate_json(record["json"])
        except Exception:
            return None
        self._parsed[report_id] = briefing
        return briefing

    def delete_report(self, report_id: str) -> bool:
        with self._lock:
            self._parsed.pop(report_id, None)
            return self.state["reports"].pop(report_id, None) is not None

    def __len__(self) -> int:
        return len(self.state["reports"])


def report_to_json(b: Briefing) -> str:
    return b.model_dump_json(indent=2)


def report_from_json(text: str | bytes) -> Briefing:
    return Briefing.model_validate_json(text)
