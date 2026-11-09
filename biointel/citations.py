"""SourceRegistry: the single source of truth for everything the system may cite.

Every record actually retrieved from an API is registered here and given a short key
(P# PubMed, T# trial, S# SEC filing). LLM outputs may only cite these keys; the validator
resolves anything else (raw PMIDs / NCT IDs) back to keys or removes it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from .schemas import QueryLog, Source, SourceKind

_KEY_RE = re.compile(r"^\[?\s*([PTS])\s*(\d+)\s*\]?$", re.I)
_NCT_RE = re.compile(r"NCT\d{8}", re.I)
_PMID_RE = re.compile(r"^(?:PMID\s*:?\s*)?(\d{1,9})$", re.I)

_PREFIX: dict[SourceKind, str] = {"pubmed": "P", "trial": "T", "sec": "S"}


class SourceRegistry:
    def __init__(self) -> None:
        self._by_key: dict[str, Source] = {}
        self._by_native: dict[tuple[str, str], str] = {}
        self._counters: dict[str, int] = {"P": 0, "T": 0, "S": 0}
        self.queries: list[QueryLog] = []

    # -- registration -----------------------------------------------------
    def add(self, kind: SourceKind, native_id: str, title: str, url: str, detail: str = "") -> Source:
        native = native_id.strip().upper() if kind == "trial" else native_id.strip()
        existing = self._by_native.get((kind, native))
        if existing:
            return self._by_key[existing]
        prefix = _PREFIX[kind]
        self._counters[prefix] += 1
        key = f"{prefix}{self._counters[prefix]}"
        src = Source(key=key, kind=kind, native_id=native, title=title.strip() or native, url=url, detail=detail)
        self._by_key[key] = src
        self._by_native[(kind, native)] = key
        return src

    def log_query(self, source: str, purpose: str, query: str, url: str, n_results: int) -> None:
        self.queries.append(QueryLog(source=source, purpose=purpose, query=query, url=url, n_results=n_results))

    # -- lookup -----------------------------------------------------------
    def get(self, key: str) -> Source | None:
        return self._by_key.get(key)

    def key_for(self, kind: SourceKind, native_id: str) -> str | None:
        native = native_id.strip().upper() if kind == "trial" else native_id.strip()
        return self._by_native.get((kind, native))

    def resolve(self, ref: str) -> str | None:
        """Map a citation reference (key, PMID, NCT ID) to a registered key, or None."""
        if not ref:
            return None
        ref = ref.strip().strip(",;.")
        m = _KEY_RE.match(ref)
        if m:
            key = f"{m.group(1).upper()}{int(m.group(2))}"
            return key if key in self._by_key else None
        nct = _NCT_RE.search(ref)
        if nct:
            return self.key_for("trial", nct.group(0).upper())
        m = _PMID_RE.match(ref)
        if m:
            return self.key_for("pubmed", m.group(1))
        # SEC accession (0000950170-22-004442) or URL
        for (kind, native), key in self._by_native.items():
            if kind == "sec" and (ref == native or ref in self._by_key[key].url):
                return key
        return None

    def normalize(self, refs: Iterable[str]) -> tuple[list[str], list[str]]:
        """Return (valid keys de-duplicated in order, invalid refs)."""
        valid: list[str] = []
        invalid: list[str] = []
        for ref in refs:
            key = self.resolve(ref)
            if key:
                if key not in valid:
                    valid.append(key)
            else:
                invalid.append(ref)
        return valid, invalid

    def by_kind(self, kind: SourceKind) -> list[Source]:
        return [s for s in self._by_key.values() if s.kind == kind]

    def all(self) -> list[Source]:
        return list(self._by_key.values())

    def __len__(self) -> int:
        return len(self._by_key)

    def __contains__(self, key: str) -> bool:
        return key in self._by_key


def cited_keys(obj) -> set[str]:
    """Collect every source id cited anywhere inside a pydantic object tree."""
    from pydantic import BaseModel

    found: set[str] = set()

    def walk(o) -> None:
        if isinstance(o, BaseModel):
            for name in type(o).model_fields:
                val = getattr(o, name)
                if name in ("source_ids", "sec_source_ids") and isinstance(val, list):
                    found.update(val)
                elif name == "source_key" and isinstance(val, str) and val:
                    found.add(val)
                else:
                    walk(val)
        elif isinstance(o, (list, tuple)):
            for item in o:
                walk(item)

    walk(obj)
    return found
