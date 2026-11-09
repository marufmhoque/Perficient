"""Condition normalization: ICD-10 (NLM Clinical Tables) and MeSH (NCBI E-utilities)."""

from __future__ import annotations

import re

from ..config import Settings
from ..schemas import ConditionInput, ResolvedCondition
from .base import Http, SourceError

ICD10_API = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


def mesh_uid_to_id(uid: str) -> str:
    """NCBI MeSH UID 68011565 -> descriptor D011565 (67xxxxxx -> supplementary C...)."""
    if uid.startswith("68"):
        return "D" + uid[2:]
    if uid.startswith("67"):
        return "C" + uid[2:]
    return uid


def mesh_id_to_uid(mesh_id: str) -> str:
    mesh_id = mesh_id.strip().upper()
    if mesh_id.startswith("D"):
        return "68" + mesh_id[1:]
    if mesh_id.startswith("C"):
        return "67" + mesh_id[1:]
    return mesh_id


_CONNECTOR_TAIL = {"with", "without", "of", "and", "or", "in", "due", "to", "for", "the", "a", "an", "other", "unspecified"}


def category_name(names: list[str]) -> str | None:
    """Common leading words of sub-code names: ['T2DM with coma', 'T2DM with ketoacidosis'] -> 'T2DM'."""
    if not names:
        return None
    split = [n.replace(",", " ,").split() for n in names]
    prefix: list[str] = []
    for words in zip(*split):
        if all(w.lower() == words[0].lower() for w in words):
            prefix.append(words[0])
        else:
            break
    while prefix and (prefix[-1].lower() in _CONNECTOR_TAIL or prefix[-1] == ","):
        prefix.pop()
    return " ".join(prefix).replace(" ,", ",") or None


async def _icd10_lookup(http: Http, code: str) -> tuple[str | None, bool]:
    """Return (name, exact). For a category code (e.g. E11) derive the shared name of its sub-codes."""
    data = await http.get_json(ICD10_API, {"sf": "code,name", "terms": code, "maxList": 50})
    rows = data[3] if isinstance(data, list) and len(data) > 3 and data[3] else []
    for row_code, name in rows:
        if row_code.upper() == code.upper():
            return name, True
    children = [name for row_code, name in rows if row_code.upper().startswith(code.upper())]
    return category_name(children), False


async def _icd10_suggest(http: Http, name: str) -> tuple[str, str] | None:
    data = await http.get_json(ICD10_API, {"sf": "code,name", "terms": name, "maxList": 5})
    rows = data[3] if isinstance(data, list) and len(data) > 3 and data[3] else []
    return (rows[0][0], rows[0][1]) if rows else None


async def _mesh_search(http: Http, settings: Settings, term: str) -> str | None:
    params = {"db": "mesh", "retmode": "json", "retmax": 3, "term": term, "tool": "biointel-prototype"}
    if settings.ncbi_api_key:
        params["api_key"] = settings.ncbi_api_key
    data = await http.get_json(EUTILS + "esearch.fcgi", params)
    ids = data.get("esearchresult", {}).get("idlist", [])
    return ids[0] if ids else None


async def _mesh_summary(http: Http, settings: Settings, uid: str) -> dict | None:
    params = {"db": "mesh", "retmode": "json", "id": uid, "tool": "biointel-prototype"}
    if settings.ncbi_api_key:
        params["api_key"] = settings.ncbi_api_key
    data = await http.get_json(EUTILS + "esummary.fcgi", params)
    doc = data.get("result", {}).get(uid)
    return doc if isinstance(doc, dict) and not doc.get("error") else None


def _candidates(name: str) -> list[str]:
    """Exact heading first, then entry-term search, then sub-phrases as exact headings."""
    words = [w for w in re.split(r"[\s,/]+", name.strip()) if w]
    cands = [f'"{name}"[MH]', name]
    for size in range(len(words) - 1, 0, -1):
        for i in range(0, len(words) - size + 1):
            sub = " ".join(words[i : i + size])
            if len(sub) >= 4:
                cands.append(f'"{sub}"[MH]')
    seen: list[str] = []
    for c in cands:
        if c not in seen:
            seen.append(c)
    return seen


async def resolve_condition(inp: ConditionInput, http: Http, settings: Settings) -> ResolvedCondition:
    notes: list[str] = []
    name = (inp.condition_term or "").strip()
    icd_code = (inp.icd10_code or "").strip().upper() or None
    icd_name: str | None = None

    if icd_code:
        try:
            icd_name, exact = await _icd10_lookup(http, icd_code)
            if not icd_name:
                notes.append(f"ICD-10 code {icd_code} not found in ICD-10-CM.")
            elif not exact:
                notes.append(f"{icd_code} is an ICD-10 category; name derived from its sub-codes.")
        except SourceError as exc:
            notes.append(f"ICD-10 lookup failed: {exc}")
    elif name:
        try:
            hit = await _icd10_suggest(http, name)
            if hit:
                icd_code, icd_name = hit
                notes.append(f"ICD-10 {icd_code} suggested from the condition name (not user-supplied).")
        except SourceError:
            pass

    mesh_uid: str | None = None
    if inp.mesh_id:
        mesh_uid = mesh_id_to_uid(inp.mesh_id)
    else:
        for cand in _candidates(name or icd_name or ""):
            try:
                mesh_uid = await _mesh_search(http, settings, cand)
            except SourceError:
                mesh_uid = None
            if mesh_uid:
                break

    mesh_heading = mesh_scope = mesh_id = None
    if mesh_uid:
        try:
            doc = await _mesh_summary(http, settings, mesh_uid)
        except SourceError:
            doc = None
        if doc:
            terms = doc.get("ds_meshterms") or []
            mesh_heading = terms[0] if terms else None
            mesh_scope = doc.get("ds_scopenote") or None
            mesh_id = doc.get("ds_meshui") or mesh_uid_to_id(mesh_uid)
    if not mesh_heading:
        notes.append("MeSH heading could not be resolved; PubMed searches use free-text terms.")

    display = name or icd_name or mesh_heading or (inp.icd10_code or inp.mesh_id or "Unknown condition")
    trial_term = name or icd_name or mesh_heading or display
    return ResolvedCondition(
        display_name=display,
        trial_query_term=trial_term,
        icd10_code=icd_code,
        icd10_name=icd_name,
        mesh_id=mesh_id,
        mesh_heading=mesh_heading,
        mesh_scope_note=mesh_scope,
        notes=notes,
    )


def pubmed_condition_clause(cond: ResolvedCondition) -> str:
    """PubMed clause for the condition. When the user's term is narrower than the MeSH heading
    (e.g. 'Plaque Psoriasis' under 'Psoriasis'), require the term or heading in title/abstract
    so exploded sub-headings (e.g. psoriatic arthritis) don't dominate."""
    heading = cond.mesh_heading
    term = cond.display_name
    if heading:
        clause = f'"{heading}"[MeSH Terms]'
        if term and term.lower() != heading.lower():
            clause = f'({clause} AND ("{term}"[tiab] OR "{heading}"[ti]))'
        return clause
    return f'"{term}"[tiab]'
