"""Module 4 - deterministic stakeholder rankings (no LLM): sponsors, trial centers, KOLs."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from .schemas import KOL, CenterRank, SponsorRank, TrialRecord
from .sources.pubmed import PubMedArticle

_GENERIC_SITE_RE = re.compile(
    r"(\b(investigative|investigational|investigator|research|study|clinical|trial)\s+site\b|\bsite\s*#?\s*\d+|"
    r"local institution|^\s*\d+\s*$|for additional information|call center|research facility\s*\d*$|"
    r"^\s*(site|center|centre)\b)",
    re.I,
)
_ACADEMIC_RE = re.compile(
    r"(universit|hospital|hôpital|hospit|ospedale|klinik|clinic\b|klinikum|charit|medical cent|medical college|"
    r"\bcollege\b|institut|school of medicine|faculty|academ|health system|nhs|\bumc\b|mayo|policlinic)",
    re.I,
)
_GENERIC_OFFICIAL_RE = re.compile(
    r"(call center|medical director|study director|clinical trial|clinical development|clinical research|"
    r"clinical operations|clinical program|sponsor|medical monitor|central contact|global|ct\.gov|information|"
    r"medical lead|medical affairs|\bteam\b|\bdirector\b|use central|trial manager|clinical science|\binc\b|\bltd\b|"
    r"pharmaceutical|corporation|\bgmbh\b|\bcares\b|therapeutics|biosciences?|biopharma|laborator|\bllc\b|\bco\.|"
    r"research (&|and) development|\bR&D\b|bristol|janssen|novartis|pfizer|lilly|abbvie|amgen|sanofi|takeda|"
    r"\bucb\b|merck|boehringer|astrazeneca|glaxo|roche|genentech|celgene|regeneron)",
    re.I,
)
_TITLE_RE = re.compile(r"^(dr\.?|prof\.?|professor|mr\.?|mrs\.?|ms\.?)\s+", re.I)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def rank_sponsors(trials: list[TrialRecord], top: int = 15) -> list[SponsorRank]:
    by: dict[str, dict] = {}
    for t in trials:
        d = by.setdefault(t.sponsor, {"class": t.sponsor_class, "ph23": 0, "total": 0, "keys": []})
        d["total"] += 1
        if any(p in ("PHASE2", "PHASE3") for p in t.phases):
            d["ph23"] += 1
        if t.source_key and len(d["keys"]) < 8:
            d["keys"].append(t.source_key)
    ranked = sorted(by.items(), key=lambda kv: (-kv[1]["ph23"], -kv[1]["total"], kv[0]))
    return [
        SponsorRank(sponsor=name, sponsor_class=d["class"], active_ph2_3=d["ph23"], active_total=d["total"], source_ids=d["keys"])
        for name, d in ranked[:top]
    ]


def rank_centers(trials: list[TrialRecord], top: int = 20) -> list[CenterRank]:
    counts: Counter[str] = Counter()
    display: dict[str, Counter[str]] = defaultdict(Counter)
    country: dict[str, str | None] = {}
    keys: dict[str, list[str]] = defaultdict(list)
    for t in trials:
        seen: set[str] = set()
        for f in t.facilities:
            if _GENERIC_SITE_RE.search(f.name):
                continue
            k = _norm(f.name)
            if not k or k in seen:
                continue
            seen.add(k)
            counts[k] += 1
            display[k][f.name.strip()] += 1
            country.setdefault(k, f.country)
            if t.source_key and len(keys[k]) < 8:
                keys[k].append(t.source_key)
    out = []
    for k, n in counts.most_common():
        name = display[k].most_common(1)[0][0]
        out.append(CenterRank(name=name, country=country.get(k), academic=bool(_ACADEMIC_RE.search(name)), trial_count=n, source_ids=keys[k]))
        if len(out) >= top:
            break
    return out


def person_key(name: str) -> str | None:
    """'Kristian Reich, MD, PhD' -> 'reich k'; returns None for generic placeholders."""
    if not name or _GENERIC_OFFICIAL_RE.search(name):
        return None
    base = _TITLE_RE.sub("", name.split(",")[0]).strip()
    tokens = [tok.strip(".") for tok in base.split() if tok.strip(".")]
    if len(tokens) < 2:
        return None
    return f"{tokens[-1].lower()} {tokens[0][0].lower()}"


def rank_officials(trials: list[TrialRecord]) -> dict[str, KOL]:
    people: dict[str, KOL] = {}
    organisations = {_norm(t.sponsor) for t in trials} | {_norm(c) for t in trials for c in t.collaborators}
    for t in trials:
        for o in t.officials:
            key = person_key(o.name)
            # registry "officials" are sometimes the sponsor itself (e.g. "UCB Cares"): not a person
            if not key or _norm(o.name.split(",")[0]) in organisations:
                continue
            kol = people.get(key)
            if kol is None:
                kol = KOL(name=o.name.split(",")[0].strip(), affiliation=o.affiliation)
                people[key] = kol
            kol.trial_count += 1
            role = (o.role or "").replace("_", " ").title()
            if role and role not in kol.roles:
                kol.roles.append(role)
            if not kol.affiliation and o.affiliation:
                kol.affiliation = o.affiliation
            if t.source_key and t.source_key not in kol.source_ids and len(kol.source_ids) < 6:
                kol.source_ids.append(t.source_key)
    return people


def merge_authors(people: dict[str, KOL], articles: list[PubMedArticle]) -> dict[str, KOL]:
    """Add first/last authors of guideline & treatment literature (seminal publications)."""
    for a in articles:
        if not ({"guidelines", "treatment"} & set(a.query_tags)) or not a.authors:
            continue
        for pos, author in ((0, a.authors[0]), (len(a.authors) - 1, a.authors[-1])):
            if not author.fore:
                continue
            key = f"{author.last.lower()} {author.fore[0].lower()}"
            kol = people.get(key)
            if kol is None:
                kol = KOL(name=f"{author.fore} {author.last}", affiliation=author.affiliation)
                people[key] = kol
            kol.publication_count += 1
            role = "First author" if pos == 0 else "Senior author"
            if role not in kol.roles:
                kol.roles.append(role)
            if not kol.affiliation and author.affiliation:
                kol.affiliation = author.affiliation
            if a.source_key and a.source_key not in kol.source_ids and len(kol.source_ids) < 8:
                kol.source_ids.append(a.source_key)
    return people


def top_kols(people: dict[str, KOL], top: int = 20) -> list[KOL]:
    def score(k: KOL) -> tuple:
        return (-(2 * k.trial_count + k.publication_count), -k.trial_count, k.name)

    ranked = sorted(people.values(), key=score)
    # affiliation strings from PubMed can be very long; keep them readable
    for k in ranked:
        if k.affiliation and len(k.affiliation) > 160:
            k.affiliation = k.affiliation[:157] + "…"
    return ranked[:top]
