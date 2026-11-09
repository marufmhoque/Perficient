"""Refresh the offline test fixtures from the live public APIs.

    python scripts/build_fixtures.py

ClinicalTrials.gov responses are fetched with the same field projection the app uses, which
excludes site contact details. The script asserts no e-mail addresses end up in the fixtures.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biointel.sources.clinicaltrials import BASE_FIELDS, CTGOV, RESULTS_FIELDS  # noqa: E402

OUT = ROOT / "tests" / "fixtures"
UA = {"User-Agent": "BioIntelPrototype/0.1 fixture-builder"}
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def ctgov(params: dict, n: int) -> dict:
    data = httpx.get(CTGOV, params={**params, "pageSize": n, "countTotal": "true"}, headers=UA, timeout=30).json()
    data.pop("nextPageToken", None)
    return data


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = {"query.cond": "Plaque Psoriasis", "query.term": "AREA[StudyType]INTERVENTIONAL"}
    files = {
        "ctgov_active.json": ctgov({**base, "filter.overallStatus": "RECRUITING,ACTIVE_NOT_RECRUITING", "fields": ",".join(BASE_FIELDS)}, 12),
        "ctgov_stopped.json": ctgov({**base, "filter.overallStatus": "TERMINATED,SUSPENDED,WITHDRAWN", "fields": ",".join(BASE_FIELDS)}, 6),
        "ctgov_results.json": ctgov(
            {"query.cond": "Plaque Psoriasis", "filter.overallStatus": "COMPLETED", "query.term": "AREA[HasResults]true AND AREA[Phase](PHASE2 OR PHASE3)",
             "fields": ",".join(RESULTS_FIELDS)}, 4),
    }
    eutils = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    files["pubmed_esearch.json"] = httpx.get(eutils + "esearch.fcgi", params={"db": "pubmed", "retmode": "json", "retmax": 3,
                                                                               "term": '"Psoriasis"[MeSH Terms] AND "Practice Guideline"[pt]'}, timeout=30).json()
    ids = files["pubmed_esearch.json"]["esearchresult"]["idlist"]
    xml_text = httpx.get(eutils + "efetch.fcgi", params={"db": "pubmed", "retmode": "xml", "id": ",".join(ids)}, timeout=30).text
    # author affiliations can embed personal e-mail addresses: redact before committing
    xml_text = EMAIL_RE.sub("author@example.org", xml_text)
    (OUT / "pubmed_efetch.xml").write_text(xml_text, encoding="utf-8")
    files["mesh_esummary.json"] = httpx.get(eutils + "esummary.fcgi", params={"db": "mesh", "retmode": "json", "id": "68011565"}, timeout=30).json()
    files["icd10_E11.json"] = httpx.get("https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search",
                                        params={"sf": "code,name", "terms": "E11", "maxList": 50}, timeout=30).json()
    sec = httpx.get("https://efts.sec.gov/LATEST/search-index", params={"q": '"deucravacitinib"', "forms": "10-K,10-Q,8-K"},
                    headers=UA, timeout=30).json()
    sec["hits"]["hits"] = sec["hits"]["hits"][:3]
    files["sec_search.json"] = sec

    for name, data in files.items():
        text = json.dumps(data, indent=1, ensure_ascii=False)
        if name.startswith("ctgov") and "@" in text:
            raise SystemExit(f"{name} contains an '@' - refusing to write possible contact data")
        (OUT / name).write_text(text, encoding="utf-8")
        print(f"wrote {name} ({len(text):,} bytes)")


if __name__ == "__main__":
    main()
