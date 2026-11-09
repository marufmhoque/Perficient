"""PubMed via NCBI E-utilities (esearch / efetch / esummary)."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from pydantic import BaseModel, Field

from ..citations import SourceRegistry
from ..config import Settings
from .base import Http

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


class Author(BaseModel):
    last: str
    fore: str = ""
    affiliation: str | None = None

    @property
    def display(self) -> str:
        initials = "".join(part[0] for part in self.fore.replace("-", " ").split() if part)
        return f"{self.last} {initials}".strip()


class PubMedArticle(BaseModel):
    pmid: str
    title: str
    abstract: str = ""
    journal: str = ""
    year: str = ""
    authors: list[Author] = Field(default_factory=list)
    pub_types: list[str] = Field(default_factory=list)
    doi: str | None = None
    query_tags: list[str] = Field(default_factory=list)  # which searches surfaced it
    source_key: str = ""

    @property
    def url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"

    @property
    def citation(self) -> str:
        if not self.authors:
            who = ""
        elif len(self.authors) > 2:
            who = f"{self.authors[0].display}, et al. "
        else:
            who = ", ".join(a.display for a in self.authors) + ". "
        return f"{who}{self.journal}. {self.year}".strip()


_EMAIL_RE = re.compile(r"\s*(?:Electronic address:\s*)?[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\.?", re.I)


def _text(el: ET.Element | None) -> str:
    return "".join(el.itertext()).strip() if el is not None else ""


def strip_emails(text: str) -> str:
    """Author affiliations often embed e-mail addresses; the app never shows contact details."""
    return _EMAIL_RE.sub("", text).strip(" ;,.") if text else text


def parse_efetch_xml(xml_text: str) -> list[PubMedArticle]:
    root = ET.fromstring(xml_text)
    out: list[PubMedArticle] = []
    for art in root.iter("PubmedArticle"):
        cit = art.find("MedlineCitation")
        if cit is None:
            continue
        pmid = _text(cit.find("PMID"))
        article = cit.find("Article")
        if not pmid or article is None:
            continue
        title = _text(article.find("ArticleTitle"))
        parts = []
        for ab in article.findall("Abstract/AbstractText"):
            label = ab.get("Label")
            txt = _text(ab)
            if txt:
                parts.append(f"{label}: {txt}" if label else txt)
        journal = _text(article.find("Journal/ISOAbbreviation")) or _text(article.find("Journal/Title"))
        year = (
            _text(article.find("Journal/JournalIssue/PubDate/Year"))
            or _text(article.find("Journal/JournalIssue/PubDate/MedlineDate"))[:4]
            or _text(article.find("ArticleDate/Year"))
        )
        authors = []
        for a in article.findall("AuthorList/Author"):
            last = _text(a.find("LastName")) or _text(a.find("CollectiveName"))
            if not last:
                continue
            aff = strip_emails(_text(a.find("AffiliationInfo/Affiliation")))
            authors.append(Author(last=last, fore=_text(a.find("ForeName")), affiliation=aff or None))
        pub_types = [_text(pt) for pt in article.findall("PublicationTypeList/PublicationType")]
        doi = None
        for eid in art.findall("PubmedData/ArticleIdList/ArticleId"):
            if eid.get("IdType") == "doi":
                doi = _text(eid)
        out.append(
            PubMedArticle(
                pmid=pmid, title=title, abstract="\n".join(parts), journal=journal, year=year,
                authors=authors, pub_types=pub_types, doi=doi,
            )
        )
    return out


class PubMedAdapter:
    name = "PubMed"

    def __init__(self, http: Http, registry: SourceRegistry, settings: Settings) -> None:
        self.http = http
        self.registry = registry
        self.settings = settings

    def _base_params(self) -> dict[str, str]:
        params = {"tool": "biointel-prototype"}
        if self.settings.ncbi_api_key:
            params["api_key"] = self.settings.ncbi_api_key
        return params

    async def search(
        self, term: str, retmax: int, purpose: str, mindate: int | None = None, sort: str = "relevance"
    ) -> list[str]:
        params = {**self._base_params(), "db": "pubmed", "retmode": "json", "retmax": retmax, "term": term, "sort": sort}
        if mindate:
            params.update({"datetype": "pdat", "mindate": str(mindate), "maxdate": "3000"})
        data = await self.http.get_json(EUTILS + "esearch.fcgi", params)
        ids = data.get("esearchresult", {}).get("idlist", [])
        public = {k: v for k, v in params.items() if k != "api_key"}
        self.registry.log_query(
            "PubMed", purpose, term, Http.build_url(EUTILS + "esearch.fcgi", public), int(data.get("esearchresult", {}).get("count", 0))
        )
        return ids

    async def fetch(self, pmids: list[str]) -> list[PubMedArticle]:
        """efetch abstracts and register every article in the SourceRegistry."""
        if not pmids:
            return []
        articles: list[PubMedArticle] = []
        for i in range(0, len(pmids), 100):
            chunk = pmids[i : i + 100]
            params = {**self._base_params(), "db": "pubmed", "retmode": "xml", "id": ",".join(chunk)}
            xml_text = await self.http.get_text(EUTILS + "efetch.fcgi", params)
            articles.extend(parse_efetch_xml(xml_text))
        for a in articles:
            src = self.registry.add("pubmed", a.pmid, a.title, a.url, detail=f"{a.citation}. PMID: {a.pmid}")
            a.source_key = src.key
        return articles

    async def existing(self, pmids: list[str]) -> dict[str, dict]:
        """esummary: return {pmid: summary} for PMIDs that resolve to real records."""
        if not pmids:
            return {}
        params = {**self._base_params(), "db": "pubmed", "retmode": "json", "id": ",".join(pmids)}
        data = await self.http.get_json(EUTILS + "esummary.fcgi", params)
        result = data.get("result", {})
        return {
            uid: result[uid]
            for uid in result.get("uids", [])
            if isinstance(result.get(uid), dict) and not result[uid].get("error")
        }
