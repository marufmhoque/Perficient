"""Adapters against recorded API responses (respx - no network)."""

from __future__ import annotations

import httpx
import pytest
import respx
from conftest import load_json, load_text

from biointel.citations import SourceRegistry
from biointel.schemas import ConditionInput
from biointel.sources import ACTIVE_STATUSES, ClinicalTrialsAdapter, Http, PubMedAdapter, SecEdgarAdapter, resolve_condition
from biointel.sources.pubmed import parse_efetch_xml
from biointel.sources.resolve import category_name, mesh_id_to_uid, mesh_uid_to_id


def test_efetch_parsing():
    arts = parse_efetch_xml(load_text("pubmed_efetch.xml"))
    assert len(arts) >= 1
    a = arts[0]
    assert a.pmid.isdigit() and a.title and a.year
    assert a.authors and a.authors[0].last
    assert a.url == f"https://pubmed.ncbi.nlm.nih.gov/{a.pmid}/"
    # affiliations never carry e-mail addresses into the app
    assert not any("@" in (au.affiliation or "") for art in arts for au in art.authors)


def test_strip_emails():
    from biointel.sources.pubmed import strip_emails

    assert strip_emails("Dept of Dermatology, Univ X, Chile. Electronic address: someone@uni.cl.") == "Dept of Dermatology, Univ X, Chile"
    assert strip_emails("Univ Y; jane.doe@example.org") == "Univ Y"


@respx.mock
async def test_pubmed_search_and_fetch_register_sources(settings):
    respx.get(url__startswith="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi").mock(
        return_value=httpx.Response(200, json=load_json("pubmed_esearch.json"))
    )
    respx.get(url__startswith="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi").mock(
        return_value=httpx.Response(200, text=load_text("pubmed_efetch.xml"))
    )
    reg = SourceRegistry()
    async with Http(settings) as http:
        pm = PubMedAdapter(http, reg, settings)
        ids = await pm.search('"Psoriasis"[MeSH Terms]', 3, "test")
        arts = await pm.fetch(ids)
    assert arts and all(a.source_key.startswith("P") for a in arts)
    assert reg.queries and reg.queries[0].source == "PubMed"
    assert "api_key" not in reg.queries[0].url


@respx.mock
async def test_ctgov_pagination_and_registration(settings):
    page = load_json("ctgov_active.json")
    studies = page["studies"]
    first = {"totalCount": len(studies), "studies": studies[:5], "nextPageToken": "abc"}
    second = {"studies": studies[5:]}

    def handler(request: httpx.Request):
        return httpx.Response(200, json=second if "pageToken=abc" in str(request.url) else first)

    route = respx.get(url__startswith="https://clinicaltrials.gov/api/v2/studies").mock(side_effect=handler)
    reg = SourceRegistry()
    async with Http(settings) as http:
        ct = ClinicalTrialsAdapter(http, reg, settings)
        trials, total = await ct.search("Plaque Psoriasis", ACTIVE_STATUSES, "test", max_records=100, phases=["PHASE2", "PHASE3"])
    assert route.call_count == 2
    assert total == len(studies) and len(trials) == len(studies)
    assert [t.source_key for t in trials][:2] == ["T1", "T2"]
    q = str(route.calls[0].request.url)
    assert "query.cond=Plaque+Psoriasis" in q and "AREA%5BPhase%5D" in q and "LocationContact" not in q


def test_build_term():
    from datetime import date

    term = ClinicalTrialsAdapter.build_term(["PHASE2", "PHASE3"], start_since=date(2021, 1, 1), has_results=True)
    assert term == "AREA[StudyType]INTERVENTIONAL AND AREA[Phase](PHASE2 OR PHASE3) AND AREA[StartDate]RANGE[2021-01-01,MAX] AND AREA[HasResults]true"


@respx.mock
async def test_sec_search_builds_filing_urls(settings):
    respx.get(url__startswith="https://efts.sec.gov/LATEST/search-index").mock(return_value=httpx.Response(200, json=load_json("sec_search.json")))
    reg = SourceRegistry()
    async with Http(settings) as http:
        srcs = await SecEdgarAdapter(http, reg, settings).search_asset("deucravacitinib", "test")
    assert srcs and srcs[0].key == "S1"
    assert srcs[0].url.startswith("https://www.sec.gov/Archives/edgar/data/")


@respx.mock
async def test_resolve_icd_category_and_mesh(settings):
    respx.get(url__startswith="https://clinicaltables.nlm.nih.gov").mock(return_value=httpx.Response(200, json=load_json("icd10_E11.json")))

    def mesh_search(request: httpx.Request):
        term = request.url.params.get("term", "")
        return httpx.Response(200, json={"esearchresult": {"idlist": ["68003924"] if "Type 2 diabetes mellitus" in term else []}})

    respx.get(url__startswith="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi").mock(side_effect=mesh_search)
    respx.get(url__startswith="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi").mock(
        return_value=httpx.Response(200, json={"result": {"uids": ["68003924"], "68003924": {"ds_meshterms": ["Diabetes Mellitus, Type 2"], "ds_scopenote": "x"}}})
    )
    async with Http(settings) as http:
        cond = await resolve_condition(ConditionInput(icd10_code="E11"), http, settings)
    assert cond.display_name == "Type 2 diabetes mellitus"
    assert cond.mesh_id == "D003924" and cond.mesh_heading == "Diabetes Mellitus, Type 2"
    assert any("category" in n for n in cond.notes)


def test_mesh_id_conversion():
    assert mesh_uid_to_id("68011565") == "D011565"
    assert mesh_id_to_uid("D011565") == "68011565"


def test_category_name():
    assert category_name(["Type 2 diabetes mellitus with coma", "Type 2 diabetes mellitus with ketoacidosis"]) == "Type 2 diabetes mellitus"
    assert category_name([]) is None


@respx.mock
async def test_http_retries_then_fails(settings):
    route = respx.get("https://clinicaltrials.gov/api/v2/studies").mock(return_value=httpx.Response(503))
    async with Http(settings) as http:
        with pytest.raises(Exception):
            await http.get_json("https://clinicaltrials.gov/api/v2/studies", retries=1)
    assert route.call_count == 2


@respx.mock
async def test_http_client_error_not_retried(settings):
    route = respx.get("https://clinicaltrials.gov/api/v2/studies").mock(return_value=httpx.Response(400, text="bad field"))
    async with Http(settings) as http:
        with pytest.raises(Exception, match="HTTP 400"):
            await http.get_json("https://clinicaltrials.gov/api/v2/studies", retries=3)
    assert route.call_count == 1
