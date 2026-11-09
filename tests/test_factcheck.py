"""Fact checker: deterministic guards around the LLM judge, claim extraction, source integrity,
and fact-checking a saved report (sources re-fetched with respx)."""

from __future__ import annotations

import httpx
import respx
from conftest import FakeLLM, load_json, load_text

from biointel.citations import SourceRegistry
from biointel.factcheck import (
    Claim,
    ClaimJudgement,
    EvidenceStore,
    FactChecker,
    claim_key,
    fact_check_briefing,
    iter_claims,
    quote_in_text,
    source_issues,
    unmatched_numbers,
)
from biointel.schemas import (
    Benchmark,
    Briefing,
    CitedStatement,
    ConditionInput,
    ExecutiveSummary,
    KOL,
    Module2,
    Module3A,
    Module4,
    ResolvedCondition,
    Source,
    TrialRecord,
)

ABSTRACT = ("Risankizumab achieved PASI 90 in 71.6% of patients at weeks 10-16, ahead of brodalumab (70.8%). "
            "Psoriasis affects about 7,550,000 US adults.")


def checker(settings, text=ABSTRACT) -> FactChecker:
    reg = SourceRegistry()
    reg.add("pubmed", "111", "NMA of biologics", "https://pubmed.ncbi.nlm.nih.gov/111/")
    reg.add("pubmed", "222", "Other paper", "https://pubmed.ncbi.nlm.nih.gov/222/")
    ev = EvidenceStore()
    ev.text["P1"] = text
    ev.text["P2"] = "Unrelated abstract about eczema."
    return FactChecker(FakeLLM(), settings, reg, ev)


def claim(text, ids=("P1",)) -> Claim:
    return Claim(key="k", location="module2.x[0]", module="module2", text=text, source_ids=list(ids))


def test_quote_in_text_tolerates_drift_but_not_fabrication():
    assert quote_in_text("achieved PASI 90 in 71.6% of patients", ABSTRACT)
    assert quote_in_text("Risankizumab  achieved PASI 90 in 71.6 % of patients at weeks 10–16", ABSTRACT)  # spacing/dash drift
    assert not quote_in_text("guselkumab achieved complete clearance in most patients", ABSTRACT)
    assert not quote_in_text("", ABSTRACT)


def test_unmatched_numbers():
    assert unmatched_numbers("PASI 90 in 71.6% of patients", ABSTRACT) == []
    assert unmatched_numbers("PASI 90 in 82.5% of patients", ABSTRACT) == ["82.5"]
    assert unmatched_numbers("about 7,550,000 adults (week 4, IL-17)", ABSTRACT) == []  # commas, small ints, names ignored
    assert unmatched_numbers("See NCT01234567", ABSTRACT) == []  # ids are not numbers


def test_supported_with_verified_quote_stays_supported(settings):
    fc = checker(settings)
    out = fc._finalize(claim("Risankizumab reached PASI 90 in 71.6% of patients."),
                       ClaimJudgement(claim_id="C1", verdict="Supported", rationale="ok", evidence_quote="PASI 90 in 71.6% of patients", evidence_source="P1"))
    assert out.verdict == "Supported" and out.quote_verified


def test_fabricated_quote_is_not_accepted(settings):
    fc = checker(settings)
    out = fc._finalize(claim("Risankizumab reached PASI 90 in 71.6% of patients."),
                       ClaimJudgement(claim_id="C1", verdict="Supported", rationale="ok",
                                      evidence_quote="Risankizumab was superior to all comparators at every timepoint", evidence_source="P1"))
    assert out.verdict == "Cannot verify" and not out.quote_verified


def test_quote_from_uncited_source_is_not_accepted(settings):
    fc = checker(settings)
    out = fc._finalize(claim("Risankizumab reached PASI 90 in 71.6% of patients.", ids=("P1",)),
                       ClaimJudgement(claim_id="C1", verdict="Supported", rationale="ok", evidence_quote="Unrelated abstract about eczema", evidence_source="P2"))
    assert out.verdict == "Cannot verify"


def test_number_missing_from_source_downgrades_supported(settings):
    fc = checker(settings)
    out = fc._finalize(claim("Risankizumab reached PASI 90 in 82.5% of patients."),
                       ClaimJudgement(claim_id="C1", verdict="Supported", rationale="ok", evidence_quote="PASI 90 in 71.6% of patients", evidence_source="P1"))
    assert out.verdict == "Partially supported" and out.unmatched_numbers == ["82.5"]


def test_not_supported_verdict_passes_through(settings):
    fc = checker(settings)
    out = fc._finalize(claim("Guselkumab is first-line for all patients."),
                       ClaimJudgement(claim_id="C1", verdict="Not supported", rationale="Source is about risankizumab.", evidence_quote="", evidence_source=""))
    assert out.verdict == "Not supported" and "risankizumab" in out.rationale


def test_iter_claims_covers_llm_content_and_skips_registry_facts():
    b = Briefing(
        id="x", input=ConditionInput(condition_term="Psoriasis"), condition=ResolvedCondition(display_name="Psoriasis", trial_query_term="Psoriasis"),
        executive_summary=ExecutiveSummary(headline="h", key_takeaways=[CitedStatement(text="A claim.", source_ids=["P1"])]),
        module2=Module2(algorithm_summary=[], lines_of_therapy=[], moa=[], benchmarks=[
            Benchmark(therapy="Risankizumab", drug_class="IL-23i", modality="mAb", primary_target="IL-23", efficacy_benchmark="PASI90 71.6%",
                      safety_warnings="Infections", source_ids=["P1"])]),
        module3a=Module3A(trials=[TrialRecord(nct_id="NCT00000001", title="t", status="RECRUITING")]),
        module4=Module4(kols=[KOL(name="Dr X", source_ids=["T1"])]),
    )
    claims = iter_claims(b)
    assert {c.module for c in claims} == {"executive_summary", "module2"}  # trial rows and KOL rankings are not LLM claims
    assert claim_key(b.module2.benchmarks[0]) in {c.key for c in claims}


def test_source_issues_flag_retractions(settings):
    reg = SourceRegistry()
    reg.add("pubmed", "1", "Retracted paper", "u")
    reg.add("sec", "acc:file", "10-K", "u2")
    ev = EvidenceStore()
    ev.text["P1"] = "Title: x\nAbstract: (no abstract available)"
    ev.pub_types["P1"] = ["Journal Article", "Retracted Publication"]
    issues = source_issues(reg, ev, {"P1", "S1"})
    kinds = {(i.key, i.severity) for i in issues}
    assert ("P1", "critical") in kinds and ("P1", "info") in kinds and ("S1", "info") in kinds


@respx.mock
async def test_fact_check_saved_report_refetches_sources(settings):
    """An imported report has no abstracts in memory: the checker must re-fetch them, then judge."""
    respx.get(url__startswith="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi").mock(
        return_value=httpx.Response(200, text=load_text("pubmed_efetch.xml"))
    )
    studies = load_json("ctgov_active.json")["studies"][:1]
    nct = studies[0]["protocolSection"]["identificationModule"]["nctId"]
    ct = respx.get(url__startswith="https://clinicaltrials.gov/api/v2/studies").mock(return_value=httpx.Response(200, json={"studies": studies}))
    from biointel.sources.pubmed import parse_efetch_xml

    pmid = parse_efetch_xml(load_text("pubmed_efetch.xml"))[0].pmid
    b = Briefing(
        id="saved", input=ConditionInput(condition_term="Psoriasis"),
        condition=ResolvedCondition(display_name="Psoriasis", trial_query_term="Psoriasis"),
        executive_summary=ExecutiveSummary(headline="h", key_takeaways=[
            CitedStatement(text="Guideline-backed statement.", source_ids=["P1"]),
            CitedStatement(text="Trial-backed statement.", source_ids=["T1"]),
        ]),
        sources=[Source(key="P1", kind="pubmed", native_id=pmid, title="Guideline", url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"),
                 Source(key="T1", kind="trial", native_id=nct, title="Trial", url=f"https://clinicaltrials.gov/study/{nct}")],
    )
    report = await fact_check_briefing(b, settings, FakeLLM(delay=0.01))
    assert "filter.ids=" in str(ct.calls[0].request.url)
    assert len(report.claims) == 2 and report.sources_with_text == 2
    assert all(c.verdict == "Supported" and c.quote_verified for c in report.claims)
