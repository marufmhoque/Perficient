"""Citation gate: fabricated identifiers never survive; real ones are normalized or verified live."""

from __future__ import annotations

from biointel.citations import SourceRegistry, cited_keys
from biointel.schemas import (
    Briefing,
    Catalyst,
    CitedStatement,
    ConditionInput,
    DifferentiationEntry,
    ExecutiveSummary,
    FailureRow,
    Module3A,
    Module3B,
    ResolvedCondition,
)
from biointel.validation import CitationValidator


def registry() -> SourceRegistry:
    reg = SourceRegistry()
    reg.add("pubmed", "30772098", "AAD-NPF guideline", "https://pubmed.ncbi.nlm.nih.gov/30772098/")
    reg.add("trial", "NCT06842199", "ICP-488 Ph3", "https://clinicaltrials.gov/study/NCT06842199")
    return reg


def briefing(**kw) -> Briefing:
    return Briefing(id="t", input=ConditionInput(condition_term="x"), condition=ResolvedCondition(display_name="x", trial_query_term="x"), **kw)


class FakePubMed:
    async def existing(self, pmids):
        return {p: {"title": "Real paper", "source": "J", "pubdate": "2020"} for p in pmids if p == "11111111"}


class FakeTrials:
    async def exists(self, nct):
        return {"protocolSection": {"identificationModule": {"nctId": nct, "briefTitle": "Real trial"}}} if nct == "NCT01234567" else None


async def test_registry_resolve_formats():
    reg = registry()
    assert reg.resolve("P1") == "P1"
    assert reg.resolve("[P1]") == "P1"
    assert reg.resolve("PMID: 30772098") == "P1"
    assert reg.resolve("30772098") == "P1"
    assert reg.resolve("nct06842199") == "T1"
    assert reg.resolve("P2") is None
    assert reg.resolve("NCT99999999") is None


async def test_fabricated_ids_removed_and_unsupported_dropped():
    es = ExecutiveSummary(
        headline="Headline mentioning NCT99999999 and PMID 99999999.",
        key_takeaways=[
            CitedStatement(text="Supported via raw PMID.", source_ids=["30772098"]),
            CitedStatement(text="Fabricated only.", source_ids=["P42", "NCT99999999"]),
            CitedStatement(text="Mixed.", source_ids=["P1", "P42"]),
        ],
    )
    b = briefing(executive_summary=es)
    report = await CitationValidator(registry(), FakePubMed(), FakeTrials()).validate(b)
    assert [s.text for s in es.key_takeaways] == ["Supported via raw PMID.", "Mixed."]
    assert es.key_takeaways[0].source_ids == ["P1"]  # normalized to key
    assert es.key_takeaways[1].source_ids == ["P1"]
    assert "NCT99999999" not in es.headline and "99999999" not in es.headline
    assert report.statements_dropped == 1
    assert report.ids_valid == 2


async def test_live_resolution_registers_real_ids():
    reg = registry()
    es = ExecutiveSummary(headline="h", key_takeaways=[CitedStatement(text="Real but not retrieved.", source_ids=["PMID 11111111", "NCT01234567"])])
    report = await CitationValidator(reg, FakePubMed(), FakeTrials()).validate(briefing(executive_summary=es))
    assert len(es.key_takeaways) == 1
    assert set(es.key_takeaways[0].source_ids) == {"P2", "T2"}
    assert report.ids_resolved_live == 2


async def test_rows_with_unknown_nct_dropped():
    reg = registry()
    m3b = Module3B(failures=[
        FailureRow(nct_id="NCT06842199", asset="a", sponsor="s", phase="Ph3", status="Terminated", stop_reason="x", evidence="whyStopped",
                   failure_category="Efficacy", takeaway="t", comparator_summary="Placebo", source_ids=["T1"]),
        FailureRow(nct_id="NCT99999999", asset="b", sponsor="s", phase="Ph3", status="Terminated", stop_reason="x", evidence="whyStopped",
                   failure_category="Efficacy", takeaway="t", comparator_summary="Placebo", source_ids=["T9"]),
    ])
    m3a = Module3A(differentiation=[DifferentiationEntry(asset="a", nct_ids=["NCT06842199", "NCT99999999"], modality="m", target_moa="t",
                                                          comparator_evidence="c", differentiation_vs_soc="d", advantage_axes=["Efficacy"], source_ids=["T1"])])
    es = ExecutiveSummary(headline="h", catalysts=[Catalyst(nct_id="NCT99999999", asset="a", sponsor="s", phase="Ph3", primary_completion="2027",
                                                            why_it_matters="w", source_ids=["T1"])])
    b = briefing(module3a=m3a, module3b=m3b, executive_summary=es)
    await CitationValidator(reg).validate(b)
    assert [f.nct_id for f in m3b.failures] == ["NCT06842199"]
    assert m3a.differentiation[0].nct_ids == ["NCT06842199"]
    assert es.catalysts == []


async def test_flag_mode_keeps_unsourced():
    es = ExecutiveSummary(headline="h", key_takeaways=[CitedStatement(text="No sources.", source_ids=[])])
    report = await CitationValidator(registry(), drop_unsourced=False).validate(briefing(executive_summary=es))
    assert len(es.key_takeaways) == 1
    assert report.issues[0].action == "flagged"


async def test_every_surviving_id_exists_in_registry():
    reg = registry()
    es = ExecutiveSummary(headline="h", key_takeaways=[CitedStatement(text="a", source_ids=["P1", "bogus", "T1"])])
    b = briefing(executive_summary=es)
    await CitationValidator(reg).validate(b)
    assert cited_keys(b) <= {s.key for s in reg.all()}
