"""End-to-end orchestration with recorded API responses and a fake LLM (no network, no API key)."""

from __future__ import annotations

import time

import httpx
import respx
from conftest import FakeLLM, load_json, load_text

from biointel.citations import cited_keys
from biointel.orchestrator import run_briefing
from biointel.schemas import ConditionInput


def mock_apis() -> None:
    active, stopped, results = load_json("ctgov_active.json"), load_json("ctgov_stopped.json"), load_json("ctgov_results.json")

    def ctgov(request: httpx.Request):
        status = request.url.params.get("filter.overallStatus", "")
        if "TERMINATED" in status:
            return httpx.Response(200, json=stopped)
        if status == "COMPLETED":
            return httpx.Response(200, json=results)
        return httpx.Response(200, json=active)

    respx.get(url__startswith="https://clinicaltrials.gov/api/v2/studies/").mock(return_value=httpx.Response(404))
    respx.get(url__startswith="https://clinicaltrials.gov/api/v2/studies").mock(side_effect=ctgov)
    respx.get(url__startswith="https://clinicaltables.nlm.nih.gov").mock(
        return_value=httpx.Response(200, json=[1, ["L40.0"], None, [["L40.0", "Psoriasis vulgaris"]]])
    )

    def esearch(request: httpx.Request):
        if request.url.params.get("db") == "mesh":
            return httpx.Response(200, json={"esearchresult": {"idlist": ["68011565"]}})
        return httpx.Response(200, json=load_json("pubmed_esearch.json"))

    respx.get(url__startswith="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi").mock(side_effect=esearch)
    respx.get(url__startswith="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi").mock(
        side_effect=lambda r: httpx.Response(
            200, json=load_json("mesh_esummary.json") if r.url.params.get("db") == "mesh" else {"result": {"uids": []}}
        )
    )
    respx.get(url__startswith="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi").mock(
        return_value=httpx.Response(200, text=load_text("pubmed_efetch.xml"))
    )
    respx.get(url__startswith="https://efts.sec.gov").mock(return_value=httpx.Response(200, json=load_json("sec_search.json")))


@respx.mock
async def test_end_to_end_with_fake_llm(settings):
    mock_apis()
    llm = FakeLLM(delay=0.3)
    events: list[tuple[str, str]] = []
    stamps: list[tuple[str, str, float]] = []

    def progress(stage, msg, state):
        events.append((stage, state))
        stamps.append((stage, state, time.monotonic()))

    b = await run_briefing(
        ConditionInput(condition_term="Plaque Psoriasis", icd10_code="L40.0"), settings=settings, llm=llm, progress=progress,
    )

    # Every module produced
    assert b.module1 and b.module2 and b.module3a and b.module3b and b.module4 and b.executive_summary
    assert b.condition.mesh_id == "D011565"
    assert b.module3a.trials and all(t.comparators for t in b.module3a.trials)

    # Agents 1 and 2 ran concurrently (PRD: parallel execution): their run intervals overlap
    def span(stage):
        ts = [t for s, _, t in stamps if s == stage]
        return min(ts), max(ts)

    (a1_start, a1_end), (a2_start, a2_end) = span("agent1"), span("agent2")
    assert a1_start < a2_end and a2_start < a1_end
    assert abs(a1_start - a2_start) < 0.05  # launched together
    assert {c[0] for c in llm.calls} >= {"Module1", "Module2", "AssetClassification", "FailureAnalysis"}

    # Zero-hallucination gate: fabricated identifiers are gone, everything cited is a real source
    source_keys = {s.key for s in b.sources}
    assert cited_keys(b) <= source_keys
    content = b.model_dump_json(exclude={"validation"})  # the validation log legitimately records what it removed
    assert "99999999" not in content  # FAKE_PMID / FAKE_NCT scrubbed from text, id lists and rows
    assert "NCT00000001" not in content  # catalyst outside the candidate list dropped
    removed = [i.problem for i in b.validation.issues]
    assert any("PMID 99999999" in p for p in removed) and any("NCT99999999" in p for p in removed)
    assert not any(s.text == "Unsupported claim." for s in b.module1.overview)
    assert all("P999" not in e.source_ids for e in b.module1.epidemiology)
    assert b.validation.statements_dropped >= 3
    # Analyst modules are validated BEFORE synthesis: the aggregator never sees fabricated keys
    agg_prompt = llm.prompts["ExecSummaryLLM"]
    assert "P999" not in agg_prompt and "Unsupported claim." not in agg_prompt
    assert b.executive_summary.key_takeaways and b.executive_summary.white_space
    assert all(d.nct_ids and "NCT99999999" not in d.nct_ids for d in b.module3a.differentiation)

    # Terminology reconciliation applied
    assert b.terminology_map.get("IL-23p19") == "IL-23 (p19)"
    assert b.module2.benchmarks[0].primary_target == "IL-23 (p19)"

    # Fact check covers every LLM-written claim, and still finds claims whose text the aggregator edited
    from biointel.factcheck import claim_key, iter_claims

    fc = b.fact_check
    assert fc is not None and fc.claims
    assert {c.key for c in iter_claims(b)} == set(fc.by_key())
    assert claim_key(b.module2.benchmarks[0]) in fc.by_key()  # text changed by terminology fix after checking
    assert sum(fc.counts().values()) == len(fc.claims)
    assert {"executive_summary", "module1", "module2"} <= {c.module for c in fc.claims}

    # Progress reported for every stage, ending complete
    stages = {s for s, _ in events}
    assert {"resolve", "agent1", "agent2", "finalize", "aggregator", "validation", "factcheck"} <= stages
    assert ("factcheck", "complete") in events
    # once Agent 2 reports complete, no later event flips its panel back to running
    a2_done = events.index(("agent2", "complete"))
    assert ("agent2", "running") not in events[a2_done:]
    assert ("validation", "complete") in events
    assert b.timings["total"] < 60 and b.queries


@respx.mock
async def test_agent_failure_degrades_gracefully(settings):
    mock_apis()

    class BrokenModule1(FakeLLM):
        async def parse(self, **kw):
            if kw["output_format"].__name__ == "Module1":
                raise RuntimeError("simulated refusal")
            return await super().parse(**kw)

    b = await run_briefing(ConditionInput(condition_term="Plaque Psoriasis"), settings=settings, llm=BrokenModule1(delay=0.01))
    assert b.module1 is None and b.module2 is not None
    assert any(e.module == "Module 1" and "simulated refusal" in e.message for e in b.errors)
    assert b.executive_summary is not None
