from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biointel.agents.aggregator import CatalystPick, ExecSummaryLLM, TermMapping  # noqa: E402
from biointel.agents.pipeline import (  # noqa: E402
    AssetClass,
    AssetClassification,
    ComparatorSoCLabel,
    DifferentiationOutput,
    FailureAnalysis,
    FailureAssessment,
    StakeholderNarrative,
)
from biointel.config import get_settings  # noqa: E402
from biointel.factcheck import ClaimJudgement, FactCheckBatch  # noqa: E402
from biointel.schemas import (  # noqa: E402
    Benchmark,
    CitedStatement,
    DifferentiationEntry,
    EpiFigure,
    Module1,
    Module2,
    MoAEntry,
    SeverityStratum,
    TherapyLine,
    UnmetNeed,
)
from biointel.sources.base import clear_response_cache  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"


def load_json(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def load_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture
def settings():
    return get_settings(response_cache_ttl_s=0, anthropic_api_key="test-key", ncbi_api_key=None)


@pytest.fixture(autouse=True)
def _clean_cache():
    clear_response_cache()
    yield
    clear_response_cache()


def _first_keys(user: str, prefix: str, n: int = 2) -> list[str]:
    keys = list(dict.fromkeys(re.findall(rf"\[({prefix}\d+)\]", user)))
    return keys[:n] or [f"{prefix}1"]


class FakeLLM:
    """Deterministic stand-in for Claude. Records (schema, start, end) for every call and
    deliberately injects fabricated citations so the validator has something to remove."""

    FAKE_PMID = "PMID 99999999"
    FAKE_NCT = "NCT99999999"

    def __init__(self, delay: float = 0.2) -> None:
        self.delay = delay
        self.calls: list[tuple[str, float, float]] = []
        self.prompts: dict[str, str] = {}
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

    async def parse(self, *, model, system, user, output_format, effort, max_tokens=None):
        self.prompts[output_format.__name__] = user
        start = time.monotonic()
        await asyncio.sleep(self.delay)
        out = self._build(output_format, user)
        self.calls.append((output_format.__name__, start, time.monotonic()))
        self.usage["calls"] += 1
        return out

    def _build(self, fmt, user: str):
        p = _first_keys(user, "P")
        t = _first_keys(user, "T")
        stmt = lambda text, ids: CitedStatement(text=text, source_ids=ids)  # noqa: E731
        if fmt is Module1:
            return Module1(
                overview=[stmt("Chronic immune-mediated skin disease.", p), stmt("Unsupported claim.", [])],
                epidemiology=[EpiFigure(metric="Prevalence", region="US", value="3%", population_note="adults", source_ids=p),
                              EpiFigure(metric="Prevalence", region="EU5", value="2%", population_note="fabricated", source_ids=["P999"])],
                severity_stratification=[SeverityStratum(severity="Moderate-to-severe", share="20%", definition="BSA>10%", source_ids=p)],
                addressable_population=[stmt("About 1.5M moderate-to-severe patients.", p)],
                pathophysiology=[stmt(f"IL-23/Th17 axis drives disease ({self.FAKE_PMID}).", p)],
                clinical_manifestations=[stmt("Erythematous scaly plaques.", p)],
                comorbidities=[stmt("Psoriatic arthritis.", p)],
                prognosis=[stmt("Lifelong relapsing course.", p)],
                unmet_needs=[UnmetNeed(domain="Administration Route", current_limitation="Injections", clinical_impact="Burden",
                                       opportunity="Oral agents", source_ids=p)],
            )
        if fmt is Module2:
            return Module2(
                algorithm_summary=[stmt("Topicals first, then systemic/biologic therapy.", p)],
                lines_of_therapy=[
                    TherapyLine(line="First-line", setting="Mild", therapies=["Calcipotriol", "Topical corticosteroids"], rationale="Guideline", source_ids=p),
                    TherapyLine(line="Second-line", setting="Moderate-to-severe", therapies=["Methotrexate", "Deucravacitinib"], rationale="Guideline", source_ids=p),
                    TherapyLine(line="Third-line / Refractory", setting="Biologic failure", therapies=["Risankizumab", "Adalimumab"], rationale="Guideline", source_ids=p),
                ],
                moa=[MoAEntry(target="IL-23 (p19)", therapy_class="IL-23 inhibitors", example_agents=["risankizumab", "guselkumab"],
                              mechanism="Blocks p19", downstream_effects="Reduces Th17", source_ids=p)],
                benchmarks=[Benchmark(therapy="Risankizumab", drug_class="IL-23i", modality="mAb", primary_target="IL-23p19",
                                      efficacy_benchmark="PASI90 ~75% wk16", safety_warnings="Infections", source_ids=p)],
            )
        if fmt is AssetClassification:
            names = re.findall(r"^- (.+?) \| sponsor:", user, re.M)
            return AssetClassification(assets=[AssetClass(asset=n, modality="Oral small molecule", target="TYK2",
                                                          mechanism="Allosteric TYK2 inhibitor", confidence="medium") for n in names])
        if fmt is FailureAnalysis:
            ncts = re.findall(r"\] (NCT\d{8})", user)
            return FailureAnalysis(
                assessments=[FailureAssessment(nct_id=n, failure_category="Business / strategic", takeaway="Portfolio decision.") for n in ncts],
                lessons=[stmt("Several stops were strategic, not efficacy-driven.", t), stmt(f"See {self.FAKE_NCT}.", t)],
            )
        if fmt is DifferentiationOutput:
            rows = re.findall(r"^- (.+?) \| modality.*?trials: (NCT\d{8}) \[(T\d+)\]", user, re.M)
            agents = re.findall(r"^- (.+)$", user.split("ACTIVE-COMPARATOR AGENTS TO LABEL (SoC or not):")[-1], re.M)
            return DifferentiationOutput(
                entries=[DifferentiationEntry(asset=a, nct_ids=[n, self.FAKE_NCT], modality="Oral small molecule", target_moa="TYK2",
                                              comparator_evidence="vs placebo", differentiation_vs_soc="Oral convenience",
                                              advantage_axes=["Route / convenience"], source_ids=[k]) for a, n, k in rows[:3]],
                pipeline_summary=[stmt("Oral TYK2 inhibitors dominate late-stage pipeline.", t)],
                comparator_labels=[ComparatorSoCLabel(agent=a.strip(), is_soc=True) for a in agents if a.strip() != "(none)"],
            )
        if fmt is StakeholderNarrative:
            return StakeholderNarrative(narrative=[stmt("Industry sponsors dominate Phase 3.", t)])
        if fmt is FactCheckBatch:
            # quote the first words of the first cited source, so the quote check can pass
            sources = dict(re.findall(r"^\[([PTS]\d+)\]\n(.+)$", user, re.M))
            judgements = []
            for cid, cites, _text in re.findall(r"^(C\d+) \(cites ([^)]*)\): (.*)$", user, re.M):
                first = next((k.strip() for k in cites.split(",") if k.strip() in sources), "")
                quote = " ".join(sources.get(first, "").split()[:10])
                judgements.append(ClaimJudgement(claim_id=cid, verdict="Supported" if quote else "Cannot verify",
                                                 rationale="Stated in the source.", evidence_quote=quote, evidence_source=first))
            return FactCheckBatch(judgements=judgements)
        if fmt is ExecSummaryLLM:
            cands = re.findall(r"\] (NCT\d{8})", user.split("CATALYST CANDIDATES")[-1])
            return ExecSummaryLLM(
                headline="Oral IL-23 pathway agents are the main competitive threat.",
                key_takeaways=[stmt("Biologics set a high efficacy bar.", p), stmt("No source here.", ["X1"])],
                white_space=[stmt("Durable oral efficacy remains unmet.", p)],
                competitive_threats=[stmt("Multiple oral TYK2 programs.", t)],
                catalysts=[CatalystPick(nct_id=n, why_it_matters="Readout") for n in cands[:2]] + [CatalystPick(nct_id="NCT00000001", why_it_matters="fake")],
                terminology_map=[TermMapping(variant="IL-23p19", canonical="IL-23 (p19)")],
            )
        raise AssertionError(f"FakeLLM has no builder for {fmt.__name__}")
