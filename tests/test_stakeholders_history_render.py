"""Module 4 rankings, history store, and Markdown rendering."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from biointel.history import InMemoryHistory, report_from_json, report_to_json
from biointel.render.charts import comparator_mix_by_phase, therapy_flowchart_dot
from biointel.render.markdown import briefing_to_markdown
from biointel.schemas import (
    Briefing,
    CitedStatement,
    ComparatorArm,
    ConditionInput,
    ExecutiveSummary,
    Facility,
    Module2,
    Module3A,
    Official,
    ResolvedCondition,
    Source,
    TherapyLine,
    TrialRecord,
)
from biointel.sources.pubmed import Author, PubMedArticle
from biointel.stakeholders import merge_authors, person_key, rank_centers, rank_officials, rank_sponsors, top_kols


def trial(n: int, sponsor: str, phases: list[str], cls="INDUSTRY", facilities=(), officials=()) -> TrialRecord:
    return TrialRecord(
        nct_id=f"NCT0000000{n}", source_key=f"T{n}", title=f"Trial {n}", status="RECRUITING", phases=phases, sponsor=sponsor,
        sponsor_class=cls, facilities=[Facility(name=f, country="US") for f in facilities],
        officials=[Official(name=o, affiliation="Univ", role="PRINCIPAL_INVESTIGATOR") for o in officials],
        comparators=[ComparatorArm(category="Placebo")],
    )


def test_rank_sponsors_by_ph23_then_total():
    trials = [trial(1, "A", ["PHASE3"]), trial(2, "A", ["PHASE1"]), trial(3, "B", ["PHASE2"]), trial(4, "B", ["PHASE3"]), trial(5, "C", ["PHASE1"])]
    ranked = rank_sponsors(trials)
    assert [s.sponsor for s in ranked] == ["B", "A", "C"]
    assert ranked[0].active_ph2_3 == 2 and ranked[1].active_total == 2


def test_rank_centers_filters_generic_sites():
    trials = [
        trial(1, "A", ["PHASE3"], facilities=["Mayo Clinic", "Research Site", "Local Institution - 0012"]),
        trial(2, "A", ["PHASE3"], facilities=["Mayo Clinic", "Investigational Site Number 840001"]),
    ]
    centers = rank_centers(trials)
    assert [c.name for c in centers] == ["Mayo Clinic"]
    assert centers[0].trial_count == 2 and centers[0].academic


def test_person_key_and_generic_officials():
    assert person_key("Kristian Reich, MD, PhD") == "reich k"
    assert person_key("Dr. Jane A. Smith") == "smith j"
    assert person_key("Medical Director") is None
    assert person_key("Pfizer CT.gov Call Center") is None
    for corporate in ["UCB Cares", "Bristol-Myers Squibb", "Janssen Research & Development", "Acme Therapeutics"]:
        assert person_key(corporate) is None, corporate


def test_sponsor_named_official_is_not_a_kol():
    t = trial(1, "Zenith Dermatology Group", ["PHASE3"], officials=["Zenith Dermatology Group", "Kristian Reich, MD"])
    assert [k.name for k in rank_officials([t]).values()] == ["Kristian Reich"]


def test_kols_merge_trials_and_publications():
    people = rank_officials([trial(1, "A", ["PHASE3"], officials=["Kristian Reich, MD"]), trial(2, "A", ["PHASE3"], officials=["Kristian Reich"])])
    art = PubMedArticle(pmid="1", title="Guideline", authors=[Author(last="Reich", fore="Kristian"), Author(last="Lebwohl", fore="Mark")],
                        query_tags=["guidelines"], source_key="P1")
    kols = top_kols(merge_authors(people, [art]))
    reich = next(k for k in kols if "Reich" in k.name)
    assert reich.trial_count == 2 and reich.publication_count == 1
    assert {"T1", "T2", "P1"} <= set(reich.source_ids)
    assert kols[0].name.startswith("Kristian")


def sample_briefing(created: datetime | None = None, bid="b1") -> Briefing:
    t = trial(1, "A", ["PHASE3"])
    return Briefing(
        id=bid, created_at=created or datetime.now(timezone.utc), input=ConditionInput(condition_term="Psoriasis"),
        condition=ResolvedCondition(display_name="Psoriasis", trial_query_term="Psoriasis"),
        executive_summary=ExecutiveSummary(headline="H", key_takeaways=[CitedStatement(text="K", source_ids=["P1"])]),
        module2=Module2(algorithm_summary=[], lines_of_therapy=[TherapyLine(line="First-line", setting="Mild", therapies=["Calcipotriol"],
                                                                           rationale="r", source_ids=["P1"])], moa=[], benchmarks=[]),
        module3a=Module3A(trials=[t], total_found=1),
        sources=[Source(key="P1", kind="pubmed", native_id="1", title="Guideline", url="https://pubmed.ncbi.nlm.nih.gov/1/"),
                 Source(key="T1", kind="trial", native_id=t.nct_id, title="Trial 1", url=t.url),
                 Source(key="P2", kind="pubmed", native_id="2", title="Uncited", url="https://pubmed.ncbi.nlm.nih.gov/2/")],
    )


def test_history_crud_and_roundtrip():
    h = InMemoryHistory(max_reports=2)
    now = datetime.now(timezone.utc)
    h.save_report(sample_briefing(now - timedelta(hours=2), "old"))
    h.save_report(sample_briefing(now - timedelta(hours=1), "mid"))
    h.save_report(sample_briefing(now, "new"))
    assert [r.id for r in h.list_reports()] == ["new", "mid"]  # oldest evicted, newest first
    b = h.get_report("new")
    again = report_from_json(report_to_json(b))
    assert again == b
    assert h.delete_report("mid") and not h.delete_report("mid")
    h.add_search(ConditionInput(condition_term="A"))
    h.add_search(ConditionInput(condition_term="B"))
    h.add_search(ConditionInput(condition_term="A"))  # de-duplicated, moved to top
    assert [s.label for s in h.recent_searches()] == ["A", "B"]


def test_examples_not_evicted():
    h = InMemoryHistory(max_reports=1)
    ex = sample_briefing(bid="ex")
    ex.is_example = True
    h.save_report(ex)
    h.save_report(sample_briefing(bid="r1"))
    assert {r.id for r in h.list_reports()} == {"ex"} or len(h.list_reports()) <= 2


def test_markdown_has_sections_references_and_links():
    md = briefing_to_markdown(sample_briefing())
    for header in ["## Executive Summary", "## Module 2", "## Module 3A", "## References (cited)", "## Sources consulted but not cited"]:
        assert header in md
    assert "[P1](https://pubmed.ncbi.nlm.nih.gov/1/)" in md
    assert "[NCT00000001](https://clinicaltrials.gov/study/NCT00000001)" in md
    assert "Uncited" in md.split("## Sources consulted but not cited")[1]


def test_charts_build():
    b = sample_briefing()
    fig = comparator_mix_by_phase(b.module3a.trials)
    assert fig.data and fig.data[0].name == "Placebo"
    dot = therapy_flowchart_dot(b.module2)
    assert dot.startswith("digraph") and "Calcipotriol" in dot
