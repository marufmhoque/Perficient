"""Comparator classification, p-value parsing and study parsing (deterministic logic)."""

from __future__ import annotations

import pytest
from conftest import load_json

from biointel.schemas import Arm, PrimaryResult, TrialRecord
from biointel.trials import (
    classify_comparators,
    core_drug_name,
    failed_primary_endpoint,
    parse_p_value,
    parse_study,
    split_embedded_placebo,
)


def arm(label, type_, *names, description=None):
    return Arm(label=label, type=type_, intervention_names=list(names), description=description)


def cats(comps):
    return [c.category for c in comps]


def test_placebo_controlled():
    comps, exp = classify_comparators(
        [arm("ICP-488", "EXPERIMENTAL", "Drug: ICP-488 Tablets"), arm("Placebo", "PLACEBO_COMPARATOR", "Drug: ICP-488 Placebo")], "RANDOMIZED"
    )
    assert cats(comps) == ["Placebo"]
    assert exp == ["ICP-488 Tablets"]


def test_vehicle_counts_as_placebo():
    comps, _ = classify_comparators([arm("Cream", "EXPERIMENTAL", "Drug: HSK44459 Cream"), arm("Vehicle", "PLACEBO_COMPARATOR", "Drug: Vehicle")], "RANDOMIZED")
    assert cats(comps) == ["Placebo"]


def test_three_arm_placebo_and_active_comparator():
    comps, _ = classify_comparators(
        [
            arm("Guselkumab", "EXPERIMENTAL", "Biological: Guselkumab", "Drug: Placebo for adalimumab"),
            arm("Adalimumab", "ACTIVE_COMPARATOR", "Biological: Adalimumab", "Drug: Placebo for guselkumab"),
            arm("Placebo", "PLACEBO_COMPARATOR", "Drug: Placebo"),
        ],
        "RANDOMIZED",
    )
    assert set(cats(comps)) == {"Active comparator", "Placebo"}
    active = next(c for c in comps if c.category == "Active comparator")
    assert active.agents == ["Adalimumab"]  # double-dummy placebo stripped


def test_shared_backbone_removed_from_active_comparator():
    comps, _ = classify_comparators(
        [
            arm("Acupuncture + Calcipotriol", "ACTIVE_COMPARATOR", "Procedure: Acupuncture", "Drug: Calcipotriol 0.005% ointment"),
            arm("Acupuncture + Herbal", "EXPERIMENTAL", "Procedure: Acupuncture", "Drug: Chinese herb cream"),
        ],
        "RANDOMIZED",
    )
    assert cats(comps) == ["Active comparator"]
    assert comps[0].agents == ["Calcipotriol 0.005% ointment"]


def test_placebo_plus_background_therapy():
    comps, _ = classify_comparators(
        [arm("Drug X + MTX", "EXPERIMENTAL", "Drug: Drug X", "Drug: Methotrexate"), arm("Placebo + MTX", "PLACEBO_COMPARATOR", "Drug: Placebo", "Drug: Methotrexate")],
        "RANDOMIZED",
    )
    # methotrexate is shared with the experimental arm? No - experimental agents include it, so background is detected via text
    assert cats(comps)[0] in ("Placebo", "Placebo + background therapy")
    comps, _ = classify_comparators(
        [arm("Drug X", "EXPERIMENTAL", "Drug: Drug X"), arm("Placebo + MTX", "PLACEBO_COMPARATOR", "Drug: Placebo", "Drug: Methotrexate")],
        "RANDOMIZED",
    )
    assert cats(comps) == ["Placebo + background therapy"]
    assert comps[0].agents == ["Methotrexate"]


def test_placebo_crossover_arm_is_not_background():
    comps, _ = classify_comparators(
        [arm("Risankizumab", "EXPERIMENTAL", "Biological: Risankizumab"),
         arm("Placebo then risankizumab", "PLACEBO_COMPARATOR", "Drug: Placebo", "Biological: Risankizumab")],
        "RANDOMIZED",
    )
    assert cats(comps) == ["Placebo"]


def test_single_arm_and_dose_comparison():
    comps, _ = classify_comparators([arm("Bimekizumab", "EXPERIMENTAL", "Biological: Bimekizumab")], "NA")
    assert cats(comps) == ["Single-arm"]
    comps, _ = classify_comparators(
        [arm("Low dose", "EXPERIMENTAL", "Drug: CS32582 capsule (low dose)"), arm("High dose", "EXPERIMENTAL", "Drug: CS32582 capsule (high dose)")],
        "RANDOMIZED",
    )
    assert cats(comps) == ["Dose-comparison"]


def test_within_arm_placebo():
    comps, exp = classify_comparators(
        [arm("Part 1：Low dose", "EXPERIMENTAL", "Drug: CS32582 capsule（low dose） or matched placebo"),
         arm("Part 1：High dose", "EXPERIMENTAL", "Drug: CS32582 capsule（high dose） or matched placebo")],
        "RANDOMIZED",
    )
    assert "Placebo" in cats(comps)
    assert exp and core_drug_name(exp[0]) == "CS32582"


def test_mislabelled_active_arms_vs_placebo():
    comps, exp = classify_comparators(
        [arm("D-2570 low", "ACTIVE_COMPARATOR", "Drug: D-2570 Tablet"), arm("D-2570 high", "ACTIVE_COMPARATOR", "Drug: D-2570 Tablet"),
         arm("Placebo", "PLACEBO_COMPARATOR", "Drug: Placebo Tablets")],
        "RANDOMIZED",
    )
    assert cats(comps) == ["Placebo"]
    assert exp == ["D-2570 Tablet"]


def test_sham_and_no_intervention():
    comps, _ = classify_comparators([arm("Device", "EXPERIMENTAL", "Device: Laser"), arm("Sham", "SHAM_COMPARATOR", "Device: Sham laser")], "RANDOMIZED")
    assert cats(comps) == ["Sham"]
    comps, _ = classify_comparators([arm("Drug", "EXPERIMENTAL", "Drug: X"), arm("Usual care", "NO_INTERVENTION")], "RANDOMIZED")
    assert cats(comps) == ["No treatment"]


def test_no_arms_is_unclear():
    comps, _ = classify_comparators([], None)
    assert cats(comps) == ["Unclear"]


def test_split_embedded_placebo():
    assert split_embedded_placebo("Drug: ABC-123 or matching placebo") == ("ABC-123", True)
    assert split_embedded_placebo("Drug: ABC-123") == ("ABC-123", False)
    assert split_embedded_placebo("Drug: Placebo") == ("Placebo", False)


@pytest.mark.parametrize(
    "raw,value,sig",
    [("<0.0001", 0.0001, True), ("0.42", 0.42, False), ("p=0.03", 0.03, True), ("0.05", 0.05, False), (">0.05", 0.05, False),
     ("≤0.05", 0.05, True), ("NS", None, None), ("", None, None), ("1.5", None, None)],
)
def test_parse_p_value(raw, value, sig):
    v, s = parse_p_value(raw)
    assert v == value and s is sig


def test_failed_primary_endpoint():
    t = TrialRecord(nct_id="NCT1", title="x", status="COMPLETED")
    assert not failed_primary_endpoint(t)  # no analyses -> not judged
    t.primary_results = [PrimaryResult(outcome="PASI75", comparison="A vs P", p_value_raw="0.06", p_value=0.06, significant=False)]
    assert failed_primary_endpoint(t)
    t.primary_results.append(PrimaryResult(outcome="PASI75", comparison="B vs P", p_value_raw="0.01", p_value=0.01, significant=True))
    assert not failed_primary_endpoint(t)  # one dose succeeded


def test_safety_primary_endpoint_is_not_an_efficacy_failure():
    # live-run defect: IMO-8400's p=0.06 was on a safety / tolerability primary outcome
    t = TrialRecord(nct_id="NCT01899729", title="x", status="COMPLETED", primary_results=[
        PrimaryResult(outcome="Number of Participants With Treatment-Emergent Adverse Events", comparison="A vs P",
                      p_value_raw="0.06", p_value=0.06, significant=False),
        PrimaryResult(outcome="Maximum Plasma Concentration (Cmax)", comparison="A vs P", p_value_raw="0.4", p_value=0.4, significant=False),
    ])
    assert not failed_primary_endpoint(t)
    t.primary_results.append(PrimaryResult(outcome="PASI 75 response at week 12", comparison="A vs P", p_value_raw="0.31",
                                           p_value=0.31, significant=False))
    assert failed_primary_endpoint(t)


def test_core_drug_name():
    assert core_drug_name("Drug: ICP-488 Tablets 6 mg") == "ICP-488"
    assert core_drug_name("Biological: Risankizumab 150 mg SC injection") == "Risankizumab"


def test_parse_real_studies_have_no_contact_data():
    data = load_json("ctgov_active.json")
    trials = [parse_study(s) for s in data["studies"]]
    assert trials and all(t.nct_id.startswith("NCT") for t in trials)
    assert all(t.comparators for t in trials)
    dumped = "".join(t.model_dump_json() for t in trials)
    assert "@" not in dumped


def test_parse_results_section():
    data = load_json("ctgov_results.json")
    trials = [parse_study(s) for s in data["studies"]]
    assert any(t.primary_results for t in trials)
    for t in trials:
        for r in t.primary_results:
            assert r.p_value_raw
