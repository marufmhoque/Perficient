"""Deterministic parsing of ClinicalTrials.gov v2 study JSON.

Everything here is code, not LLM: registry facts, comparator-arm classification,
primary-endpoint p-value parsing and failure detection.
"""

from __future__ import annotations

import re
import unicodedata

from .schemas import Arm, ComparatorArm, Facility, Intervention, Official, PrimaryResult, TrialRecord

_TYPE_PREFIX_RE = re.compile(
    r"^(Drug|Biological|Device|Procedure|Other|Behavioral|Dietary Supplement|Radiation|Genetic|"
    r"Combination Product|Diagnostic Test)\s*:\s*",
    re.I,
)
_PLACEBO_RE = re.compile(r"\b(placebo|placebos|vehicle|dummy|pbo|matching\s+capsule|matching\s+tablet)\b", re.I)
_SHAM_RE = re.compile(r"\bsham\b", re.I)
_BACKGROUND_RE = re.compile(
    r"\b(background|add-on|add on|in addition to|in combination with|stable dose|standard of care|"
    r"standard care|\bSoC\b|concomitant|on top of)\b",
    re.I,
)
_FORM_WORDS_RE = re.compile(
    r"\b(tablets?|capsules?|cream|ointment|gel|foam|lotion|solution|injection|injectable|infusion|"
    r"subcutaneous|intravenous|oral|topical|s\.?c\.?|i\.?v\.?|prefilled|syringe|pen|auto-?injector|"
    r"suspension|patch|spray|low|medium|high|dose|doses|dosage|regimen|q\d+w|qd|bid|tid|once|twice|daily|weekly)\b",
    re.I,
)
_DOSE_RE = re.compile(r"\b\d+(\.\d+)?\s*(mg|mcg|µg|ug|g|ml|%|mg/kg|iu|units?)\b", re.I)
# "Drug X or matched placebo": randomization to drug vs placebo happens *within* the arm
_EMBEDDED_PLACEBO_RE = re.compile(r"\s*(?:,|/|\bor\b)\s*(?:a\s+)?(?:matched|matching|identical)?\s*(?:placebo|vehicle)\b.*$", re.I)


def strip_type(name: str) -> str:
    # NFKC folds full-width punctuation (e.g. '（low dose）', 'Part 1：') into ASCII equivalents
    return _TYPE_PREFIX_RE.sub("", unicodedata.normalize("NFKC", name or "")).strip()


def split_embedded_placebo(name: str) -> tuple[str, bool]:
    """'CS32582 capsule (low dose) or matched placebo' -> ('CS32582 capsule (low dose)', True)."""
    clean = strip_type(name)
    head = _EMBEDDED_PLACEBO_RE.sub("", clean).strip()
    if head and head != clean and not is_placebo_name(head):
        return head, True
    return clean, False


def is_placebo_name(name: str) -> bool:
    return bool(_PLACEBO_RE.search(name or "")) or bool(_SHAM_RE.search(name or ""))


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def core_drug_name(name: str) -> str:
    """'ICP-488 Tablets 6 mg' -> 'ICP-488'; used for asset grouping and SEC search."""
    s = strip_type(name)
    s = re.sub(r"\(.*?\)", " ", s)
    s = _DOSE_RE.sub(" ", s)
    s = _FORM_WORDS_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip(" ,;:-+/")
    return s or strip_type(name)


def _same_agent(a: str, b: str) -> bool:
    na, nb = norm(core_drug_name(a)), norm(core_drug_name(b))
    return bool(na and nb) and (na == nb or (min(len(na), len(nb)) >= 5 and (na in nb or nb in na)))


def _arm_agents(arm: Arm) -> tuple[list[str], bool]:
    """(non-placebo agents in the arm, arm embeds a placebo control)."""
    agents: list[str] = []
    embedded = False
    for n in arm.intervention_names:
        head, emb = split_embedded_placebo(n)
        embedded = embedded or emb
        if head and not is_placebo_name(head):
            agents.append(head)
    return agents, embedded


def classify_comparators(arms: list[Arm], allocation: str | None) -> tuple[list[ComparatorArm], list[str]]:
    """Return (comparator arms, experimental agents) from CT.gov arm groups."""
    types = [(a.type or "").upper() for a in arms]
    has_placebo_arm = any(t in ("PLACEBO_COMPARATOR", "SHAM_COMPARATOR") for t in types)
    # Sponsor mislabel: no EXPERIMENTAL arm, but "active comparator" arms vs placebo
    # -> the "active" arms are really the investigational drug (e.g. two doses vs placebo).
    investigational_types = {"EXPERIMENTAL"}
    if "EXPERIMENTAL" not in types and has_placebo_arm:
        investigational_types.add("ACTIVE_COMPARATOR")

    experimental: list[str] = []
    embedded_placebo_arms: list[str] = []
    for arm in arms:
        if (arm.type or "").upper() in investigational_types:
            agents, embedded = _arm_agents(arm)
            if embedded:
                embedded_placebo_arms.append(arm.label)
            for a in agents:
                if not any(_same_agent(a, e) for e in experimental):
                    experimental.append(a)

    def is_experimental(name: str) -> bool:
        return any(_same_agent(name, e) for e in experimental)

    comps: list[ComparatorArm] = []
    if embedded_placebo_arms:
        comps.append(ComparatorArm(category="Placebo", arm_label=f"within-arm placebo: {embedded_placebo_arms[0]}"))
    for arm in arms:
        t = (arm.type or "").upper()
        real, _ = _arm_agents(arm)
        names = [strip_type(n) for n in arm.intervention_names]
        placebo_like = any(is_placebo_name(n) for n in names) or bool(_PLACEBO_RE.search(arm.label or ""))
        text = f"{arm.label or ''} {arm.description or ''}"

        if t in investigational_types and t != "EXPERIMENTAL":
            # mislabelled investigational arm (see above): another dose of the study drug
            comps.append(ComparatorArm(category="Dose-comparison", agents=real, arm_label=arm.label))
        elif t == "SHAM_COMPARATOR" or (t != "EXPERIMENTAL" and _SHAM_RE.search(arm.label or "") and not real):
            comps.append(ComparatorArm(category="Sham", arm_label=arm.label))
        elif t == "PLACEBO_COMPARATOR" or (t in ("OTHER", "") and placebo_like and not real) or (
            t == "EXPERIMENTAL" and placebo_like and not real
        ):
            # Interventions in a placebo arm that are not the experimental drug = background therapy.
            # (Excluding experimental agents avoids misreading "placebo -> drug X" crossover arms.)
            background = [n for n in real if not is_experimental(n)]
            if background or _BACKGROUND_RE.search(text):
                comps.append(ComparatorArm(category="Placebo + background therapy", agents=background, arm_label=arm.label))
            else:
                comps.append(ComparatorArm(category="Placebo", arm_label=arm.label))
        elif t == "NO_INTERVENTION":
            comps.append(ComparatorArm(category="No treatment", arm_label=arm.label))
        elif t == "ACTIVE_COMPARATOR":
            # Drop interventions shared with experimental arms (common backbone, e.g. acupuncture in both arms)
            distinct = [n for n in real if not is_experimental(n)]
            if distinct:
                comps.append(ComparatorArm(category="Active comparator", agents=distinct, arm_label=arm.label))
            elif real:  # same drug as the experimental arm: a different dose / regimen, not an external control
                comps.append(ComparatorArm(category="Dose-comparison", agents=real, arm_label=arm.label))

    # A dose-comparison alongside a real control is just design detail; keep it only if it is all we have.
    if any(c.category != "Dose-comparison" for c in comps):
        comps = [c for c in comps if c.category != "Dose-comparison"]
    # de-duplicate identical placebo entries
    deduped: list[ComparatorArm] = []
    for c in comps:
        if c.category in ("Placebo", "Sham", "No treatment") and any(d.category == c.category for d in deduped):
            continue
        deduped.append(c)
    comps = deduped

    if not comps:
        if not arms:
            comps = [ComparatorArm(category="Unclear")]
        elif len(arms) == 1:
            comps = [ComparatorArm(category="Single-arm", arm_label=arms[0].label)]
        else:
            exp_arms = [a for a in arms if (a.type or "").upper() == "EXPERIMENTAL"]
            sets = [{norm(core_drug_name(n)) for n in _arm_agents(a)[0]} for a in exp_arms]
            common = set.intersection(*sets) if len(sets) >= 2 else set()
            common.discard("")
            if common:
                comps = [ComparatorArm(category="Dose-comparison", arm_label=" | ".join(a.label for a in exp_arms[:4]))]
            elif allocation in (None, "NA", "NON_RANDOMIZED") and len(exp_arms) == len(arms):
                comps = [ComparatorArm(category="Single-arm", arm_label=" | ".join(a.label for a in arms[:4]))]
            else:
                comps = [ComparatorArm(category="Unclear", arm_label=" | ".join(a.label for a in arms[:4]))]
    return comps, experimental


_P_RE = re.compile(r"^(<=|>=|≤|≥|<|>|=)?\s*(\d*\.?\d+(?:[eE][-+]?\d+)?)")


def parse_p_value(raw: str | None) -> tuple[float | None, bool | None]:
    """'<0.0001' -> (0.0001, True); '0.42' -> (0.42, False); unparseable -> (None, None)."""
    if not raw:
        return None, None
    s = raw.strip().replace(" ", "").replace("p", "").replace("P", "").lstrip("=")
    m = _P_RE.match(s)
    if not m:
        return None, None
    qual = m.group(1) or "="
    try:
        value = float(m.group(2))
    except ValueError:
        return None, None
    if value > 1:
        return None, None
    if qual in (">", ">=", "≥"):
        return value, False if value >= 0.05 else None
    if qual in ("<", "<=", "≤"):
        return value, value <= 0.05
    return value, value < 0.05


def _primary_results(raw: dict) -> list[PrimaryResult]:
    out: list[PrimaryResult] = []
    measures = raw.get("resultsSection", {}).get("outcomeMeasuresModule", {}).get("outcomeMeasures", []) or []
    for om in measures:
        if om.get("type") != "PRIMARY":
            continue
        groups = {g.get("id"): g.get("title", "") for g in om.get("groups", []) or []}
        for an in om.get("analyses", []) or []:
            raw_p = an.get("pValue")
            if not raw_p:
                continue
            p, sig = parse_p_value(raw_p)
            comparison = " vs ".join(groups.get(g, g) for g in an.get("groupIds", []) or [])
            out.append(
                PrimaryResult(
                    outcome=(om.get("title") or "")[:200], comparison=comparison[:200], p_value_raw=str(raw_p)[:40],
                    p_value=p, significant=sig, analysis_type=an.get("nonInferiorityType"),
                )
            )
    return out


def failed_primary_endpoint(trial: TrialRecord) -> bool:
    """True when every primary-endpoint analysis with a parseable p-value is non-significant."""
    judged = [r for r in trial.primary_results if r.significant is not None]
    return bool(judged) and not any(r.significant for r in judged)


def parse_study(raw: dict) -> TrialRecord:
    ps = raw.get("protocolSection", {})
    idm = ps.get("identificationModule", {})
    stm = ps.get("statusModule", {})
    spm = ps.get("sponsorCollaboratorsModule", {})
    cdm = ps.get("conditionsModule", {})
    dm = ps.get("designModule", {})
    aim = ps.get("armsInterventionsModule", {})
    om = ps.get("outcomesModule", {})
    clm = ps.get("contactsLocationsModule", {})

    arms = [
        Arm(
            label=a.get("label", "") or "",
            type=a.get("type"),
            description=(a.get("description") or "")[:400] or None,
            intervention_names=a.get("interventionNames", []) or [],
        )
        for a in aim.get("armGroups", []) or []
    ]
    interventions = [
        Intervention(name=i.get("name", ""), type=i.get("type"), description=(i.get("description") or "")[:300] or None)
        for i in aim.get("interventions", []) or []
        if i.get("name")
    ]
    allocation = (dm.get("designInfo") or {}).get("allocation")
    comparators, experimental = classify_comparators(arms, allocation)
    if not experimental:  # no arm data: fall back to drug/biological interventions
        experimental = [
            i.name for i in interventions
            if (i.type or "").upper() in ("DRUG", "BIOLOGICAL", "GENETIC", "COMBINATION_PRODUCT") and not is_placebo_name(i.name)
        ][:4]

    facilities = [
        Facility(name=loc["facility"], city=loc.get("city"), country=loc.get("country"))
        for loc in clm.get("locations", []) or []
        if loc.get("facility")
    ]
    lead = spm.get("leadSponsor", {}) or {}

    return TrialRecord(
        nct_id=idm.get("nctId", ""),
        title=idm.get("briefTitle", "") or idm.get("officialTitle", ""),
        acronym=idm.get("acronym"),
        status=stm.get("overallStatus", "UNKNOWN"),
        phases=dm.get("phases", []) or [],
        sponsor=lead.get("name", "Unknown"),
        sponsor_class=lead.get("class"),
        collaborators=[c.get("name", "") for c in spm.get("collaborators", []) or [] if c.get("name")],
        conditions=cdm.get("conditions", []) or [],
        arms=arms,
        interventions=interventions,
        allocation=allocation,
        enrollment=(dm.get("enrollmentInfo") or {}).get("count"),
        start_date=(stm.get("startDateStruct") or {}).get("date"),
        primary_completion_date=(stm.get("primaryCompletionDateStruct") or {}).get("date"),
        completion_date=(stm.get("completionDateStruct") or {}).get("date"),
        last_update=(stm.get("lastUpdatePostDateStruct") or {}).get("date"),
        why_stopped=(stm.get("whyStopped") or "").replace("\\[", "[").replace("\\]", "]") or None,
        has_results=bool(raw.get("hasResults")),
        primary_outcomes=[(p.get("measure") or "")[:200] for p in om.get("primaryOutcomes", []) or [] if p.get("measure")],
        primary_results=_primary_results(raw),
        officials=[
            Official(name=o["name"], affiliation=o.get("affiliation"), role=o.get("role"))
            for o in clm.get("overallOfficials", []) or []
            if o.get("name")
        ],
        facilities=facilities,
        n_sites=len(facilities),
        experimental_agents=experimental,
        comparators=comparators,
    )
