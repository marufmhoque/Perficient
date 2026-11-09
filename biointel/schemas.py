"""Domain schemas shared by all agents.

Conventions
-----------
* Every claim-bearing object carries ``source_ids``: keys from the SourceRegistry
  (``P#`` = PubMed article, ``T#`` = ClinicalTrials.gov record, ``S#`` = SEC filing).
* Models that Claude fills via structured outputs have **no defaults** (all fields required),
  which keeps their JSON schema compatible with structured outputs.
* Registry-derived facts (``TrialRecord``, ``Source``) are built by code, never by the LLM.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import ClassVar, Literal

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

PHASES = ["EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4"]

ALL_PATIENTS = "All patients"
# "Group: value" labels keep the dropdown scannable and searchable.
PATIENT_SEGMENTS = [
    ALL_PATIENTS,
    "Severity: Mild",
    "Severity: Moderate",
    "Severity: Severe",
    "Severity: Mild-to-moderate",
    "Severity: Moderate-to-severe",
    "Severity: Refractory / treatment-resistant",
    "Age: Pediatric (<12 y)",
    "Age: Adolescent (12-17 y)",
    "Age: Adult (18-64 y)",
    "Age: Older adult (65+ y)",
    "Treatment history: Treatment-naive",
    "Treatment history: Inadequate response to first-line therapy",
    "Treatment history: Biologic / advanced-therapy naive",
    "Treatment history: Biologic / advanced-therapy experienced",
    "Treatment history: Intolerant to standard of care",
    "Disease course: Newly diagnosed / early disease",
    "Disease course: Established / chronic disease",
    "Disease course: Relapsed / recurrent",
    "Disease course: Advanced / metastatic",
    "Special population: Pregnant or breastfeeding",
    "Special population: Women of childbearing potential",
    "Special population: Renal impairment",
    "Special population: Hepatic impairment",
    "Special population: Immunocompromised",
    "Special population: Significant comorbidities",
]
# ClinicalTrials.gov standard age groups for the age segments (used as a registry filter)
SEGMENT_STD_AGES = {
    "Age: Pediatric (<12 y)": "CHILD",
    "Age: Adolescent (12-17 y)": "CHILD",
    "Age: Adult (18-64 y)": "ADULT",
    "Age: Older adult (65+ y)": "OLDER_ADULT",
}


class ConditionInput(BaseModel):
    """Union of the PRD's Agent 1 and Agent 2 input schemas."""

    condition_term: str = ""
    icd10_code: str | None = None  # optional - resolved from the name when blank
    mesh_id: str | None = None  # optional - resolved from the name when blank
    geographies: list[str] = Field(default_factory=lambda: ["US", "EU5"])
    target_patient_segment: str | None = None  # free-text description (PRD field); None = all patients
    patient_segments: list[str] = Field(default_factory=list)  # structured picks from PATIENT_SEGMENTS
    development_phases: list[str] = Field(default_factory=lambda: ["PHASE1", "PHASE2", "PHASE3"])
    time_horizon_years: int = 5
    include_discontinued: bool = True

    def label(self) -> str:
        bits = [self.condition_term or self.icd10_code or self.mesh_id or "?"]
        if self.icd10_code and self.condition_term:
            bits.append(self.icd10_code)
        return " · ".join(bits)

    @property
    def all_patients(self) -> bool:
        picks = [s for s in self.patient_segments if s != ALL_PATIENTS]
        return not picks and not (self.target_patient_segment or "").strip()

    def segment_description(self) -> str:
        """Human-readable segment for prompts and report headers."""
        if self.all_patients:
            return ALL_PATIENTS
        picks = [s.split(": ", 1)[-1] for s in self.patient_segments if s != ALL_PATIENTS]
        extra = (self.target_patient_segment or "").strip()
        # older saved reports stored the segment only in target_patient_segment
        return "; ".join(dict.fromkeys(picks + ([extra] if extra else [])))

    def std_ages(self) -> list[str]:
        return sorted({SEGMENT_STD_AGES[s] for s in self.patient_segments if s in SEGMENT_STD_AGES})


class ResolvedCondition(BaseModel):
    display_name: str
    trial_query_term: str  # term passed to ClinicalTrials.gov query.cond
    icd10_code: str | None = None
    icd10_name: str | None = None
    mesh_id: str | None = None
    mesh_heading: str | None = None
    mesh_scope_note: str | None = None
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Sources & citations
# ---------------------------------------------------------------------------

SourceKind = Literal["pubmed", "trial", "sec"]


class Source(BaseModel):
    SKIP_TEXT_SCAN: ClassVar[bool] = True

    key: str  # P1 / T1 / S1
    kind: SourceKind
    native_id: str  # PMID / NCT ID / SEC accession+file
    title: str
    url: str
    detail: str = ""  # citation line, e.g. "Smith J, et al. J Am Acad Dermatol. 2024"
    retrieved_at: datetime = Field(default_factory=utcnow)


class QueryLog(BaseModel):
    SKIP_TEXT_SCAN: ClassVar[bool] = True

    source: str  # "PubMed", "ClinicalTrials.gov", "SEC EDGAR", ...
    purpose: str
    query: str
    url: str
    n_results: int
    at: datetime = Field(default_factory=utcnow)


class CitedStatement(BaseModel):
    text: str = Field(description="One concise, factual statement.")
    source_ids: list[str] = Field(description="Source keys (e.g. P3, T12) that support the statement.")


# ---------------------------------------------------------------------------
# Module 1 - Condition brief (LLM-filled)
# ---------------------------------------------------------------------------


class EpiFigure(BaseModel):
    metric: str = Field(description="e.g. Prevalence, Incidence, DALYs")
    region: str = Field(description="e.g. Global, US, EU5, UK")
    value: str = Field(description="Value with units exactly as reported, e.g. '2-3% of adults'")
    population_note: str = Field(description="Population / year / method caveats")
    source_ids: list[str]


class SeverityStratum(BaseModel):
    severity: str = Field(description="Mild / Moderate / Severe or equivalent")
    share: str = Field(description="Share of patients, as reported")
    definition: str = Field(description="How severity is defined (e.g. BSA, PASI)")
    source_ids: list[str]


class UnmetNeed(BaseModel):
    domain: str = Field(description="e.g. Efficacy / Response Rate, Safety & Tolerability, Administration Route")
    current_limitation: str
    clinical_impact: str
    opportunity: str = Field(description="Opportunity for novel assets")
    source_ids: list[str]


class Module1(BaseModel):
    overview: list[CitedStatement]
    epidemiology: list[EpiFigure]
    severity_stratification: list[SeverityStratum]
    addressable_population: list[CitedStatement]
    pathophysiology: list[CitedStatement]
    clinical_manifestations: list[CitedStatement]
    comorbidities: list[CitedStatement]
    prognosis: list[CitedStatement]
    unmet_needs: list[UnmetNeed]


# ---------------------------------------------------------------------------
# Module 2 - Standard of care & MoA (LLM-filled)
# ---------------------------------------------------------------------------


class TherapyLine(BaseModel):
    line: Literal["First-line", "Second-line", "Third-line / Refractory"]
    setting: str = Field(description="Population / severity this applies to")
    therapies: list[str] = Field(description="Generic drug names or classes")
    rationale: str
    source_ids: list[str]


class MoAEntry(BaseModel):
    target: str = Field(description="Molecular target, e.g. IL-23 (p19), TNF-alpha, TYK2")
    therapy_class: str
    example_agents: list[str] = Field(description="Generic names of approved agents")
    mechanism: str = Field(description="How target engagement changes signalling")
    downstream_effects: str = Field(description="Cascade and tissue-level resolution")
    source_ids: list[str]


class Benchmark(BaseModel):
    therapy: str
    drug_class: str
    modality: str
    primary_target: str
    efficacy_benchmark: str = Field(description="Primary endpoint benchmark as reported, with timepoint")
    safety_warnings: str = Field(description="Major safety warnings / boxed warnings")
    source_ids: list[str]


class Module2(BaseModel):
    algorithm_summary: list[CitedStatement]
    lines_of_therapy: list[TherapyLine]
    moa: list[MoAEntry]
    benchmarks: list[Benchmark]


# ---------------------------------------------------------------------------
# Trials (registry facts, code-built)
# ---------------------------------------------------------------------------


class Arm(BaseModel):
    label: str
    type: str | None = None  # EXPERIMENTAL, ACTIVE_COMPARATOR, PLACEBO_COMPARATOR, ...
    description: str | None = None
    intervention_names: list[str] = Field(default_factory=list)


class Intervention(BaseModel):
    name: str
    type: str | None = None
    description: str | None = None


class Official(BaseModel):
    name: str
    affiliation: str | None = None
    role: str | None = None


class Facility(BaseModel):
    name: str
    city: str | None = None
    country: str | None = None


class PrimaryResult(BaseModel):
    outcome: str
    comparison: str  # group titles joined, e.g. "Drug 10 mg vs Placebo"
    p_value_raw: str
    p_value: float | None = None
    significant: bool | None = None
    analysis_type: str | None = None  # SUPERIORITY / NON_INFERIORITY / ...


ComparatorCategory = Literal[
    "Placebo",
    "Placebo + background therapy",
    "Sham",
    "No treatment",
    "Active comparator",
    "Dose-comparison",
    "Single-arm",
    "Unclear",
]


class ComparatorArm(BaseModel):
    category: ComparatorCategory
    agents: list[str] = Field(default_factory=list)  # active comparator / background drugs
    arm_label: str = ""
    is_soc: bool | None = None  # set for active comparators once Module 2 (SoC) is known


class TrialRecord(BaseModel):
    SKIP_TEXT_SCAN: ClassVar[bool] = True

    nct_id: str
    source_key: str = ""  # T#
    title: str
    acronym: str | None = None
    status: str
    phases: list[str] = Field(default_factory=list)
    sponsor: str = "Unknown"
    sponsor_class: str | None = None
    collaborators: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    arms: list[Arm] = Field(default_factory=list)
    interventions: list[Intervention] = Field(default_factory=list)
    allocation: str | None = None
    enrollment: int | None = None
    start_date: str | None = None
    primary_completion_date: str | None = None
    completion_date: str | None = None
    last_update: str | None = None
    why_stopped: str | None = None
    has_results: bool = False
    primary_outcomes: list[str] = Field(default_factory=list)
    primary_results: list[PrimaryResult] = Field(default_factory=list)
    officials: list[Official] = Field(default_factory=list)
    facilities: list[Facility] = Field(default_factory=list)  # truncated after Module 4 counting
    n_sites: int = 0

    # Derived
    experimental_agents: list[str] = Field(default_factory=list)
    comparators: list[ComparatorArm] = Field(default_factory=list)

    # LLM enrichment (classification only)
    modality: str | None = None
    target_moa: str | None = None

    @property
    def url(self) -> str:
        return f"https://clinicaltrials.gov/study/{self.nct_id}"

    @property
    def phase_label(self) -> str:
        if not self.phases:
            return "N/A"
        mapping = {"EARLY_PHASE1": "Early Ph1", "PHASE1": "Ph1", "PHASE2": "Ph2", "PHASE3": "Ph3", "PHASE4": "Ph4", "NA": "N/A"}
        return "/".join(mapping.get(p, p) for p in self.phases)

    @property
    def max_phase(self) -> int:
        order = {"EARLY_PHASE1": 0, "PHASE1": 1, "PHASE2": 2, "PHASE3": 3, "PHASE4": 4}
        return max((order.get(p, -1) for p in self.phases), default=-1)

    @property
    def asset_name(self) -> str:
        return ", ".join(self.experimental_agents) if self.experimental_agents else (self.acronym or self.title[:60])

    @property
    def comparator_class(self) -> str:
        """Single headline class for charts (strongest comparator wins)."""
        cats = {c.category for c in self.comparators}
        actives = [c for c in self.comparators if c.category == "Active comparator"]
        if actives:
            if any(c.is_soc for c in actives):
                return "Active – current SoC"
            if all(c.is_soc is False for c in actives):
                return "Active – other"
            return "Active (unclassified)"
        for cat, label in [
            ("Placebo + background therapy", "Placebo + background therapy"),
            ("Placebo", "Placebo"),
            ("Sham", "Sham"),
            ("No treatment", "No treatment"),
            ("Dose-comparison", "Dose-comparison (no control)"),
            ("Single-arm", "Single-arm (no comparator)"),
        ]:
            if cat in cats:
                return label
        return "Unclear"

    @property
    def comparator_summary(self) -> str:
        parts: list[str] = []
        for c in self.comparators:
            if c.category == "Active comparator":
                tag = " (SoC)" if c.is_soc else (" (non-SoC)" if c.is_soc is False else "")
                parts.append(f"Active: {', '.join(c.agents) or c.arm_label}{tag}")
            elif c.category == "Placebo + background therapy":
                parts.append(f"Placebo + {', '.join(c.agents)}" if c.agents else "Placebo + background therapy")
            else:
                parts.append(c.category)
        # de-duplicate while keeping order
        seen: list[str] = []
        for p in parts:
            if p not in seen:
                seen.append(p)
        return " · ".join(seen) if seen else "Unclear"


# ---------------------------------------------------------------------------
# Module 3A / 3B / 4
# ---------------------------------------------------------------------------

AdvantageAxis = Literal[
    "Efficacy", "Safety", "Route / convenience", "Dosing frequency", "Selectivity", "Novel mechanism", "Population", "Cost / access"
]


class DifferentiationEntry(BaseModel):
    asset: str
    nct_ids: list[str] = Field(description="NCT IDs of this asset's trials (from the provided list only)")
    modality: str
    target_moa: str
    comparator_evidence: str = Field(description="What it is being tested against and what that implies")
    differentiation_vs_soc: str = Field(description="How it aims to beat current SoC")
    advantage_axes: list[AdvantageAxis]
    source_ids: list[str]


class Module3A(BaseModel):
    trials: list[TrialRecord] = Field(default_factory=list)
    total_found: int = 0
    pipeline_summary: list[CitedStatement] = Field(default_factory=list)
    differentiation: list[DifferentiationEntry] = Field(default_factory=list)


FailureCategory = Literal[
    "Efficacy", "Safety", "Business / strategic", "Enrollment / feasibility", "Regulatory", "Other / unspecified"
]


class FailureRow(BaseModel):
    nct_id: str
    asset: str
    sponsor: str
    phase: str
    status: str
    stop_reason: str  # verbatim whyStopped and/or primary endpoint p-value evidence
    evidence: Literal["whyStopped", "primary endpoint p-value", "both"]
    failure_category: FailureCategory
    takeaway: str
    comparator_summary: str
    source_ids: list[str] = Field(default_factory=list)
    sec_source_ids: list[str] = Field(default_factory=list)


class Module3B(BaseModel):
    trials: list[TrialRecord] = Field(default_factory=list)
    failures: list[FailureRow] = Field(default_factory=list)
    lessons: list[CitedStatement] = Field(default_factory=list)
    total_stopped_found: int = 0


class SponsorRank(BaseModel):
    sponsor: str
    sponsor_class: str | None = None
    active_ph2_3: int
    active_total: int
    source_ids: list[str] = Field(default_factory=list)


class CenterRank(BaseModel):
    name: str
    country: str | None = None
    academic: bool = False
    trial_count: int
    source_ids: list[str] = Field(default_factory=list)


class KOL(BaseModel):
    name: str
    affiliation: str | None = None
    roles: list[str] = Field(default_factory=list)
    trial_count: int = 0
    publication_count: int = 0
    source_ids: list[str] = Field(default_factory=list)


class Module4(BaseModel):
    sponsors: list[SponsorRank] = Field(default_factory=list)
    centers: list[CenterRank] = Field(default_factory=list)
    kols: list[KOL] = Field(default_factory=list)
    narrative: list[CitedStatement] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Executive summary, validation, briefing
# ---------------------------------------------------------------------------


class Catalyst(BaseModel):
    nct_id: str
    asset: str
    sponsor: str
    phase: str
    primary_completion: str
    why_it_matters: str
    source_ids: list[str] = Field(default_factory=list)


class ExecutiveSummary(BaseModel):
    headline: str
    key_takeaways: list[CitedStatement] = Field(default_factory=list)
    white_space: list[CitedStatement] = Field(default_factory=list)
    competitive_threats: list[CitedStatement] = Field(default_factory=list)
    catalysts: list[Catalyst] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    location: str
    problem: str
    action: Literal["dropped", "removed id", "flagged", "resolved live"]


class ValidationReport(BaseModel):
    ids_checked: int = 0
    ids_valid: int = 0
    ids_resolved_live: int = 0
    statements_checked: int = 0
    statements_dropped: int = 0
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.issues


class ModuleError(BaseModel):
    module: str
    message: str


# ---------------------------------------------------------------------------
# Fact check (claim-by-claim grounding + source integrity)
# ---------------------------------------------------------------------------

Verdict = Literal["Supported", "Partially supported", "Not supported", "Cannot verify"]


class ClaimCheck(BaseModel):
    SKIP_TEXT_SCAN: ClassVar[bool] = True

    claim_key: str  # hash of the claim text (renderers look claims up by it)
    location: str  # e.g. module1.epidemiology[3]
    module: str
    claim: str
    source_ids: list[str] = Field(default_factory=list)
    verdict: Verdict
    rationale: str = ""
    evidence_quote: str = ""
    evidence_source: str = ""
    quote_verified: bool = False
    unmatched_numbers: list[str] = Field(default_factory=list)


class SourceIssue(BaseModel):
    SKIP_TEXT_SCAN: ClassVar[bool] = True

    key: str
    severity: Literal["critical", "warning", "info"]
    issue: str


class FactCheckReport(BaseModel):
    SKIP_TEXT_SCAN: ClassVar[bool] = True

    claims: list[ClaimCheck] = Field(default_factory=list)
    source_issues: list[SourceIssue] = Field(default_factory=list)
    sources_with_text: int = 0
    model: str = ""
    checked_at: datetime = Field(default_factory=utcnow)

    def counts(self) -> dict[str, int]:
        out = {"Supported": 0, "Partially supported": 0, "Not supported": 0, "Cannot verify": 0}
        for c in self.claims:
            out[c.verdict] += 1
        return out

    @property
    def supported_share(self) -> float | None:
        return (self.counts()["Supported"] / len(self.claims)) if self.claims else None

    def by_key(self) -> dict[str, ClaimCheck]:
        return {c.claim_key: c for c in self.claims}


class Briefing(BaseModel):
    id: str
    created_at: datetime = Field(default_factory=utcnow)
    input: ConditionInput
    condition: ResolvedCondition
    executive_summary: ExecutiveSummary | None = None
    module1: Module1 | None = None
    module2: Module2 | None = None
    module3a: Module3A | None = None
    module3b: Module3B | None = None
    module4: Module4 | None = None
    sources: list[Source] = Field(default_factory=list)
    queries: list[QueryLog] = Field(default_factory=list)
    validation: ValidationReport = Field(default_factory=ValidationReport)
    fact_check: FactCheckReport | None = None
    errors: list[ModuleError] = Field(default_factory=list)
    timings: dict[str, float] = Field(default_factory=dict)
    models: dict[str, str] = Field(default_factory=dict)
    usage: dict[str, int] = Field(default_factory=dict)
    terminology_map: dict[str, str] = Field(default_factory=dict)
    is_example: bool = False

    @property
    def title(self) -> str:
        return self.condition.display_name

    def source_map(self) -> dict[str, Source]:
        return {s.key: s for s in self.sources}
