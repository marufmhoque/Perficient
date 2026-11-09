"""ClinicalTrials.gov API v2 adapter (https://clinicaltrials.gov/api/v2)."""

from __future__ import annotations

from datetime import date

from ..citations import SourceRegistry
from ..config import Settings
from ..schemas import TrialRecord
from ..trials import parse_study
from .base import Http

CTGOV = "https://clinicaltrials.gov/api/v2/studies"

# Field projection (piece names). Deliberately excludes site contacts (names / emails / phones).
BASE_FIELDS = [
    "NCTId", "BriefTitle", "Acronym", "OverallStatus", "WhyStopped", "Phase", "StudyType",
    "LeadSponsorName", "LeadSponsorClass", "CollaboratorName", "Condition",
    "ArmGroupLabel", "ArmGroupType", "ArmGroupDescription", "ArmGroupInterventionName",
    "InterventionName", "InterventionType", "InterventionDescription",
    "DesignAllocation", "EnrollmentCount", "StartDate", "PrimaryCompletionDate", "CompletionDate",
    "LastUpdatePostDate", "PrimaryOutcomeMeasure", "HasResults",
    "OverallOfficialName", "OverallOfficialAffiliation", "OverallOfficialRole",
    "LocationFacility", "LocationCity", "LocationCountry",
]
RESULTS_FIELDS = BASE_FIELDS + ["resultsSection.outcomeMeasuresModule"]

ACTIVE_STATUSES = ["RECRUITING", "ACTIVE_NOT_RECRUITING", "NOT_YET_RECRUITING", "ENROLLING_BY_INVITATION"]
STOPPED_STATUSES = ["TERMINATED", "SUSPENDED", "WITHDRAWN"]


class ClinicalTrialsAdapter:
    name = "ClinicalTrials.gov"

    def __init__(self, http: Http, registry: SourceRegistry, settings: Settings) -> None:
        self.http = http
        self.registry = registry
        self.settings = settings

    @staticmethod
    def build_term(
        phases: list[str] | None = None,
        start_since: date | None = None,
        updated_since: date | None = None,
        has_results: bool | None = None,
        std_ages: list[str] | None = None,
    ) -> str:
        terms = ["AREA[StudyType]INTERVENTIONAL"]
        if phases:
            terms.append("AREA[Phase](" + " OR ".join(phases) + ")")
        if std_ages:  # CHILD / ADULT / OLDER_ADULT
            terms.append("AREA[StdAge](" + " OR ".join(std_ages) + ")")
        if start_since:
            terms.append(f"AREA[StartDate]RANGE[{start_since.isoformat()},MAX]")
        if updated_since:
            terms.append(f"AREA[LastUpdatePostDate]RANGE[{updated_since.isoformat()},MAX]")
        if has_results:
            terms.append("AREA[HasResults]true")
        return " AND ".join(terms)

    async def search(
        self,
        condition: str,
        statuses: list[str],
        purpose: str,
        max_records: int,
        phases: list[str] | None = None,
        start_since: date | None = None,
        updated_since: date | None = None,
        has_results: bool | None = None,
        with_results: bool = False,
        std_ages: list[str] | None = None,
    ) -> tuple[list[TrialRecord], int]:
        term = self.build_term(phases, start_since, updated_since, has_results, std_ages)
        params = {
            "query.cond": condition,
            "filter.overallStatus": ",".join(statuses),
            "query.term": term,
            "pageSize": min(100, max_records),
            "countTotal": "true",
            "fields": ",".join(RESULTS_FIELDS if with_results else BASE_FIELDS),
        }
        raw_studies: list[dict] = []
        total = 0
        page_token: str | None = None
        first_url = Http.build_url(CTGOV, params)
        while len(raw_studies) < max_records:
            page_params = dict(params)
            if page_token:
                page_params["pageToken"] = page_token
            data = await self.http.get_json(CTGOV, page_params)
            if not page_token:
                total = int(data.get("totalCount", 0))
            raw_studies.extend(data.get("studies", []) or [])
            page_token = data.get("nextPageToken")
            if not page_token:
                break

        self.registry.log_query(
            "ClinicalTrials.gov", purpose,
            f'condition="{condition}"; status={",".join(statuses)}; {term}', first_url, total,
        )
        trials = []
        for raw in raw_studies[:max_records]:
            t = parse_study(raw)
            if not t.nct_id:
                continue
            src = self.registry.add(
                "trial", t.nct_id, t.title, t.url,
                detail=f"{t.sponsor} · {t.phase_label} · {t.status.replace('_', ' ').title()}",
            )
            t.source_key = src.key
            trials.append(t)
        return trials, total

    async def exists(self, nct_id: str) -> dict | None:
        """Resolve a single NCT ID live (used by the citation validator)."""
        try:
            data = await self.http.get_json(
                f"{CTGOV}/{nct_id}", {"fields": "NCTId,BriefTitle,OverallStatus,Phase,LeadSponsorName"}
            )
        except Exception:
            return None
        ident = data.get("protocolSection", {}).get("identificationModule", {})
        return data if ident.get("nctId", "").upper() == nct_id.upper() else None
