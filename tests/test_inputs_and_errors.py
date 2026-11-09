"""Briefing-failure fixes (fail fast on account errors, placeholder keys), optional / validated
ICD-10 and MeSH, and patient segments."""

from __future__ import annotations

import httpx
import pytest
import respx
from conftest import FakeLLM

from biointel.config import _is_placeholder
from biointel.llm import LLMAccountError, _account_error
from biointel.orchestrator import run_briefing
from biointel.schemas import ALL_PATIENTS, ConditionInput
from biointel.sources import ClinicalTrialsAdapter, Http, resolve_condition
from biointel.sources.resolve import valid_icd10, valid_mesh


def test_placeholder_keys_are_ignored():
    for v in ["sk-ant-...", "", "   ", None, "<your-key>"]:
        assert _is_placeholder(v)
    assert not _is_placeholder("sk-ant-api03-realLookingKey")


@pytest.mark.parametrize(
    "secrets,expected",
    [
        ({"anthropic_api_key": "sk-ant-real-1"}, "sk-ant-real-1"),  # name lower-cased
        ({"anthropic.api_key": "sk-ant-real-2"}, "sk-ant-real-2"),  # [anthropic] section
        ({"anthropic_api_key": ' "sk-ant-real-3" '}, "sk-ant-real-3"),  # stray quotes / spaces
        ({"anthropic_api_key": "sk-ant-..."}, None),  # template placeholder
        ({"openai_key": "x"}, None),
    ],
)
def test_secret_lookup_variants(monkeypatch, secrets, expected):
    import biointel.config as cfg

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("anthropic_api_key", raising=False)
    monkeypatch.setattr(cfg, "_flat_streamlit_secrets", lambda: secrets)
    assert cfg._secret("ANTHROPIC_API_KEY") == expected


def test_secrets_diagnostic_names_what_exists_but_never_values(monkeypatch):
    import biointel.config as cfg

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(cfg, "_flat_streamlit_secrets", lambda: {"anthropic_key": "sk-ant-secret-value"})
    msg = cfg.secrets_diagnostic()
    assert "anthropic_key" in msg and "none is named ANTHROPIC_API_KEY" in msg
    assert "sk-ant-secret-value" not in msg

    def bad_toml():
        raise FileNotFoundError("Error parsing secrets file at /x/secrets.toml: Invalid value (at line 1, column 21)")

    monkeypatch.setattr(cfg, "_flat_streamlit_secrets", bad_toml)
    assert "without quotes" in cfg.secrets_diagnostic() and "line 1, column 21" in cfg.secrets_diagnostic()

    def missing():
        raise FileNotFoundError("No secrets found.")

    monkeypatch.setattr(cfg, "_flat_streamlit_secrets", missing)
    assert "No secrets are configured" in cfg.secrets_diagnostic()


def test_billing_and_auth_errors_are_account_errors():
    import anthropic

    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    billing = anthropic.BadRequestError(
        "Your credit balance is too low to access the Anthropic API.", response=httpx.Response(400, request=req), body=None
    )
    auth = anthropic.AuthenticationError("invalid x-api-key", response=httpx.Response(401, request=req), body=None)
    other = anthropic.BadRequestError("max_tokens too large", response=httpx.Response(400, request=req), body=None)
    assert "credits" in str(_account_error(billing))
    assert "rejected" in str(_account_error(auth))
    assert _account_error(other) is None


@respx.mock
async def test_account_error_stops_run_before_any_retrieval(settings):
    respx.get(url__startswith="https://clinicaltables.nlm.nih.gov").mock(return_value=httpx.Response(200, json=[0, [], None, []]))
    respx.get(url__startswith="https://eutils.ncbi.nlm.nih.gov").mock(return_value=httpx.Response(200, json={"esearchresult": {"idlist": []}}))
    trials = respx.get(url__startswith="https://clinicaltrials.gov").mock(return_value=httpx.Response(200, json={"studies": []}))

    class NoCredits(FakeLLM):
        fatal = None

        async def preflight(self):
            self.fatal = LLMAccountError("Your Anthropic account has no API credits left.")
            raise self.fatal

    events = []
    with pytest.raises(LLMAccountError, match="no API credits"):
        await run_briefing(ConditionInput(condition_term="Psoriasis"), settings=settings, llm=NoCredits(),
                           progress=lambda s, m, st: events.append((s, st)))
    assert trials.call_count == 0  # no agent work started
    assert ("resolve", "error") in events


@pytest.mark.parametrize("code,ok", [("L40.0", True), ("E11", True), ("E11.65", True), ("U07.1", True), ("L400", False), ("psoriasis", False), ("40.0", False)])
def test_icd10_format(code, ok):
    assert valid_icd10(code) is ok


@pytest.mark.parametrize("mesh,ok", [("D011565", True), ("C000657245", True), ("011565", False), ("D11", False), ("Psoriasis", False)])
def test_mesh_format(mesh, ok):
    assert valid_mesh(mesh) is ok


def _mock_resolution(icd_rows=None, mesh_heading="Psoriasis"):
    respx.get(url__startswith="https://clinicaltables.nlm.nih.gov").mock(return_value=httpx.Response(200, json=[1, [], None, icd_rows or []]))
    respx.get(url__startswith="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi").mock(
        return_value=httpx.Response(200, json={"esearchresult": {"idlist": ["68011565"]}}))
    respx.get(url__startswith="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi").mock(
        return_value=httpx.Response(200, json={"result": {"uids": ["68011565"], "68011565": {"ds_meshterms": [mesh_heading], "ds_scopenote": ""}}}))


@respx.mock
async def test_name_only_resolves_both_ids(settings):
    _mock_resolution(icd_rows=[["L40.0", "Psoriasis vulgaris"]])
    async with Http(settings) as http:
        cond = await resolve_condition(ConditionInput(condition_term="Plaque Psoriasis"), http, settings)
    assert cond.mesh_id == "D011565" and cond.icd10_code == "L40.0"
    assert any("suggested" in n for n in cond.notes)


@respx.mock
async def test_malformed_ids_are_ignored_not_fatal(settings):
    _mock_resolution()
    async with Http(settings) as http:
        cond = await resolve_condition(ConditionInput(condition_term="Psoriasis", icd10_code="psoriasis", mesh_id="12345"), http, settings)
    assert cond.mesh_heading == "Psoriasis"
    assert sum("ignored" in n for n in cond.notes) == 2


@respx.mock
async def test_mismatched_icd_is_flagged(settings):
    _mock_resolution(icd_rows=[["L40.0", "Psoriasis vulgaris"]])
    async with Http(settings) as http:
        cond = await resolve_condition(ConditionInput(condition_term="Asthma", icd10_code="L40.0"), http, settings)
    assert cond.display_name == "Asthma" and cond.trial_query_term == "Asthma"
    assert any("may not match" in n for n in cond.notes)


def test_segments():
    assert ConditionInput().all_patients and ConditionInput().segment_description() == ALL_PATIENTS
    assert ConditionInput(patient_segments=[ALL_PATIENTS]).all_patients
    inp = ConditionInput(patient_segments=["Severity: Moderate-to-severe", "Age: Adolescent (12-17 y)", "Age: Pediatric (<12 y)"],
                         target_patient_segment="scalp-predominant")
    assert not inp.all_patients
    assert inp.segment_description() == "Moderate-to-severe; Adolescent (12-17 y); Pediatric (<12 y); scalp-predominant"
    assert inp.std_ages() == ["CHILD"]
    legacy = ConditionInput(target_patient_segment="Moderate-to-Severe")  # reports saved before the dropdown
    assert legacy.segment_description() == "Moderate-to-Severe"


def test_age_segments_filter_registry_search():
    term = ClinicalTrialsAdapter.build_term(["PHASE3"], std_ages=["CHILD", "OLDER_ADULT"])
    assert term.endswith("AREA[StdAge](CHILD OR OLDER_ADULT)")
