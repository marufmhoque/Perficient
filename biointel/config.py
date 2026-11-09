"""Runtime settings. Secrets come from environment variables first, then Streamlit secrets."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _secret(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    try:  # Streamlit is optional (tests / CLI run without it)
        import streamlit as st

        return st.secrets.get(name, default)  # type: ignore[no-any-return]
    except Exception:
        return default


@dataclass(frozen=True)
class Settings:
    # --- Models (Claude) ---
    agent_model: str = "claude-sonnet-5-5"
    aggregator_model: str = "claude-opus-5-5"
    agent_effort: str = "medium"
    aggregator_effort: str = "medium"
    max_tokens: int = 16000

    # --- Retrieval limits (keep the run under the 180 s NFR) ---
    pubmed_guidelines_max: int = 12
    pubmed_treatment_max: int = 15
    pubmed_epi_max: int = 12
    pubmed_patho_max: int = 8
    abstract_char_limit: int = 2500
    ctgov_active_max: int = 150
    ctgov_stopped_max: int = 60
    ctgov_results_max: int = 40
    sec_assets_max: int = 6
    llm_assets_max: int = 120

    # --- Time budget (seconds) ---
    total_budget_s: float = 170.0
    http_timeout_s: float = 25.0
    response_cache_ttl_s: float = 24 * 3600

    # --- Behaviour ---
    drop_unsourced: bool = True  # drop (True) or flag (False) statements without valid citations
    history_max_reports: int = 50
    history_max_searches: int = 25

    # --- Secrets ---
    anthropic_api_key: str | None = field(default=None, repr=False)
    ncbi_api_key: str | None = field(default=None, repr=False)
    sec_user_agent: str = "BioIntelPrototype/0.1 (research prototype)"


def get_settings(**overrides) -> Settings:
    base = dict(
        anthropic_api_key=_secret("ANTHROPIC_API_KEY"),
        ncbi_api_key=_secret("NCBI_API_KEY"),
        sec_user_agent=_secret("SEC_USER_AGENT", Settings.sec_user_agent) or Settings.sec_user_agent,
    )
    for key in ("AGENT_MODEL", "AGGREGATOR_MODEL"):
        value = _secret(key)
        if value:
            base[key.lower()] = value
    base.update(overrides)
    return Settings(**base)
