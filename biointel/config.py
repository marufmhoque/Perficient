"""Runtime settings. Secrets come from environment variables first, then Streamlit secrets."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _clean(value) -> str | None:
    """Strip whitespace and stray quotes that often sneak in when pasting a key into a secrets box."""
    if value is None or not isinstance(value, (str, int, float)):
        return None
    return str(value).strip().strip('"').strip("'").strip() or None


def _is_placeholder(value: str | None) -> bool:
    """Template values such as 'sk-ant-...' must never shadow a real secret."""
    v = _clean(value)
    return not v or v.endswith("...") or v in ("<your-key>", "changeme")


# Accepted spellings per secret (case-insensitive; "section.key" = a key inside a [section] table)
_ALIASES: dict[str, tuple[str, ...]] = {
    "ANTHROPIC_API_KEY": ("anthropic_api_key", "anthropic.api_key", "anthropic.anthropic_api_key", "claude_api_key",
                          "claude.api_key", "api_keys.anthropic", "api_keys.anthropic_api_key"),
}


def _flat_streamlit_secrets() -> dict[str, object]:
    """st.secrets flattened to lower-case 'name' and 'section.name' keys. Raises if secrets can't be read."""
    import streamlit as st

    flat: dict[str, object] = {}
    for key in st.secrets.keys():
        value = st.secrets[key]
        if hasattr(value, "keys"):  # a [section] table
            for sub in value.keys():
                flat[f"{key}.{sub}".lower()] = value[sub]
        else:
            flat[key.lower()] = value
    return flat


def _secret(name: str, default: str | None = None) -> str | None:
    for env_name in (name, name.lower()):
        value = os.environ.get(env_name)
        if not _is_placeholder(value):
            return _clean(value)
    try:  # Streamlit is optional (tests / CLI run without it)
        flat = _flat_streamlit_secrets()
    except Exception:
        return default
    for candidate in (name.lower(), *_ALIASES.get(name, ())):
        if not _is_placeholder(flat.get(candidate)):
            return _clean(flat[candidate])
    return default


def secrets_diagnostic(name: str = "ANTHROPIC_API_KEY") -> str:
    """Explain why a secret wasn't found, naming what *is* configured. Never reveals values."""
    env = os.environ.get(name)
    if env is not None and _is_placeholder(env):
        return f"The environment variable {name} is set to a placeholder ('{_clean(env) or 'empty'}'), and no real key was found."
    try:
        flat = _flat_streamlit_secrets()
    except Exception as exc:
        msg = str(exc)
        if "pars" in msg.lower():  # Streamlit reports invalid TOML with the same "not found" exception type
            where = msg[msg.rfind(":") + 1:].strip() if ":" in msg else ""
            return ("The secrets could not be parsed as TOML" + (f" - {where}" if where else "") + ". "
                    "The usual cause is a key pasted without quotes: write ANTHROPIC_API_KEY = \"sk-ant-...\".")
        if isinstance(exc, FileNotFoundError):
            return "No secrets are configured for this app (no secrets.toml locally / nothing saved under Settings -> Secrets)."
        return f"The app's secrets could not be read ({type(exc).__name__})."
    if not flat:
        return "The secrets store is empty: nothing has been saved under Settings -> Secrets."
    for candidate in (name.lower(), *_ALIASES.get(name, ())):
        if candidate in flat:
            return f"A secret named '{candidate}' exists but its value is empty or a placeholder."
    names = ", ".join(sorted(flat))
    return f"Secrets were found ({names}), but none is named {name}."


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

    # --- Fact check (claim-by-claim grounding against cited source text) ---
    fact_check: bool = True
    fact_check_model: str | None = None  # defaults to agent_model
    fact_check_effort: str = "medium"

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
