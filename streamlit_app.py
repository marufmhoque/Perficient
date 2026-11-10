"""Biomedical & Clinical Strategy Intelligence Platform - Streamlit front end.

Pages: New briefing (query form, live agent progress, report tabs) and Report library
(every saved report, shared across viewers; open / download / delete / import).
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import streamlit as st

from biointel.config import get_settings, secrets_diagnostic
from biointel.history import InMemoryHistory, new_state, report_from_json, report_to_json
from biointel.factcheck import fact_check_briefing
from biointel.llm import ClaudeLLM, LLMAccountError, LLMError
from biointel.orchestrator import STAGES, run_briefing
from biointel.render.markdown import briefing_to_markdown
from biointel.render.streamlit_view import render_briefing, slug
from biointel.schemas import ALL_PATIENTS, PATIENT_SEGMENTS, PHASES, ConditionInput
from biointel.sources.resolve import valid_icd10, valid_mesh

st.set_page_config(page_title="BioIntel · Clinical Strategy Intelligence", page_icon="🧬", layout="wide")

SAMPLES_DIR = Path(__file__).parent / "samples"
GEOGRAPHIES = ["Global", "US", "EU5", "UK", "Japan", "China"]
PHASE_LABELS = {"EARLY_PHASE1": "Early Phase 1", "PHASE1": "Phase 1", "PHASE2": "Phase 2", "PHASE3": "Phase 3", "PHASE4": "Phase 4"}
FORM_DEFAULTS = {
    "f_condition": "Plaque Psoriasis",
    "f_icd10": "",  # optional: resolved from the name when blank
    "f_mesh": "",  # optional: resolved from the name when blank
    "f_geo": ["US", "EU5"],
    "f_segments": [ALL_PATIENTS],
    "f_segment_notes": "",
    "f_phases": ["PHASE1", "PHASE2", "PHASE3"],
    "f_horizon": 5,
    "f_discontinued": True,
}


@st.cache_resource
def get_history_state() -> dict:
    """One library per app process, shared by every viewer.

    Only plain data is cached: Streamlit Cloud reloads changed modules without restarting the
    process, and cached model instances or methods from the old code then clash with the new
    classes (pydantic ValidationError). The facade below is rebuilt from current code every run."""
    return new_state()


def get_history() -> InMemoryHistory:
    s = get_settings()
    return InMemoryHistory(max_reports=s.history_max_reports, max_searches=s.history_max_searches, state=get_history_state())


def load_examples(history: InMemoryHistory) -> None:
    """Add bundled example reports that this process hasn't loaded yet.

    Runs on every page load (cheap: one stat per file) because Streamlit Cloud can update the code
    without restarting the process, so a library cached before a sample was added would never see it."""
    for path in sorted(SAMPLES_DIR.glob("*.json")):
        if path.name.startswith("_"):
            continue  # local test artefacts
        mtime = path.stat().st_mtime
        if history.example_files.get(path.name) == mtime:
            continue
        try:
            briefing = report_from_json(path.read_text(encoding="utf-8"))
            briefing.is_example = True
            history.save_report(briefing)
            history.example_errors.pop(path.name, None)
        except Exception as exc:  # a bad sample must never break the app
            history.example_errors[path.name] = f"{type(exc).__name__}: {str(exc)[:300]}"
        history.example_files[path.name] = mtime


history = get_history()
load_examples(history)
settings = get_settings()


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

for key, value in FORM_DEFAULTS.items():
    st.session_state.setdefault(key, value)
st.session_state.setdefault("current_report_id", None)


def fill_form(inp: ConditionInput) -> None:
    segments = [s for s in inp.patient_segments if s in PATIENT_SEGMENTS]
    st.session_state.update(
        f_condition=inp.condition_term,
        f_icd10=inp.icd10_code or "",
        f_mesh=inp.mesh_id or "",
        f_geo=[g for g in inp.geographies if g in GEOGRAPHIES],
        f_segments=segments or ([] if inp.target_patient_segment else [ALL_PATIENTS]),
        f_segment_notes=inp.target_patient_segment or "",
        f_phases=[p for p in inp.development_phases if p in PHASES],
        f_horizon=inp.time_horizon_years,
        f_discontinued=inp.include_discontinued,
    )


def build_input() -> tuple[ConditionInput | None, list[str]]:
    """Read the form; return (input, problems). ICD-10 and MeSH are optional but must be well-formed if given."""
    ss = st.session_state
    problems = []
    icd = ss.f_icd10.strip().upper()
    mesh = ss.f_mesh.strip().upper()
    if icd and not valid_icd10(icd):
        problems.append(f"ICD-10 code “{icd}” isn't a valid format (e.g. L40.0, E11, E11.65). Fix it or leave the field blank.")
    if mesh and not valid_mesh(mesh):
        problems.append(f"MeSH ID “{mesh}” isn't a valid format (e.g. D011565). Fix it or leave the field blank.")
    if not (ss.f_condition.strip() or icd or mesh):
        problems.append("Enter a condition name, or an ICD-10 code / MeSH ID.")
    picks = [s for s in ss.f_segments if s != ALL_PATIENTS] if ALL_PATIENTS not in ss.f_segments or len(ss.f_segments) > 1 else []
    if problems:
        return None, problems
    return ConditionInput(
        condition_term=ss.f_condition.strip(),
        icd10_code=icd or None,
        mesh_id=mesh or None,
        geographies=ss.f_geo,
        patient_segments=picks or [ALL_PATIENTS],
        target_patient_segment=ss.f_segment_notes.strip() or None,
        development_phases=ss.f_phases or ["PHASE1", "PHASE2", "PHASE3"],
        time_horizon_years=int(ss.f_horizon),
        include_discontinued=bool(ss.f_discontinued),
    ), []


def open_report(report_id: str) -> None:
    st.session_state["current_report_id"] = report_id


def history_sidebar() -> None:
    st.sidebar.divider()
    st.sidebar.markdown("#### 🕘 History")
    st.sidebar.caption("Shared across all viewers · resets when the app restarts")
    searches = history.recent_searches(8)
    if searches:
        st.sidebar.markdown("**Recent searches** · click to refill the form")
        for i, s in enumerate(searches):
            st.sidebar.button(
                f"↻ {s.label}", key=f"search_{i}", on_click=fill_form, args=(s.input,), width="stretch",
                help=f"{s.at:%Y-%m-%d %H:%M} UTC · phases {', '.join(PHASE_LABELS.get(p, p) for p in s.input.development_phases)}",
            )
    reports = history.list_reports()[:8]
    if reports:
        st.sidebar.markdown("**Saved reports** · click to open")
        for r in reports:
            tag = " · example" if r.is_example else ""
            st.sidebar.button(
                f"📄 {r.title}", key=f"report_{r.id}", on_click=open_report, args=(r.id,), width="stretch",
                help=f"{r.created_at:%Y-%m-%d %H:%M} UTC · {r.n_trials} trials · {r.citations_valid} citations verified{tag}",
            )
    if not searches and not reports:
        st.sidebar.caption("No searches yet.")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def run_with_progress(inp: ConditionInput):
    """Run the agent pipeline, streaming stage updates into st.status panels."""
    st.markdown("#### Running agents")
    elapsed = st.empty()
    panels: dict[str, object] = {}
    t0 = time.monotonic()
    order = list(STAGES)
    slots = {stage: st.empty() for stage in order}

    def progress(stage: str, message: str, state: str) -> None:
        if stage not in panels:
            panels[stage] = slots[stage].status(STAGES.get(stage, stage), state="running", expanded=False)
        panel = panels[stage]
        if state == "running":
            panel.write(message)
            panel.update(label=f"{STAGES.get(stage, stage)} — {message}", state="running")
        else:
            panel.update(label=f"{STAGES.get(stage, stage)} — {message}", state="complete" if state == "complete" else "error")
        elapsed.caption(f"⏱️ {time.monotonic() - t0:.0f} s elapsed · PRD budget 180 s · Agents 1 and 2 run in parallel")

    return asyncio.run(run_briefing(inp, settings=settings, progress=progress))


def fact_check_existing(briefing) -> None:
    """Fact-check a saved / imported report that has none yet, then save the result to the library."""
    llm = ClaudeLLM(settings)

    async def go():
        await llm.preflight()
        return await fact_check_briefing(briefing, settings, llm)

    with st.spinner("Re-fetching every cited source and checking each claim against it (about 30-60 s)…"):
        try:
            report = asyncio.run(go())
        except LLMAccountError as exc:
            st.error(f"**The fact check couldn't run:** {exc}", icon="💳")
            return
        except Exception as exc:
            st.error(f"Fact check failed: {exc}")
            return
    updated = briefing.model_copy(deep=True)
    updated.fact_check = report
    history.save_report(updated)
    st.rerun()


def briefing_page() -> None:
    with st.sidebar:
        st.markdown("### 🔎 New briefing")
        with st.form("query"):
            st.text_input("Condition / disease", key="f_condition", placeholder="e.g. Plaque Psoriasis")
            c1, c2 = st.columns(2)
            c1.text_input("ICD-10 · optional", key="f_icd10", placeholder="e.g. L40.0",
                          help="Leave blank to resolve it from the condition name. Enter a code to pin the exact condition.")
            c2.text_input("MeSH ID · optional", key="f_mesh", placeholder="e.g. D011565",
                          help="Leave blank to resolve it from the condition name. Enter an ID to pin the literature search.")
            st.multiselect("Geographies", GEOGRAPHIES, key="f_geo")
            st.multiselect(
                "Target patient segment", PATIENT_SEGMENTS, key="f_segments", placeholder="All patients",
                help="Pick one or more segments, or “All patients”. Age segments also filter the trial registry search; "
                     "the others focus the epidemiology, standard-of-care and differentiation analysis.",
            )
            st.text_input("Other segment details · optional", key="f_segment_notes",
                          placeholder="e.g. scalp-predominant disease, HLA-B27+, post-MI",
                          help="Free text for any segment not in the list; combined with the selections above.")
            st.multiselect("Development phases", PHASES, key="f_phases", format_func=lambda p: PHASE_LABELS.get(p, p))
            st.slider("Time horizon (years)", 1, 15, key="f_horizon")
            st.checkbox("Include discontinued / failed trials", key="f_discontinued")
            submitted = st.form_submit_button(
                "Generate briefing", type="primary", width="stretch", disabled=not settings.anthropic_api_key
            )
        if not settings.anthropic_api_key:
            st.warning(
                "`ANTHROPIC_API_KEY` is not configured, so new briefings are disabled. Saved and example reports still open.\n\n"
                f"**Why:** {secrets_diagnostic()}\n\n"
                "**Fix:** on Streamlit Cloud open **⋮ → Settings → Secrets**, paste the line below (with the quotes), "
                "click **Save**, then **⋮ → Reboot app**. Locally, put it in `.streamlit/secrets.toml`.\n\n"
                '`ANTHROPIC_API_KEY = "sk-ant-..."`',
                icon="🔑",
            )
    history_sidebar()

    st.title("Biomedical & Clinical Strategy Intelligence")
    st.caption(
        "Agent 1 (PubMed · guidelines) and Agent 2 (ClinicalTrials.gov · SEC EDGAR) run in parallel; the Aggregator writes the "
        "executive brief and every PMID / NCT ID is verified before display."
    )

    if submitted:
        inp, problems = build_input()
        if problems:
            for p in problems:
                st.error(p, icon="✏️")
            return
        if ALL_PATIENTS in st.session_state.f_segments and len(st.session_state.f_segments) > 1:
            st.caption("“All patients” was combined with specific segments - the specific segments were used.")
        try:
            briefing = run_with_progress(inp)
        except LLMAccountError as exc:
            st.error(f"**The briefing couldn't run:** {exc}", icon="💳")
            st.caption("Nothing was saved. Retrieval stops before any work when the Claude account can't serve requests.")
            return
        except LLMError as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # surface unexpected failures without killing the session
            st.exception(exc)
            return
        history.save_report(briefing)
        history.add_search(inp, report_id=briefing.id)
        st.session_state["current_report_id"] = briefing.id
        st.rerun()

    current = history.get_report(st.session_state["current_report_id"]) if st.session_state["current_report_id"] else None
    if current:
        render_briefing(current, on_fact_check=fact_check_existing if settings.anthropic_api_key else None)
        return

    st.info("Enter a condition in the sidebar and click **Generate briefing** (typically 1–3 minutes), or open a saved report.", icon="👈")
    reports = history.list_reports()
    if reports:
        st.markdown("#### Saved reports")
        for r in reports[:6]:
            cols = st.columns([4, 2, 3, 1.4], vertical_alignment="center")
            cols[0].markdown(f"**{r.title}**" + (" · *example*" if r.is_example else ""))
            cols[1].caption(f"{r.created_at:%Y-%m-%d %H:%M} UTC")
            cols[2].caption(f"{r.n_trials} trials · {r.n_pubmed} PubMed · {r.citations_valid} cited IDs verified")
            cols[3].button("Open", key=f"home_open_{r.id}", on_click=open_report, args=(r.id,), width="stretch")


def library_page() -> None:
    history_sidebar()
    st.title("📚 Report library")
    for name, err in history.example_errors.items():
        st.error(f"Bundled example `{name}` could not be loaded: {err}", icon="📦")
    st.info(
        "Reports are shared with everyone using this app and kept in memory: they reset when the app restarts or redeploys. "
        "Download the JSON of any report you want to keep, then import it here later.",
        icon="ℹ️",
    )

    with st.expander("⬆️ Import a report (.json)"):
        upload = st.file_uploader("Report JSON exported from this app", type=["json"], key="import_json")
        if upload is not None and st.button("Import report", type="primary"):
            try:
                briefing = report_from_json(upload.getvalue())
            except Exception as exc:
                st.error(f"Not a valid report file: {exc}")
            else:
                briefing.is_example = False
                history.save_report(briefing)
                st.success(f"Imported “{briefing.condition.display_name}”.")

    reports = history.list_reports()
    if not reports:
        st.caption("No reports yet - generate one from the New briefing page.")
        return

    st.dataframe(
        [
            {"Condition": r.title + (" (example)" if r.is_example else ""), "ICD-10": r.icd10 or "", "MeSH": r.mesh or "",
             "Created (UTC)": r.created_at.strftime("%Y-%m-%d %H:%M"), "Run time (s)": round(r.duration_s or 0),
             "Active trials": r.n_trials, "Failures": r.n_failures, "PubMed sources": r.n_pubmed,
             "All sources": r.n_sources, "Citations verified": r.citations_valid}
            for r in reports
        ],
        hide_index=True, width="stretch",
    )

    labels = {r.id: f"{r.title} · {r.created_at:%Y-%m-%d %H:%M}" + (" · example" if r.is_example else "") for r in reports}
    chosen = st.selectbox("Select a report", list(labels), format_func=labels.get)
    briefing = history.get_report(chosen) if chosen else None
    if not briefing:
        return
    name = f"briefing_{slug(briefing.condition.display_name)}_{briefing.created_at:%Y%m%d}"
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Open report", type="primary", width="stretch"):
        st.session_state["current_report_id"] = briefing.id
        st.switch_page(PAGES["briefing"])
    c2.download_button("⬇️ Markdown", briefing_to_markdown(briefing), file_name=f"{name}.md", mime="text/markdown", width="stretch")
    c3.download_button("⬇️ JSON", report_to_json(briefing), file_name=f"{name}.json", mime="application/json", width="stretch")
    with c4.popover("🗑️ Delete", width="stretch"):
        st.caption("Removes the report for every viewer of this app. Download it first if you may need it.")
        if st.button("Delete permanently", key=f"del_{briefing.id}", type="primary"):
            history.delete_report(briefing.id)
            if st.session_state.get("current_report_id") == briefing.id:
                st.session_state["current_report_id"] = None
            st.rerun()


PAGES = {
    "briefing": st.Page(briefing_page, title="New briefing", icon="🧬", default=True),
    "library": st.Page(library_page, title="Report library", icon="📚"),
}
st.navigation(list(PAGES.values())).run()
