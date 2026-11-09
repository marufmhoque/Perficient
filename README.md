# Perficient · Biomedical & Clinical Strategy Intelligence Platform

Interview Building Challenge: an agentic workflow that turns a medical condition (name, ICD-10 or MeSH) into an
executive **competitive-intelligence briefing**. It covers epidemiology, standard of care, the active pipeline,
failed trials, and the stakeholder landscape, and every claim is cited to a real PubMed or ClinicalTrials.gov record.

## Architecture

```
            Condition / ICD-10 / MeSH
                       │  resolve (NLM ICD-10-CM · NCBI MeSH)
          ┌────────────┴────────────┐        run in parallel (asyncio.gather)
          ▼                         ▼
   Agent 1 · Clinical & SoC   Agent 2 · Pipeline & Failures
   PubMed E-utilities         ClinicalTrials.gov API v2 · SEC EDGAR
   → Module 1, Module 2       → Module 3A, 3B, 4 (+ comparator arms)
          └────────────┬────────────┘
                       ▼
   Finalize: pipeline vs SoC (needs Module 2) · SoC labels for active comparators · KOL merge
                       ▼
   Citation gate #1 (analyst modules)  →  Aggregator (exec summary, catalysts, terminology)
                       ▼
   Citation gate #2 (executive summary) → Briefing (tabs · Markdown · JSON · shared library)
```

| Path | Role |
| --- | --- |
| `streamlit_app.py` | UI: query form, live agent progress, report tabs, history sidebar, report library page |
| `biointel/orchestrator.py` | Stage sequencing, 170 s budget, graceful degradation per module |
| `biointel/agents/` | `clinical_soc.py` (Agent 1), `pipeline.py` (Agent 2), `aggregator.py` |
| `biointel/sources/` | Adapters: PubMed, ClinicalTrials.gov, SEC EDGAR, ICD-10/MeSH resolver, WHO ICTRP stub |
| `biointel/trials.py` | Deterministic trial parsing: comparator arms, p-values, failure detection |
| `biointel/stakeholders.py` | Deterministic sponsor / center / KOL rankings |
| `biointel/citations.py` · `validation.py` | Source registry and the zero-hallucination citation gate |
| `biointel/history.py` | Shared in-memory history of searches and reports, JSON import/export |
| `biointel/render/` | Markdown export, Plotly charts, Graphviz treatment flowchart, Streamlit views |

### Design choices
- **Facts come from code, judgement comes from the LLM.** NCT IDs, phases, statuses, sponsors, dates, stop reasons,
  p-values, investigators and PMIDs are copied directly from API responses. Claude (via the `anthropic` SDK with
  structured outputs) only classifies and synthesizes: modality/target, line of therapy, failure category, takeaways
  and narrative. Agents use `claude-sonnet-5-5`; the aggregator uses `claude-opus-5-5`.
- **Citations are checked against what was retrieved.** Every retrieved record gets a key (`P#` PubMed, `T#` trial,
  `S#` SEC filing). Claude may cite only those keys. The validator then:
  - normalizes raw PMIDs and NCT IDs to keys;
  - verifies unknown IDs live, so a real record is kept and a fabricated one is removed;
  - scrubs unverifiable IDs from free text;
  - drops any statement left without a source.

  The analyst modules are validated *before* the aggregator runs, so the executive summary is never built on
  unsupported claims.
- **Comparator arm for every trial.** Comparators are derived from the ClinicalTrials.gov arm-group types. Each trial
  is classed as placebo (or placebo plus background therapy), active comparator, sham, no treatment,
  dose-comparison, single-arm or unclear. The classifier also handles messy registry data: within-arm placebo
  ("drug X or matched placebo"), sponsors who label their own doses as "active comparator", placebo-to-drug
  crossover arms, and double-dummy designs.

  Active comparators are then checked against the Module 2 standard of care. Claude labels the remaining agents,
  e.g. sponsor code names such as BMS-986165, which is deucravacitinib.
- **Failure audit.** This combines terminated, suspended and withdrawn trials (with their verbatim `whyStopped`)
  with completed trials whose primary-endpoint analyses were all non-significant. Those p-values are parsed from
  the results section, and the details are cross-checked against SEC filings.
- **Modular sources.** A new registry (e.g. the EU CTR, or patents) is a new adapter that registers `TrialRecord`s or
  `Source`s. Downstream agents and the aggregator don't change; see `sources/ictrp.py`.
- **History.** The library is shared across all viewers and held in process memory, so it resets on restart.
  Reports can be downloaded as JSON and re-imported. Files in `samples/*.json` are preloaded as example reports.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
copy .streamlit\secrets.toml.example .streamlit\secrets.toml   # then add your ANTHROPIC_API_KEY
streamlit run streamlit_app.py
```

## Deploy to Streamlit Community Cloud

1. On https://share.streamlit.io, click **Create app** and choose repo `marufmhoque/Perficient`, branch `main`,
   main file `streamlit_app.py`.
2. Under **Advanced settings**, choose **Python 3.12** and paste the contents of `.streamlit/secrets.toml.example`
   with your real `ANTHROPIC_API_KEY`. You can also add these later under **Settings → Secrets**.
3. Deploy. Every push to `main` redeploys automatically. Without a key, the app still opens saved and example
   reports, but new briefings are disabled.

## Tests

```bash
pytest                                  # 52 offline tests: recorded API fixtures + fake LLM, no key needed
python scripts/smoke.py "Plaque Psoriasis" --icd10 L40.0    # live run: real APIs + Claude, < 180 s,
                                                            # re-resolves every cited PMID / NCT ID
python scripts/build_fixtures.py        # refresh the recorded fixtures (no site-contact data)
```

The offline suite covers comparator classification edge cases, p-value parsing, adapter parsing and pagination,
the citation gate (fabricated IDs removed, real ones verified live), stakeholder rankings, history and JSON
round-trips, Markdown rendering, and an end-to-end orchestration run. That run checks the two agents execute
concurrently and that no fabricated identifier survives.
