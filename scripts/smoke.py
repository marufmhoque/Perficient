"""Live end-to-end smoke test: real APIs + real Claude calls.

    python scripts/smoke.py "Plaque Psoriasis" --icd10 L40.0 [--save samples/plaque_psoriasis.json]

Prints per-stage timings, checks the 180 s budget, re-resolves every cited NCT ID and PMID
against the live registries (must be 100%), and checks that every module was produced.
Needs ANTHROPIC_API_KEY (environment or .streamlit/secrets.toml).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biointel.citations import SourceRegistry, cited_keys  # noqa: E402
from biointel.config import get_settings  # noqa: E402
from biointel.history import report_to_json  # noqa: E402
from biointel.llm import LLMAccountError  # noqa: E402
from biointel.orchestrator import run_briefing  # noqa: E402
from biointel.schemas import Briefing, ConditionInput  # noqa: E402
from biointel.sources import ClinicalTrialsAdapter, Http, PubMedAdapter  # noqa: E402


def load_key_from_secrets() -> str | None:
    path = ROOT / ".streamlit" / "secrets.toml"
    if path.exists():
        return tomllib.loads(path.read_text(encoding="utf-8")).get("ANTHROPIC_API_KEY")
    return None


async def verify_live(b: Briefing) -> tuple[int, int, list[str]]:
    """Re-resolve every cited PubMed / trial source against the live APIs."""
    smap = b.source_map()
    keys = [k for k in cited_keys(b) if k in smap]
    pmids = [smap[k].native_id for k in keys if smap[k].kind == "pubmed"]
    ncts = [smap[k].native_id for k in keys if smap[k].kind == "trial"]
    settings = get_settings(response_cache_ttl_s=0)
    failures: list[str] = []
    async with Http(settings) as http:
        reg = SourceRegistry()
        pm, ct = PubMedAdapter(http, reg, settings), ClinicalTrialsAdapter(http, reg, settings)
        found: set[str] = set()
        for i in range(0, len(pmids), 150):
            found |= set((await pm.existing(pmids[i : i + 150])).keys())
        failures += [f"PMID {p}" for p in pmids if p not in found]
        sem = asyncio.Semaphore(6)

        async def check(n: str):
            async with sem:
                return n, await ct.exists(n)

        for n, data in await asyncio.gather(*(check(n) for n in ncts)):
            if not data:
                failures.append(n)
    return len(pmids) + len(ncts), len(pmids) + len(ncts) - len(failures), failures


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("condition")
    ap.add_argument("--icd10")
    ap.add_argument("--mesh")
    ap.add_argument("--save", help="write the briefing JSON here (e.g. samples/<name>.json)")
    args = ap.parse_args()

    overrides = {}
    if not get_settings().anthropic_api_key and (key := load_key_from_secrets()):
        overrides["anthropic_api_key"] = key
    settings = get_settings(**overrides)
    if not settings.anthropic_api_key:
        print("ANTHROPIC_API_KEY not set (environment or .streamlit/secrets.toml).")
        return 2

    t0 = time.monotonic()

    def progress(stage: str, msg: str, state: str) -> None:
        print(f"[{time.monotonic() - t0:6.1f}s] {stage:<10} {state:<8} {msg}")

    inp = ConditionInput(condition_term=args.condition, icd10_code=args.icd10, mesh_id=args.mesh)
    try:
        b = await run_briefing(inp, settings=settings, progress=progress)
    except LLMAccountError as exc:
        print(f"\nSMOKE TEST NOT RUN - Claude account problem: {exc}")
        return 2

    print("\n=== Timings (s) ===", b.timings)
    print("=== Token usage ===", b.usage)
    modules = {
        "executive_summary": b.executive_summary, "module1": b.module1, "module2": b.module2,
        "module3a": b.module3a, "module3b": b.module3b, "module4": b.module4,
    }
    for name, m in modules.items():
        print(f"{name:<18} {'OK' if m else 'MISSING'}")
    if b.module3a:
        from collections import Counter

        print("comparator mix:", Counter(t.comparator_class for t in b.module3a.trials).most_common())
    print("errors:", [f"{e.module}: {e.message}" for e in b.errors] or "none")
    v = b.validation
    print(f"validation: {v.ids_valid}/{v.ids_checked} ids valid, {v.ids_resolved_live} resolved live, {v.statements_dropped} items dropped")
    if b.fact_check:
        print(f"fact check: {b.fact_check.counts()} over {len(b.fact_check.claims)} claims; "
              f"source issues: {[(i.key, i.severity) for i in b.fact_check.source_issues]}")
    else:
        print("fact check: MISSING")

    checked, ok, failures = await verify_live(b)
    print(f"live re-resolution of cited sources: {ok}/{checked}" + (f"  FAILED: {failures[:10]}" if failures else ""))

    if args.save:
        out = ROOT / args.save
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report_to_json(b), encoding="utf-8")
        print("saved", out)

    passed = b.timings.get("total", 999) < 180 and not failures and all(modules.values())
    print("\nSMOKE TEST", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
