"""SEC EDGAR full-text search (efts.sec.gov) - corroborating filings for discontinued assets."""

from __future__ import annotations

from ..citations import SourceRegistry
from ..config import Settings
from ..schemas import Source
from .base import Http

EFTS = "https://efts.sec.gov/LATEST/search-index"


class SecEdgarAdapter:
    name = "SEC EDGAR"

    def __init__(self, http: Http, registry: SourceRegistry, settings: Settings) -> None:
        self.http = http
        self.registry = registry
        self.settings = settings

    async def search_asset(self, asset: str, purpose: str, max_hits: int = 2) -> list[Source]:
        asset = asset.strip()
        if len(asset) < 4:
            return []
        params = {"q": f'"{asset}"', "forms": "10-K,10-Q,8-K"}
        data = await self.http.get_json(EFTS, params, headers={"User-Agent": self.settings.sec_user_agent})
        hits = (data.get("hits") or {}).get("hits") or []
        total = ((data.get("hits") or {}).get("total") or {}).get("value", len(hits))
        self.registry.log_query("SEC EDGAR", purpose, f'"{asset}" in 10-K/10-Q/8-K', Http.build_url(EFTS, params), int(total))
        out: list[Source] = []
        for hit in hits[:max_hits]:
            src = hit.get("_source") or {}
            try:
                cik = int((src.get("ciks") or ["0"])[0])
                adsh = src["adsh"]
                filename = hit["_id"].split(":", 1)[1]
            except (KeyError, IndexError, ValueError):
                continue
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{adsh.replace('-', '')}/{filename}"
            company = (src.get("display_names") or ["Unknown filer"])[0]
            form = src.get("form") or (src.get("root_forms") or ["filing"])[0]
            out.append(
                self.registry.add(
                    "sec", f"{adsh}:{filename}", f"{form} - {company}", url,
                    detail=f"Filed {src.get('file_date', '?')}; mentions \"{asset}\"",
                )
            )
        return out
