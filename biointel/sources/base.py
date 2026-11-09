"""Shared HTTP layer for all data adapters.

* one ``httpx.AsyncClient`` per run
* per-host minimum interval (NCBI: 3 req/s without key, 10 with; SEC: 10 req/s)
* retries with exponential backoff on 429 / 5xx / network errors
* process-wide TTL response cache (repeat searches are instant; shared across Streamlit sessions)
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Protocol
from urllib.parse import urlencode, urlparse

import httpx

from ..config import Settings

# (url with sorted params) -> (stored_at, text)
_RESPONSE_CACHE: dict[str, tuple[float, str]] = {}
_CACHE_MAX_ENTRIES = 2000


class SourceError(RuntimeError):
    pass


class _Retryable(Exception):
    pass


class SourceAdapter(Protocol):
    """Marker protocol: every data adapter exposes a human-readable name."""

    name: str


class _HostLimiter:
    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def wait(self) -> None:
        async with self._lock:
            delay = self._last + self.min_interval - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self._last = time.monotonic()


class Http:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.settings = settings
        self.client = httpx.AsyncClient(
            timeout=settings.http_timeout_s,
            follow_redirects=True,
            headers={"User-Agent": "BioIntelPrototype/0.1 (+https://github.com/marufmhoque/Perficient)"},
            transport=transport,
        )
        ncbi_interval = 0.11 if settings.ncbi_api_key else 0.36
        self._limiters: dict[str, _HostLimiter] = {
            "eutils.ncbi.nlm.nih.gov": _HostLimiter(ncbi_interval),
            "efts.sec.gov": _HostLimiter(0.12),
            "www.sec.gov": _HostLimiter(0.12),
            "clinicaltrials.gov": _HostLimiter(0.05),
        }
        self.request_count = 0
        self.cache_hits = 0

    async def __aenter__(self) -> "Http":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self.client.aclose()

    @staticmethod
    def build_url(url: str, params: dict[str, Any] | None = None) -> str:
        if not params:
            return url
        clean = {k: v for k, v in params.items() if v is not None}
        return f"{url}?{urlencode(sorted(clean.items()), doseq=True)}"

    async def get_text(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        use_cache: bool = True,
        retries: int = 3,
    ) -> str:
        full = self.build_url(url, params)
        ttl = self.settings.response_cache_ttl_s
        if use_cache and ttl > 0:
            hit = _RESPONSE_CACHE.get(full)
            if hit and time.time() - hit[0] < ttl:
                self.cache_hits += 1
                return hit[1]

        host = urlparse(url).netloc
        limiter = self._limiters.get(host)
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            if limiter:
                await limiter.wait()
            try:
                self.request_count += 1
                resp = await self.client.get(full, headers=headers)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise _Retryable(f"{host} HTTP {resp.status_code}")
                if resp.status_code >= 400:  # client error: not retried
                    raise SourceError(f"GET {full[:200]} -> HTTP {resp.status_code}: {resp.text[:200]}")
                text = resp.text
                if use_cache and ttl > 0:
                    if len(_RESPONSE_CACHE) >= _CACHE_MAX_ENTRIES:
                        _RESPONSE_CACHE.pop(next(iter(_RESPONSE_CACHE)))
                    _RESPONSE_CACHE[full] = (time.time(), text)
                return text
            except (httpx.TransportError, _Retryable) as exc:
                last_exc = exc
                if attempt == retries:
                    break
                await asyncio.sleep(min(8.0, 0.8 * (2**attempt)))
        raise SourceError(f"GET {full[:200]} failed after {retries + 1} attempts: {last_exc}")

    async def get_json(self, url: str, params: dict[str, Any] | None = None, **kw) -> Any:
        text = await self.get_text(url, params, **kw)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            _RESPONSE_CACHE.pop(self.build_url(url, params), None)
            raise SourceError(f"Invalid JSON from {url}: {exc}") from exc


def clear_response_cache() -> None:
    _RESPONSE_CACHE.clear()
