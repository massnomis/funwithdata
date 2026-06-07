"""
Stride API client.

A thin, typed wrapper around the Open Bus Stride API. Centralizes:
    * connection pooling and retry/backoff
    * cursor-based pagination for `*/list` endpoints
    * concurrent fan-out for "fetch N routes' details in parallel" patterns
    * structured exceptions so the UI layer can decide how to surface errors

The client is intentionally Streamlit-agnostic. Streamlit caching wrappers
live in `data_processor.py` / `ui_components.py` so this layer remains
unit-testable in isolation.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    API_BASE_URL,
    BATCH_SIZE,
    HARD_PAGE_LIMIT,
    REQUEST_TIMEOUT_SEC,
)

log = logging.getLogger(__name__)

JSON = dict[str, Any]
Params = Mapping[str, Any]


# ---------------------------------------------------------------------------
# Exception hierarchy — let callers distinguish transient from fatal
# ---------------------------------------------------------------------------

class StrideAPIError(Exception):
    """Base error for all client failures."""


class StrideTimeoutError(StrideAPIError):
    """Server didn't respond in time. Usually transient — retry with smaller window."""


class StrideValidationError(StrideAPIError):
    """HTTP 422 — the request shape itself is wrong."""

    def __init__(self, message: str, details: Any = None) -> None:
        super().__init__(message)
        self.details = details


class StrideServerError(StrideAPIError):
    """5xx after exhausting retries."""


# ---------------------------------------------------------------------------
# Health probe result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HealthStatus:
    ok: bool
    latency_ms: float | None
    detail: str


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class StrideClient:
    """Connection-pooled HTTP client for the Stride API.

    Lifecycle: instantiate once per process. Streamlit users should wrap
    construction in `@st.cache_resource` so a single client persists across
    reruns instead of opening a fresh socket pool every interaction.
    """

    def __init__(
        self,
        base_url: str = API_BASE_URL,
        timeout: int = REQUEST_TIMEOUT_SEC,
        max_pool: int = 50,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = self._build_session(max_pool)

    @staticmethod
    def _build_session(max_pool: int) -> requests.Session:
        session = requests.Session()
        # Exponential backoff: 0.5, 1.0, 2.0, 4.0, 8.0 seconds.
        # 429 / 5xx retried; the underlying urllib3 also honors Retry-After.
        retry = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=max_pool,
            pool_maxsize=max_pool,
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({"Accept": "application/json"})
        return session

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def get(self, endpoint: str, params: Params | None = None) -> Any:
        """Single GET. Returns parsed JSON. Raises a `StrideAPIError` subclass on failure."""
        params = self._clean_params(params)
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            resp = self._session.get(url, params=params, timeout=self.timeout)
        except requests.exceptions.ReadTimeout as e:
            raise StrideTimeoutError(f"{endpoint} timed out after {self.timeout}s") from e
        except requests.exceptions.RequestException as e:
            raise StrideAPIError(f"{endpoint} request failed: {e}") from e

        if resp.status_code == 422:
            details: Any = None
            try:
                details = resp.json()
            except ValueError:
                details = resp.text
            raise StrideValidationError(f"{endpoint}: invalid parameters", details)

        if resp.status_code >= 500:
            raise StrideServerError(f"{endpoint}: {resp.status_code} {resp.reason}")

        if not resp.ok:
            raise StrideAPIError(f"{endpoint}: {resp.status_code} {resp.reason}")

        try:
            return resp.json()
        except ValueError as e:
            raise StrideAPIError(f"{endpoint}: response was not valid JSON") from e

    def list_all(
        self,
        endpoint: str,
        params: Params | None = None,
        *,
        page_size: int = BATCH_SIZE,
        on_progress: Callable[[int], None] | None = None,
    ) -> list[JSON]:
        """List endpoint with auto-pagination.

        Stride rejects `limit > 15000` with HTTP 500, so callers cannot bypass
        pagination with a "give me everything" sentinel — we always page.

        Args:
            endpoint: The `gtfs_*/list` or `siri_*/list` endpoint.
            params: Query parameters (any `None` values are dropped).
            page_size: Rows per request. Must stay under the server-side abuse cap.
            on_progress: Optional `(records_so_far: int) -> None` for progress bars.
        """
        out: list[JSON] = []
        offset = 0
        for page in range(HARD_PAGE_LIMIT):
            page_params = dict(params or {})
            page_params["offset"] = offset
            page_params["limit"] = page_size

            chunk = self.get(endpoint, page_params)
            if not isinstance(chunk, list) or not chunk:
                break
            out.extend(chunk)
            if on_progress is not None:
                on_progress(len(out))
            if len(chunk) < page_size:
                break
            offset += page_size
        else:
            log.warning(
                "list_all(%s) hit HARD_PAGE_LIMIT=%d (%d rows). Narrow your filters.",
                endpoint, HARD_PAGE_LIMIT, len(out),
            )
        return out

    def fan_out(
        self,
        calls: Iterable[tuple[str, Params]],
        *,
        max_workers: int = 8,
    ) -> list[tuple[Params, Any | StrideAPIError]]:
        """Issue many GETs concurrently.

        Returns a list of `(params, result_or_exception)` tuples in completion
        order — not input order. Exceptions are returned, not raised, so a
        single slow/failed route doesn't kill the batch.
        """
        calls = list(calls)
        results: list[tuple[Params, Any | StrideAPIError]] = []
        if not calls:
            return results

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_input = {
                pool.submit(self.get, ep, params): params for ep, params in calls
            }
            for fut in as_completed(future_to_input):
                params = future_to_input[fut]
                try:
                    results.append((params, fut.result()))
                except StrideAPIError as e:
                    results.append((params, e))
                except Exception as e:  # noqa: BLE001 - surface unexpected to caller
                    results.append((params, StrideAPIError(str(e))))
        return results

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> HealthStatus:
        """Cheap liveness probe — 1-row routes list."""
        t0 = time.perf_counter()
        try:
            self.get("gtfs_routes/list", {"limit": 1})
        except StrideAPIError as e:
            return HealthStatus(ok=False, latency_ms=None, detail=str(e))
        latency_ms = (time.perf_counter() - t0) * 1000
        return HealthStatus(ok=True, latency_ms=latency_ms, detail="OK")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_params(params: Params | None) -> dict[str, Any]:
        """Drop None values so they don't become literal 'None' in the URL."""
        if not params:
            return {}
        return {k: v for k, v in params.items() if v is not None}
