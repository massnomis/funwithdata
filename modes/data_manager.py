"""
Data Manager — explicit control over the disk + memory cache layer.

Surfaces success/failure as toasts and gives a per-file status grid.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Callable

import pandas as pd
import streamlit as st

from api_client import StrideAPIError
from config import DISK_CACHE_DIR
from ui_components import (
    SidebarState,
    get_agencies,
    get_routes_today,
    get_stops_today,
    invalidate_reference_caches,
)


_CACHE_FILES = ("routes.json", "stops.json", "agencies.json")


def render(_: SidebarState) -> None:
    st.subheader("💾 Bulk Data Manager")
    st.caption(
        "Reference data (routes, stops, agencies) is cached to disk per UTC day "
        "and held in memory for the session. Refresh manually after upstream changes."
    )

    _render_actions()
    st.divider()
    _render_cache_status()


# ---------------------------------------------------------------------------

def _render_actions() -> None:
    cols = st.columns(4)
    if cols[0].button("📥 Refresh routes", use_container_width=True):
        _refresh("routes", get_routes_today)
    if cols[1].button("📥 Refresh stops", use_container_width=True):
        _refresh("stops", get_stops_today)
    if cols[2].button("📥 Refresh agencies", use_container_width=True):
        _refresh("agencies", get_agencies)
    if cols[3].button("🗑️ Clear all caches", type="secondary", use_container_width=True):
        invalidate_reference_caches()
        st.toast("All caches cleared.", icon="🧹")


def _refresh(label: str, loader: Callable[[], list]) -> None:
    """Single-cache refresh: drop in-memory + disk, re-fetch, time it."""
    loader.clear()  # type: ignore[attr-defined]
    path = os.path.join(
        DISK_CACHE_DIR,
        datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        f"{label}.json",
    )
    if os.path.exists(path):
        os.remove(path)

    with st.status(f"Refreshing {label}...", expanded=False) as status:
        t0 = time.perf_counter()
        try:
            data = loader()
        except StrideAPIError as e:
            status.update(label=f"Failed: {e}", state="error", expanded=True)
            return
        elapsed = time.perf_counter() - t0
        status.update(
            label=f"{label}: {len(data):,} records in {elapsed:.1f}s",
            state="complete",
        )
        st.toast(f"{label.capitalize()} cache refreshed.", icon="✅")


def _render_cache_status() -> None:
    st.subheader("📦 Cache state")
    folder = os.path.join(DISK_CACHE_DIR, datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    rows: list[dict] = []
    for name in _CACHE_FILES:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            rows.append({
                "file": name,
                "status": "✅ cached",
                "size (MB)": round(size_mb, 2),
                "modified": mtime.strftime("%Y-%m-%d %H:%M:%S"),
            })
        else:
            rows.append({"file": name, "status": "❌ missing", "size (MB)": None, "modified": None})

    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
