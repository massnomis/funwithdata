"""
Live SIRI (vehicle position) explorer.

The SIRI panel is wrapped in `st.fragment` so:
    * auto-refresh re-runs only this block, not the entire sidebar/headers
    * changing the line-filter doesn't refetch the API (data is cached for ~20s anyway)
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from data_processor import (
    bearing_to_compass,
    bearing_to_emoji,
    bearing_to_icon_descriptor,
    downcast_for_display,
)
from ui_components import (
    SIRI_TOOLTIP,
    SidebarState,
    build_paths_layer,
    build_siri_icon_layer,
    build_user_pin_layer,
    fetch_many_route_geometries,
    get_enriched_siri,
    now_local_str,
    render_deck,
    render_fleet_table,
    render_metric_row,
)


def render(state: SidebarState) -> None:
    """Entry point: header + the live fragment."""
    st.subheader(
        f"📡 Real-time activity · {state.city} "
        f"(radius {state.radius_km:g} km, last {state.lookback_min} min)"
    )
    # Manual refresh (works even when auto-refresh is off).
    if st.button("🔄 Refresh now", type="primary"):
        st.cache_data.clear()

    if state.auto_refresh:
        _live_panel_autorefresh(state)
    else:
        _live_panel(state)


# `run_every` makes Streamlit re-execute just this fragment on a timer.
@st.fragment(run_every="20s")
def _live_panel_autorefresh(state: SidebarState) -> None:
    _live_panel(state)


@st.fragment
def _live_panel(state: SidebarState) -> None:
    """The data-driven section. Anything below this point is fragment-local."""
    with st.spinner("Fetching live vehicle positions..."):
        try:
            siri = get_enriched_siri(
                state.lat, state.lon, state.radius_km, state.lookback_min
            )
        except Exception as e:  # noqa: BLE001 - surface as a toast, never crash
            st.toast(f"SIRI fetch failed: {e}", icon="❌")
            return

    if siri.empty:
        st.warning(
            "No active buses in this window. Try widening radius or lookback, "
            "or pick a busier city."
        )
        return

    siri = _augment_for_display(siri)
    _render_metrics(siri)

    line_filter, show_paths = _render_filters(siri)
    filtered = _apply_line_filter(siri, line_filter)

    paths = _resolve_paths_to_draw(filtered, line_filter, show_paths, state.viz.max_paths)
    _render_map(filtered, paths, state)

    st.subheader("📊 Fleet monitor")
    render_fleet_table(downcast_for_display(filtered))


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def _augment_for_display(siri: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns used by the map and the table."""
    if "bearing" in siri.columns:
        siri = siri.assign(
            compass_direction=bearing_to_emoji(siri["bearing"]),
            icon_data=bearing_to_icon_descriptor(siri["bearing"]),
        )
    else:
        siri = siri.assign(compass_direction="❓", icon_data=None)
    return siri


def _render_metrics(siri: pd.DataFrame) -> None:
    n_active = len(siri)

    def _fmt(value: float | None, unit: str) -> str:
        return f"{value:.1f} {unit}" if value is not None and not pd.isna(value) else "—"

    velocity = siri["velocity"] if "velocity" in siri.columns else pd.Series(dtype=float)
    avg = velocity.mean() if not velocity.empty else None
    peak = velocity.max() if not velocity.empty else None

    render_metric_row([
        ("Active vehicles", str(n_active), None),
        ("Avg speed", _fmt(avg, "km/h"), None),
        ("Max speed", _fmt(peak, "km/h"), None),
        ("Last refresh", now_local_str(), "live"),
    ])


def _render_filters(siri: pd.DataFrame) -> tuple[list[str], bool]:
    st.markdown("#### 🎚️ Filters")
    lines_available = sorted(siri["route_short_name"].dropna().astype(str).unique())
    col_filter, col_paths = st.columns([3, 1])
    with col_filter:
        selected = st.multiselect(
            "Filter by line",
            options=lines_available,
            placeholder="All lines",
            key="live_line_filter",
        )
    with col_paths:
        show_paths = st.checkbox(
            "Show route paths", value=False, key="live_show_paths",
            help="Overlay scheduled route geometry. Slower.",
        )
    return selected, show_paths


def _apply_line_filter(siri: pd.DataFrame, selected: list[str]) -> pd.DataFrame:
    if not selected:
        return siri
    return siri[siri["route_short_name"].astype(str).isin(selected)].copy()


def _resolve_paths_to_draw(
    filtered: pd.DataFrame,
    selected_lines: list[str],
    show_paths: bool,
    cap: int,
) -> list:
    """Decide which route IDs to fetch geometry for and fetch them concurrently."""
    if not (selected_lines or show_paths):
        return []
    if "gtfs_route_id" not in filtered.columns:
        return []
    ids = (
        filtered["gtfs_route_id"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    if not ids:
        return []
    if len(ids) > cap:
        st.toast(
            f"Limiting paths to {cap} of {len(ids)} routes (raise the cap in sidebar).",
            icon="⚠️",
        )
        ids = ids[:cap]
    return fetch_many_route_geometries([int(x) for x in ids])


def _render_map(
    filtered: pd.DataFrame, geometries: list, state: SidebarState,
) -> None:
    layers = [
        build_paths_layer(geometries, state.viz.path_width),
        build_siri_icon_layer(filtered),
        build_user_pin_layer(state.lat, state.lon, state.viz.pin_radius),
    ]
    render_deck(
        layers,
        center_lat=state.lat,
        center_lon=state.lon,
        zoom=13,
        pitch=45,
        tooltip=SIRI_TOOLTIP,
    )
    st.caption(
        "🔴 your search location · 🚌 directional arrows = vehicle bearing · "
        "colored lines = scheduled route paths"
    )
