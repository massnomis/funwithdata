"""
Route Explorer — search by line number or operator, draw selected routes on a map.

Keeps filter state in `st.session_state` so re-runs don't lose user input.
The expensive bit (fetching route shapes) happens only when the user clicks "Draw".
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ui_components import (
    STOP_TOOLTIP,
    SidebarState,
    build_paths_layer,
    build_stops_layer,
    fetch_many_route_geometries,
    get_master_routes,
    render_deck,
)
from data_processor import filter_routes


def render(state: SidebarState) -> None:
    st.subheader("🗺️ Route Explorer")
    st.caption("Search planned routes by line number or by operator.")

    routes = get_master_routes()
    if routes.empty:
        st.warning(
            "No route data cached yet. Open the **💾 Data Manager** and "
            "refresh the routes cache."
        )
        return

    search_mode = st.radio(
        "Search by", ["Line number", "Operator"], horizontal=True, key="re_search_mode",
    )

    if search_mode == "Line number":
        filtered = _filter_by_line(routes)
    else:
        filtered = _filter_by_operator(routes)

    if filtered.empty:
        return

    st.success(f"Found {len(filtered):,} route variants.", icon="✅")
    _render_table(filtered, search_mode)
    _render_map_section(filtered, state)


# ---------------------------------------------------------------------------

def _filter_by_line(routes: pd.DataFrame) -> pd.DataFrame:
    query = st.text_input(
        "Line number or text",
        placeholder="e.g. 480",
        key="re_line_query",
    ).strip()
    if not query:
        return pd.DataFrame()
    return filter_routes(routes, line_query=query)


def _filter_by_operator(routes: pd.DataFrame) -> pd.DataFrame:
    operators = sorted(
        a for a in routes["agency_name"].dropna().unique() if isinstance(a, str)
    )
    if not operators:
        st.info("No operators found in the cached routes.")
        return pd.DataFrame()
    selected = st.selectbox("Operator", operators, key="re_operator")
    return filter_routes(routes, agency=selected)


def _render_table(filtered: pd.DataFrame, search_mode: str) -> None:
    """Show a table of matching routes (collapsed by default for line searches)."""
    cols = [c for c in ("route_short_name", "agency_name", "route_long_name", "id")
            if c in filtered.columns]
    expanded = search_mode == "Operator"
    with st.expander("📋 Matching routes", expanded=expanded):
        st.dataframe(
            filtered[cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "route_short_name": st.column_config.TextColumn("Line", width="small"),
                "agency_name": st.column_config.TextColumn("Operator", width="medium"),
                "route_long_name": st.column_config.TextColumn("Description", width="large"),
                "id": st.column_config.NumberColumn("Route ID", width="small"),
            },
        )


def _render_map_section(filtered: pd.DataFrame, state: SidebarState) -> None:
    display = (
        "Line "
        + filtered["route_short_name"].astype(str)
        + " · "
        + filtered["route_long_name"].astype(str).str.slice(0, 80)
        + " (ID "
        + filtered["id"].astype(str)
        + ")"
    )
    options = display.tolist()
    id_lookup = dict(zip(options, filtered["id"].astype(int).tolist()))

    selected_labels = st.multiselect(
        "Select routes to draw on the map",
        options=options,
        max_selections=20,
        key="re_selected_labels",
    )

    if not st.button("🎨 Draw selected routes", type="primary", disabled=not selected_labels):
        return

    route_ids = [id_lookup[label] for label in selected_labels]
    geometries = fetch_many_route_geometries(route_ids)
    if not geometries:
        st.error("Couldn't fetch geometry for the selected routes.")
        return

    layers = [
        build_paths_layer(geometries, state.viz.path_width),
        build_stops_layer(geometries, state.viz.dot_radius),
    ]
    # Center on the first stop of the first route — keeps the map relevant.
    first_geom = geometries[0]
    lon, lat = first_geom.path[0]
    render_deck(
        layers,
        center_lat=lat,
        center_lon=lon,
        zoom=11,
        pitch=0,
        tooltip=STOP_TOOLTIP,
    )
    st.caption(f"Drew {len(geometries)} route variants.")
