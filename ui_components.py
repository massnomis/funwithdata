"""
Reusable UI building blocks and the Streamlit caching boundary.

This module is the only place that knows how to:
    * instantiate the Stride client (memoized as a resource)
    * cache reference data on disk + in memory
    * cache short-lived live data with a TTL
    * render shared widgets (sidebar, metric cards, fleet table, maps)

Modes import these helpers instead of touching `streamlit.cache_*` directly,
so caching policy stays in one place.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import pydeck as pdk
import streamlit as st

from api_client import HealthStatus, StrideAPIError, StrideClient, StrideValidationError
from config import (
    CACHE_TTL_LIVE_SEC,
    CACHE_TTL_REFERENCE_SEC,
    CACHE_TTL_SHAPE_SEC,
    CITY_PRESETS,
    DEFAULT_CITY,
    DISK_CACHE_DIR,
    VizSettings,
)
from data_processor import (
    BBox,
    RouteGeometry,
    build_master_routes,
    build_route_geometry,
    enrich_siri_with_routes,
    haversine_bbox,
    sanitize_siri,
)


# ===========================================================================
#  Cached resources & data
# ===========================================================================

@st.cache_resource(show_spinner=False)
def get_client() -> StrideClient:
    """One pooled HTTP client per Streamlit session."""
    return StrideClient()


@st.cache_data(ttl=60, show_spinner=False)
def get_api_health() -> HealthStatus:
    """Liveness probe — short TTL so we re-check after outages."""
    return get_client().health()


# ---------------------------------------------------------------------------
# Reference data (routes / stops / agencies): long-lived, disk-backed
# ---------------------------------------------------------------------------

def _disk_cache_path(filename: str) -> str:
    """`data/YYYY-MM-DD/<filename>` — partitioned by UTC date."""
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    folder = os.path.join(DISK_CACHE_DIR, day)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, filename)


def _load_from_disk(filename: str) -> list[dict] | None:
    path = _disk_cache_path(filename)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_to_disk(filename: str, data: list[dict]) -> str:
    path = _disk_cache_path(filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return path


@st.cache_data(ttl=CACHE_TTL_REFERENCE_SEC, show_spinner=False)
def get_routes_today() -> list[dict]:
    """Today's routes. Reads disk first, falls back to API + writes disk."""
    cached = _load_from_disk("routes.json")
    if cached is not None:
        return cached
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    data = get_client().list_all(
        "gtfs_routes/list",
        {"date_from": today, "date_to": today},
    )
    _save_to_disk("routes.json", data)
    return data


@st.cache_data(ttl=CACHE_TTL_REFERENCE_SEC, show_spinner=False)
def get_stops_today() -> list[dict]:
    cached = _load_from_disk("stops.json")
    if cached is not None:
        return cached
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    data = get_client().list_all(
        "gtfs_stops/list",
        {"date_from": today, "date_to": today},
    )
    _save_to_disk("stops.json", data)
    return data


@st.cache_data(ttl=CACHE_TTL_REFERENCE_SEC, show_spinner=False)
def get_agencies() -> list[dict]:
    cached = _load_from_disk("agencies.json")
    if cached is not None:
        return cached
    data = get_client().list_all("gtfs_agencies/list", {})
    _save_to_disk("agencies.json", data)
    return data


@st.cache_data(ttl=CACHE_TTL_REFERENCE_SEC, show_spinner=False)
def get_master_routes() -> pd.DataFrame:
    """Routes joined with agency names — the lookup table modes consume."""
    return build_master_routes(get_routes_today(), get_agencies())


def invalidate_reference_caches() -> None:
    """Drop in-memory caches AND disk artifacts. Called by the Data Manager."""
    get_routes_today.clear()
    get_stops_today.clear()
    get_agencies.clear()
    get_master_routes.clear()
    for name in ("routes.json", "stops.json", "agencies.json"):
        path = _disk_cache_path(name)
        if os.path.exists(path):
            os.remove(path)


# ---------------------------------------------------------------------------
# Live data (SIRI): short TTL, no disk
# ---------------------------------------------------------------------------

# Stride's `siri_vehicle_locations/list` rejects limit=-1 with HTTP 500
# ("due to abuse, maximum limit per request is 15000 items"). We send an
# explicit cap. A 10-minute window in a city-scale radius rarely exceeds this.
SIRI_HARD_LIMIT = 5000


@st.cache_data(ttl=CACHE_TTL_LIVE_SEC, show_spinner=False)
def fetch_live_siri(
    lat: float, lon: float, radius_km: float, lookback_min: int
) -> pd.DataFrame:
    """SIRI vehicle locations within radius, parsed and sanitized."""
    bbox = haversine_bbox(lat, lon, radius_km)
    # Stride wants ISO-8601 with offset; let `pd.Timestamp` handle the formatting.
    now = pd.Timestamp.now(tz="Asia/Jerusalem")
    from_time = now - pd.Timedelta(minutes=lookback_min)
    params: dict[str, Any] = {
        "recorded_at_time_from": from_time.isoformat(),
        "order_by": "recorded_at_time desc",
        "limit": SIRI_HARD_LIMIT,
        **bbox.as_api_params(),
    }
    # Single-shot; pagination would just repeat the same recent vehicles.
    records = get_client().get("siri_vehicle_locations/list", params)
    if not isinstance(records, list):
        records = []
    return sanitize_siri(records)


# ---------------------------------------------------------------------------
# Route shapes: mid-life TTL — they change rarely within a day
# ---------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL_SHAPE_SEC, show_spinner=False)
def fetch_route_geometry(route_id: int) -> RouteGeometry | None:
    """Path + stops for one route's first ride today."""
    client = get_client()
    today = datetime.now(tz=None).date().isoformat()

    rides = client.list_all("gtfs_rides/list", {
        "gtfs_route__date_from": today,
        "gtfs_route__date_to": today,
        "gtfs_route_id": route_id,
        "order_by": "start_time asc",
    })
    if not rides:
        return None
    ride_id = rides[0]["id"]

    ride_stops = client.list_all("gtfs_ride_stops/list", {
        "gtfs_ride_ids": str(ride_id),
        "order_by": "stop_sequence",
    })
    return build_route_geometry(ride_id, ride_stops)


def fetch_many_route_geometries(route_ids: list[int]) -> list[RouteGeometry]:
    """Concurrent fan-out for "show all selected routes on the map".

    Uses the cached single-route fetcher per id so partial cache hits remain cheap.
    """
    geoms: list[RouteGeometry] = []
    if not route_ids:
        return geoms
    progress = st.progress(0.0, text=f"Loading {len(route_ids)} route shapes...")
    for i, rid in enumerate(route_ids, start=1):
        try:
            geom = fetch_route_geometry(int(rid))
        except (ValueError, TypeError, StrideAPIError):
            geom = None
        if geom is not None and geom.is_drawable:
            geoms.append(geom)
        progress.progress(i / len(route_ids), text=f"Loading route shapes ({i}/{len(route_ids)})")
    progress.empty()
    return geoms


# ===========================================================================
#  Shared UI widgets
# ===========================================================================

@dataclass
class SidebarState:
    """Snapshot of sidebar selections returned to the mode renderer."""
    mode: str
    viz: VizSettings
    city: str
    lat: float
    lon: float
    radius_km: float
    lookback_min: int
    auto_refresh: bool


APP_MODES: tuple[str, ...] = (
    "📡 Live Traffic",
    "🗺️ Route Explorer",
    "🌐 Network Graph",
    "💾 Data Manager",
    "🔍 API Explorer",
)


def render_sidebar() -> SidebarState:
    """Build the sidebar and return user selections as a typed snapshot."""
    with st.sidebar:
        st.title("🚌 Stride Explorer")
        st.caption("Israeli public transit, live.")

        mode = st.radio("Mode", APP_MODES, key="app_mode")
        st.divider()

        # --- Viz settings, collapsed by default ---
        with st.expander("🎨 Visualization", expanded=False):
            viz = VizSettings(
                dot_radius=st.slider("Stop radius (m)", 10, 500, 50, key="viz_dot"),
                path_width=st.slider("Path width (m)", 5, 200, 30, key="viz_path"),
                arrow_size=st.slider("Arrow size", 10, 100, 45, key="viz_arrow"),
                pin_radius=st.slider("Pin radius (m)", 50, 1000, 150, key="viz_pin"),
                max_paths=st.slider("Max paths drawn", 10, 500, 50, key="viz_max",
                                    help="Cap on live route paths drawn simultaneously."),
            )

        # --- Location controls — only modes that need a center point ---
        city = DEFAULT_CITY
        lat, lon = CITY_PRESETS[DEFAULT_CITY].lat, CITY_PRESETS[DEFAULT_CITY].lon
        radius_km = 2.0
        lookback_min = 10
        auto_refresh = False

        if mode in ("📡 Live Traffic", "🗺️ Route Explorer"):
            st.subheader("📍 Location")
            city = st.selectbox(
                "City preset",
                list(CITY_PRESETS.keys()) + ["Custom"],
                key="city_preset",
            )
            if city == "Custom":
                lat = st.number_input("Latitude", value=32.0853, format="%.4f", key="custom_lat")
                lon = st.number_input("Longitude", value=34.7818, format="%.4f", key="custom_lon")
            else:
                preset = CITY_PRESETS[city]
                lat, lon = preset.lat, preset.lon

        if mode == "📡 Live Traffic":
            radius_km = st.slider("Scan radius (km)", 0.5, 10.0, 2.0, 0.5, key="live_radius")
            lookback_min = st.slider("Lookback (min)", 1, 30, 10, key="live_lookback")
            auto_refresh = st.toggle("Auto-refresh", value=False, key="live_auto",
                                     help="Re-fetch SIRI every 20s.")

        st.divider()
        _render_health_badge()
        st.caption("Data: Open Bus Stride API · Hasadna")

        return SidebarState(
            mode=mode, viz=viz, city=city, lat=lat, lon=lon,
            radius_km=radius_km, lookback_min=lookback_min,
            auto_refresh=auto_refresh,
        )


def _render_health_badge() -> None:
    """Compact API status indicator at the bottom of the sidebar."""
    health = get_api_health()
    if health.ok and health.latency_ms is not None:
        st.success(f"API online · {health.latency_ms:.0f} ms", icon="🟢")
    else:
        st.error(f"API offline: {health.detail}", icon="🔴")


# ---------------------------------------------------------------------------
# Result-status helpers — replaces ad-hoc print/expander/st.error blocks
# ---------------------------------------------------------------------------

def report_api_error(err: StrideAPIError, *, where: str) -> None:
    """Surface an API failure consistently. Caller decides whether to stop()."""
    if isinstance(err, StrideValidationError):
        st.toast(f"Invalid request to {where}", icon="⚠️")
        with st.expander(f"Validation error · {where}", expanded=False):
            st.json(err.details if err.details is not None else str(err))
    else:
        st.toast(f"{where} failed: {err}", icon="❌")


# ---------------------------------------------------------------------------
# Map rendering — pydeck layers + Deck factory
# ---------------------------------------------------------------------------

def build_siri_icon_layer(siri: pd.DataFrame) -> pdk.Layer | None:
    """Vehicle locations as directional Twemoji arrows."""
    if siri.empty or "icon_data" not in siri.columns:
        return None
    # Drop rows without an icon descriptor (e.g. missing bearing).
    drawable = siri[siri["icon_data"].notna()].copy()
    if drawable.empty:
        return None
    return pdk.Layer(
        "IconLayer",
        data=drawable.to_dict("records"),
        get_icon="icon_data",
        get_size=4,
        size_scale=8,
        get_position="[lon, lat]",
        pickable=True,
    )


def build_paths_layer(
    geometries: list[RouteGeometry], path_width: int
) -> pdk.Layer | None:
    if not geometries:
        return None
    from data_processor import palette_color
    data = [
        {"path": g.path, "color": list(palette_color(i))}
        for i, g in enumerate(geometries)
        if g.is_drawable
    ]
    if not data:
        return None
    return pdk.Layer(
        "PathLayer",
        data=data,
        get_path="path",
        get_width=path_width,
        get_color="color",
        width_min_pixels=2,
        pickable=False,
    )


def build_user_pin_layer(lat: float, lon: float, radius: int) -> pdk.Layer:
    return pdk.Layer(
        "ScatterplotLayer",
        data=[{"lat": lat, "lon": lon}],
        get_position="[lon, lat]",
        get_radius=radius,
        get_fill_color=[255, 0, 0, 180],
        stroked=True,
        get_line_color=[255, 255, 255],
        get_line_width=2,
    )


def build_stops_layer(
    geometries: list[RouteGeometry], dot_radius: int
) -> pdk.Layer | None:
    if not geometries:
        return None
    from data_processor import palette_color
    rows: list[dict] = []
    for i, g in enumerate(geometries):
        color = list(palette_color(i))
        for stop in g.stops:
            rows.append({**stop, "color": color})
    if not rows:
        return None
    return pdk.Layer(
        "ScatterplotLayer",
        data=rows,
        get_position="coordinates",
        get_color="color",
        get_radius=dot_radius * 2,
        pickable=True,
    )


SIRI_TOOLTIP = {
    "html": (
        "<div style='font-family: sans-serif; padding: 4px 8px;'>"
        "<b>Line:</b> {route_short_name}<br/>"
        "<b>Destination:</b> {route_long_name}<br/>"
        "<b>Speed:</b> {velocity} km/h<br/>"
        "<b>Bearing:</b> {bearing}°"
        "</div>"
    ),
    "style": {"backgroundColor": "rgba(15, 23, 42, 0.92)", "color": "white"},
}

STOP_TOOLTIP = {
    "html": (
        "<div style='font-family: sans-serif; padding: 4px 8px;'>"
        "<b>Stop:</b> {name}<br/>"
        "<b>Code:</b> {code}<br/>"
        "<b>Sequence:</b> {seq}"
        "</div>"
    ),
    "style": {"backgroundColor": "rgba(15, 23, 42, 0.92)", "color": "white"},
}


def render_deck(
    layers: list[pdk.Layer],
    *,
    center_lat: float,
    center_lon: float,
    zoom: int = 13,
    pitch: int = 45,
    tooltip: dict | None = None,
) -> None:
    """Render a pydeck chart. None layers are silently dropped."""
    actual = [layer for layer in layers if layer is not None]
    if not actual:
        st.info("Nothing to draw yet.", icon="🗺️")
        return
    st.pydeck_chart(
        pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(
                latitude=center_lat, longitude=center_lon,
                zoom=zoom, pitch=pitch,
            ),
            layers=actual,
            tooltip=tooltip or {},
        ),
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Fleet table
# ---------------------------------------------------------------------------

FLEET_PRIORITY_COLS = (
    "route_short_name", "agency_name", "velocity", "compass_direction",
    "recorded_at_time", "route_long_name", "lon", "lat", "line_ref",
    "operator_ref", "bearing", "distance_from_journey_start", "siri_ride_stop_id",
)


def render_fleet_table(df: pd.DataFrame) -> None:
    """Live fleet table with native Streamlit column configs."""
    if df.empty:
        st.info("No vehicles to show.")
        return
    visible_cols = [c for c in FLEET_PRIORITY_COLS if c in df.columns]
    remaining = sorted(c for c in df.columns if c not in visible_cols)
    final = visible_cols + remaining

    st.dataframe(
        df[final],
        column_config={
            "route_short_name": st.column_config.TextColumn("Line", width="small"),
            "agency_name": st.column_config.TextColumn("Operator", width="medium"),
            "velocity": st.column_config.ProgressColumn(
                "Speed (km/h)", format="%d", min_value=0, max_value=120
            ),
            "compass_direction": st.column_config.TextColumn("Heading", width="small"),
            "recorded_at_time": st.column_config.DatetimeColumn("Last signal", format="HH:mm:ss"),
            "route_long_name": st.column_config.TextColumn("Description", width="large"),
            "lat": st.column_config.NumberColumn("Lat", format="%.5f"),
            "lon": st.column_config.NumberColumn("Lon", format="%.5f"),
        },
        hide_index=True,
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Metric cards — wrapped so colors/icons stay consistent
# ---------------------------------------------------------------------------

def render_metric_row(metrics: list[tuple[str, str, str | None]]) -> None:
    """Render N metrics in equal-width columns.

    Each tuple is `(label, value, delta_or_None)`.
    """
    if not metrics:
        return
    cols = st.columns(len(metrics))
    for col, (label, value, delta) in zip(cols, metrics):
        with col:
            st.metric(label, value, delta=delta)


# ---------------------------------------------------------------------------
# Live-traffic enrichment helper used by the live mode (kept here so the
# Streamlit cache lives next to its cousins)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL_LIVE_SEC, show_spinner=False)
def get_enriched_siri(
    lat: float, lon: float, radius_km: float, lookback_min: int
) -> pd.DataFrame:
    """Fetch + sanitize + enrich SIRI in one cached call.

    Caching key includes location/radius/lookback so distinct queries
    don't collide. Short TTL ensures freshness.
    """
    raw = fetch_live_siri(lat, lon, radius_km, lookback_min)
    if raw.empty:
        return raw
    enriched = enrich_siri_with_routes(raw, get_master_routes())
    return enriched


def now_local_str(fmt: str = "%H:%M:%S") -> str:
    """Wall-clock time in Asia/Jerusalem, formatted for display."""
    return pd.Timestamp.now(tz="Asia/Jerusalem").strftime(fmt)
