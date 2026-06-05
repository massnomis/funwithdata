"""
Data transformation layer.

Pure-ish functions for cleaning, enriching, and reshaping Stride API responses
into analysis-ready pandas frames. No Streamlit imports — call sites do their
own caching with `@st.cache_data`.

Performance: vectorized pandas everywhere. The legacy implementation used
`DataFrame.apply(..., axis=1)` for SIRI enrichment and time parsing; both are
O(n) Python loops that dominated runtime on >5k-row payloads. We replace them
with merges and `pd.to_datetime`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, NamedTuple

import numpy as np
import pandas as pd

from config import (
    BEARING_ICONS,
    ISRAEL_LAT_RANGE,
    ISRAEL_LON_RANGE,
    ROUTE_PALETTE,
)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

class BBox(NamedTuple):
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def as_api_params(self) -> dict[str, float]:
        """Stride uses suffix operators for ranges on lat/lon."""
        return {
            "lat__greater_or_equal": self.min_lat,
            "lat__lower_or_equal":   self.max_lat,
            "lon__greater_or_equal": self.min_lon,
            "lon__lower_or_equal":   self.max_lon,
        }


def haversine_bbox(lat: float, lon: float, radius_km: float) -> BBox:
    """Approximate lat/lon envelope around a point. Good enough for filtering."""
    lat_delta = radius_km / 111.0
    # 111 km/deg longitude at the equator, scaled by cos(latitude).
    cos_lat = max(math.cos(math.radians(lat)), 1e-6)
    lon_delta = abs(radius_km / (111.0 * cos_lat))
    return BBox(
        min_lat=lat - lat_delta,
        max_lat=lat + lat_delta,
        min_lon=lon - lon_delta,
        max_lon=lon + lon_delta,
    )


def palette_color(index: int) -> tuple[int, int, int]:
    """Pick a distinct RGB color for the Nth route on a map."""
    return ROUTE_PALETTE[index % len(ROUTE_PALETTE)]


# ---------------------------------------------------------------------------
# Bearings — vectorized
# ---------------------------------------------------------------------------

# Boundaries in degrees for 8-point compass. Order matches `_COMPASS_LABELS`.
_BEARING_BINS = np.array([0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360.001])
_COMPASS_LABELS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
_COMPASS_EMOJI = {
    "N":  "⬆️ N", "NE": "↗️ NE", "E":  "➡️ E",  "SE": "↘️ SE",
    "S":  "⬇️ S", "SW": "↙️ SW", "W":  "⬅️ W",  "NW": "↖️ NW",
}


def bearing_to_compass(bearings: pd.Series) -> pd.Series:
    """Vectorized: bearing degrees → 'N'/'NE'/... Returns NaN for missing input."""
    cleaned = pd.to_numeric(bearings, errors="coerce") % 360
    cats = pd.cut(cleaned, bins=_BEARING_BINS, labels=_COMPASS_LABELS, ordered=False)
    # pd.cut with duplicate labels yields a Categorical; convert to plain strings.
    return cats.astype("object").where(cleaned.notna(), other=None)


def bearing_to_emoji(bearings: pd.Series) -> pd.Series:
    """Vectorized: bearing degrees → human-readable arrow+letter."""
    compass = bearing_to_compass(bearings)
    return compass.map(_COMPASS_EMOJI).fillna("❓")


def bearing_to_icon_descriptor(bearings: pd.Series) -> pd.Series:
    """
    Build a deck.gl IconLayer descriptor for each row.

    deck.gl wants `{"url", "width", "height", "anchorY"}`. We emit one dict per
    bearing or None where the bearing is missing.
    """
    compass = bearing_to_compass(bearings)

    def _descriptor(direction: str | None) -> dict | None:
        if direction is None or direction not in BEARING_ICONS:
            return None
        return {
            "url": BEARING_ICONS[direction],
            "width": 28,
            "height": 28,
            "anchorY": 28,
        }

    return compass.map(_descriptor)


# ---------------------------------------------------------------------------
# SIRI (live vehicle locations)
# ---------------------------------------------------------------------------

SIRI_NUMERIC_COLS = ("lat", "lon", "bearing", "velocity")


def sanitize_siri(records: Iterable[dict] | pd.DataFrame) -> pd.DataFrame:
    """Normalize raw SIRI records into a tidy DataFrame.

    * Coerces numerics
    * Drops rows missing lat/lon
    * Drops rows outside the Israel envelope (garbage GPS pings)
    * De-duplicates on (vehicle, timestamp), keeping the latest record
    """
    df = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    if df.empty:
        return df

    for col in SIRI_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["lat", "lon"])
    lat_lo, lat_hi = ISRAEL_LAT_RANGE
    lon_lo, lon_hi = ISRAEL_LON_RANGE
    df = df[
        df["lat"].between(lat_lo, lat_hi) &
        df["lon"].between(lon_lo, lon_hi)
    ]

    if "recorded_at_time" in df.columns:
        df["recorded_at_time"] = pd.to_datetime(df["recorded_at_time"], errors="coerce", utc=True)

    dedupe_cols = [c for c in ("siri_ride__vehicle_ref", "recorded_at_time") if c in df.columns]
    if dedupe_cols and "recorded_at_time" in df.columns:
        df = df.sort_values("recorded_at_time").drop_duplicates(subset=dedupe_cols, keep="last")
    elif dedupe_cols:
        df = df.drop_duplicates(subset=dedupe_cols, keep="last")

    return df.reset_index(drop=True)


def enrich_siri_with_routes(siri: pd.DataFrame, master_routes: pd.DataFrame) -> pd.DataFrame:
    """Attach `route_short_name`, `route_long_name`, `agency_name`, `gtfs_route_id` to SIRI rows.

    Two merges replace a per-row Python lookup:
        1. By `gtfs_ride__gtfs_route_id` → exact route match (preferred)
        2. By `line_ref` → fallback when the SIRI feed lacks the gtfs FK

    The fallback uses the most-recent route variant per `line_ref`.
    """
    if siri.empty or master_routes.empty:
        return siri.assign(
            route_short_name="Unknown",
            route_long_name="Unknown",
            agency_name="Unknown",
            gtfs_route_id=pd.NA,
        )

    cols = ["id", "line_ref", "route_short_name", "route_long_name", "agency_name"]
    available = [c for c in cols if c in master_routes.columns]
    routes = master_routes[available].copy()
    if "line_ref" in routes.columns:
        routes["line_ref"] = routes["line_ref"].astype(str)

    # Pick only the lookup columns the master frame actually carries.
    lookup_targets = [c for c in ("route_short_name", "route_long_name", "agency_name")
                      if c in routes.columns]

    # --- Pass 1: exact gtfs_route_id match ---
    if "gtfs_ride__gtfs_route_id" in siri.columns and "id" in routes.columns and lookup_targets:
        primary = routes.rename(columns={"id": "gtfs_route_id"}).drop_duplicates("gtfs_route_id")
        merged = siri.merge(
            primary[["gtfs_route_id", *lookup_targets]],
            left_on="gtfs_ride__gtfs_route_id",
            right_on="gtfs_route_id",
            how="left",
            suffixes=("", "_p1"),
        )
    else:
        merged = siri.copy()
        if "gtfs_route_id" not in merged.columns:
            merged["gtfs_route_id"] = pd.NA
        for col in ("route_short_name", "route_long_name", "agency_name"):
            if col not in merged.columns:
                merged[col] = pd.NA

    # --- Pass 2: line_ref fallback for rows that didn't match ---
    if "line_ref" in routes.columns and lookup_targets and "id" in routes.columns:
        rename_map = {"id": "gtfs_route_id_fb"} | {c: f"{c}_fb" for c in lookup_targets}
        fallback = (
            routes.sort_values("id", ascending=False)
                  .drop_duplicates("line_ref")
                  .rename(columns=rename_map)
        )

        # Coalesce siri_route__line_ref → line_ref (mirrors the legacy `or` logic).
        siri_ref = merged["siri_route__line_ref"] if "siri_route__line_ref" in merged.columns else None
        gtfs_ref = merged["line_ref"] if "line_ref" in merged.columns else None
        if siri_ref is not None and gtfs_ref is not None:
            siri_line_ref = siri_ref.astype("string").fillna(gtfs_ref.astype("string"))
        elif siri_ref is not None:
            siri_line_ref = siri_ref.astype("string")
        elif gtfs_ref is not None:
            siri_line_ref = gtfs_ref.astype("string")
        else:
            siri_line_ref = pd.Series([pd.NA] * len(merged), index=merged.index, dtype="string")
        merged["_line_ref_key"] = siri_line_ref

        fb_cols = ["line_ref", "gtfs_route_id_fb", *[f"{c}_fb" for c in lookup_targets]]
        fb_cols = [c for c in fb_cols if c in fallback.columns]
        merged = merged.merge(
            fallback[fb_cols],
            left_on="_line_ref_key",
            right_on="line_ref",
            how="left",
            suffixes=("", "_fb_drop"),
        )

        for col in ("route_short_name", "route_long_name", "agency_name", "gtfs_route_id"):
            fb_col = f"{col}_fb"
            if fb_col in merged.columns and col in merged.columns:
                merged[col] = merged[col].fillna(merged[fb_col])

        drop_cols = [c for c in merged.columns if c.endswith(("_fb", "_fb_drop"))]
        merged = merged.drop(columns=drop_cols + ["_line_ref_key"], errors="ignore")
        # Avoid duplicate line_ref columns if both sides had one.
        if "line_ref_x" in merged.columns and "line_ref_y" in merged.columns:
            merged["line_ref"] = merged["line_ref_x"].fillna(merged["line_ref_y"])
            merged = merged.drop(columns=["line_ref_x", "line_ref_y"], errors="ignore")

    for col in ("route_short_name", "route_long_name", "agency_name"):
        if col in merged.columns:
            merged[col] = merged[col].fillna("Unknown")

    return merged


def downcast_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Shrink memory footprint before passing into st.dataframe."""
    if df.empty:
        return df
    out = df.copy()
    for col in out.select_dtypes(include=["int64"]).columns:
        out[col] = pd.to_numeric(out[col], downcast="integer")
    for col in out.select_dtypes(include=["float64"]).columns:
        out[col] = pd.to_numeric(out[col], downcast="float")
    for col in out.select_dtypes(include=["object"]).columns:
        # Categorical only when the column has meaningful repetition.
        # Skip columns with unhashable values (e.g. dicts, lists).
        try:
            nunique = out[col].nunique(dropna=True)
        except TypeError:
            continue
        if nunique and nunique / max(len(out), 1) < 0.5:
            out[col] = out[col].astype("category")
    return out


# ---------------------------------------------------------------------------
# Routes (reference data)
# ---------------------------------------------------------------------------

def build_master_routes(routes: list[dict], agencies: list[dict]) -> pd.DataFrame:
    """Routes lookup table with `agency_name` guaranteed present.

    The `gtfs_routes/list` payload already carries `agency_name`. We only fall
    back to the `gtfs_agencies/list` lookup for rows where it's missing.
    """
    if not routes:
        return pd.DataFrame()

    df = pd.DataFrame(routes)

    # Ensure agency_name exists; fill gaps from the agencies table if available.
    if "agency_name" not in df.columns:
        df["agency_name"] = pd.NA

    needs_fill = df["agency_name"].isna()
    if needs_fill.any() and agencies:
        agency_df = pd.DataFrame(agencies)
        if {"operator_ref", "agency_name"}.issubset(agency_df.columns):
            # Keep the most recent record per operator (sort by date if present).
            sort_col = "date" if "date" in agency_df.columns else "operator_ref"
            lookup = (
                agency_df[["operator_ref", "agency_name", *( [sort_col] if sort_col != "operator_ref" else [] )]]
                .dropna(subset=["operator_ref", "agency_name"])
                .sort_values(sort_col)
                .drop_duplicates(subset="operator_ref", keep="last")
                .set_index("operator_ref")["agency_name"]
            )
            if "operator_ref" in df.columns:
                fill = df["operator_ref"].map(lookup)
                df["agency_name"] = df["agency_name"].fillna(fill)

    df["agency_name"] = df["agency_name"].fillna("Unknown")
    return df


def filter_routes(
    routes: pd.DataFrame,
    *,
    line_query: str | None = None,
    agency: str | None = None,
) -> pd.DataFrame:
    """Filter the master routes table by either line number/text OR operator."""
    if routes.empty:
        return routes

    out = routes
    if line_query:
        q = line_query.strip().lower()
        short = routes["route_short_name"].astype(str).str.lower()
        long_ = routes["route_long_name"].astype(str).str.lower()
        out = routes[short.eq(q) | long_.str.contains(q, na=False)]
    elif agency:
        out = routes[routes["agency_name"].eq(agency)]
    else:
        return routes.iloc[0:0]
    return out.copy()


# ---------------------------------------------------------------------------
# Timetables / ride geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RouteGeometry:
    """Geometric representation of one ride for a route."""
    ride_id: int
    path: list[list[float]]                # [[lon, lat], ...] in stop order
    stops: list[dict]                       # one dict per stop with name/code/seq/coords

    @property
    def is_drawable(self) -> bool:
        return len(self.path) >= 2


def build_route_geometry(
    ride_id: int,
    ride_stops: list[dict],
) -> RouteGeometry | None:
    """Build path + stops from `gtfs_ride_stops/list` output for one ride."""
    if not ride_stops:
        return None
    df = pd.DataFrame(ride_stops)
    df = df[df.get("gtfs_ride_id", ride_id) == ride_id]
    lat = pd.to_numeric(df.get("gtfs_stop__lat"), errors="coerce")
    lon = pd.to_numeric(df.get("gtfs_stop__lon"), errors="coerce")
    valid = lat.notna() & lon.notna()
    df = df[valid].copy()
    df["gtfs_stop__lat"] = lat[valid]
    df["gtfs_stop__lon"] = lon[valid]

    if df.empty:
        return None

    df = df.sort_values("stop_sequence")
    stops = [
        {
            "name": row.get("gtfs_stop__name", "Unknown"),
            "code": row.get("gtfs_stop__code"),
            "seq": int(row.get("stop_sequence", 0) or 0),
            "time": row.get("arrival_time") or "N/A",
            "coordinates": [float(row["gtfs_stop__lon"]), float(row["gtfs_stop__lat"])],
        }
        for _, row in df.iterrows()
    ]
    path = [s["coordinates"] for s in stops]
    return RouteGeometry(ride_id=ride_id, path=path, stops=stops)


# ---------------------------------------------------------------------------
# GTFS network (for network analysis mode)
# ---------------------------------------------------------------------------

class GTFSFrames(NamedTuple):
    """Cleaned and aligned GTFS-shaped frames ready for graph construction."""
    stops: pd.DataFrame
    routes: pd.DataFrame
    trips: pd.DataFrame
    stop_times: pd.DataFrame


def _parse_iso_time_to_seconds(series: pd.Series) -> pd.Series:
    """Vectorized: '2026-01-13T08:00:00+02:00' or '08:00:00' → seconds since midnight."""
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if parsed.notna().any():
        local = parsed.dt.tz_convert("Asia/Jerusalem")
        seconds = local.dt.hour * 3600 + local.dt.minute * 60 + local.dt.second
        return seconds.astype("Float64")

    # Fallback path: plain HH:MM:SS strings without date.
    s = series.astype("string").str.split(":", n=2, expand=True)
    if s.shape[1] < 3:
        return pd.Series([pd.NA] * len(series), index=series.index, dtype="Float64")
    h = pd.to_numeric(s[0], errors="coerce")
    m = pd.to_numeric(s[1], errors="coerce")
    sec = pd.to_numeric(s[2], errors="coerce")
    return (h * 3600 + m * 60 + sec).astype("Float64")


def _seconds_to_hhmmss(series: pd.Series) -> pd.Series:
    """Vectorized: seconds → 'HH:MM:SS'. NaN → '00:00:00'."""
    s = series.fillna(0).astype("int64")
    h = (s // 3600).astype(str).str.zfill(2)
    m = ((s % 3600) // 60).astype(str).str.zfill(2)
    sec = (s % 60).astype(str).str.zfill(2)
    return h + ":" + m + ":" + sec


def clean_gtfs_frames(
    stops_raw: list[dict],
    routes_raw: list[dict],
    rides_raw: list[dict],
    stop_times_raw: list[dict],
    bbox: BBox | None = None,
) -> GTFSFrames | None:
    """Normalize raw Stride GTFS payloads to standard column names.

    Returns None if any frame is empty after filtering — there's no graph to build.
    """
    stops = pd.DataFrame(stops_raw)
    routes = pd.DataFrame(routes_raw)
    rides = pd.DataFrame(rides_raw)
    stop_times = pd.DataFrame(stop_times_raw)

    if stops.empty or routes.empty or rides.empty or stop_times.empty:
        return None

    # --- Rename to GTFS-standard columns ---
    stops = stops.rename(columns={
        "id": "stop_id", "lat": "stop_lat", "lon": "stop_lon",
        "name": "stop_name", "code": "stop_code", "city": "stop_city",
    })
    routes = routes.rename(columns={
        "id": "route_id", "operator_ref": "agency_id",
    })
    rides = rides.rename(columns={
        "id": "trip_id", "gtfs_route_id": "route_id",
        "start_time": "trip_start_time", "end_time": "trip_end_time",
    })
    stop_times = stop_times.rename(columns={
        "id": "stop_time_id", "gtfs_ride_id": "trip_id", "gtfs_stop_id": "stop_id",
        "arrival_time": "arrival_time_raw", "departure_time": "departure_time_raw",
    })

    for col, df in [("stop_id", stops), ("route_id", routes),
                    ("trip_id", rides), ("route_id", rides),
                    ("trip_id", stop_times), ("stop_id", stop_times)]:
        if col in df.columns:
            df[col] = df[col].astype("string")

    # --- Geographic filter ---
    if bbox is not None and "stop_lat" in stops.columns:
        stops = stops[
            stops["stop_lat"].between(bbox.min_lat, bbox.max_lat) &
            stops["stop_lon"].between(bbox.min_lon, bbox.max_lon)
        ].copy()

    # --- Vectorized time parsing ---
    stop_times["arrival_time_sec"] = _parse_iso_time_to_seconds(stop_times["arrival_time_raw"])
    stop_times["departure_time_sec"] = _parse_iso_time_to_seconds(stop_times["departure_time_raw"])
    stop_times["departure_time_sec"] = (
        stop_times["departure_time_sec"].fillna(stop_times["arrival_time_sec"])
    )
    stop_times = stop_times.dropna(subset=["arrival_time_sec"]).copy()
    stop_times["arrival_time"] = _seconds_to_hhmmss(stop_times["arrival_time_sec"])
    stop_times["departure_time"] = _seconds_to_hhmmss(stop_times["departure_time_sec"])

    if "stop_sequence" in stop_times.columns:
        stop_times["stop_sequence"] = pd.to_numeric(
            stop_times["stop_sequence"], errors="coerce"
        ).fillna(0).astype(int)
    else:
        stop_times = stop_times.sort_values(["trip_id", "arrival_time_sec"])
        stop_times["stop_sequence"] = stop_times.groupby("trip_id").cumcount() + 1

    # --- Prune orphans ---
    stop_times = stop_times[
        stop_times["stop_id"].isin(stops["stop_id"]) &
        stop_times["trip_id"].isin(rides["trip_id"])
    ]
    valid_trips = stop_times["trip_id"].unique()
    valid_stops = stop_times["stop_id"].unique()
    trips = rides[rides["trip_id"].isin(valid_trips)].copy()
    routes_pruned = routes[routes["route_id"].isin(trips["route_id"])].copy()
    stops_pruned = stops[stops["stop_id"].isin(valid_stops)].copy()

    if stops_pruned.empty or routes_pruned.empty or trips.empty or stop_times.empty:
        return None

    return GTFSFrames(
        stops=stops_pruned,
        routes=routes_pruned,
        trips=trips,
        stop_times=stop_times.reset_index(drop=True),
    )


def gtfs_to_edge_weights(stop_times: pd.DataFrame) -> pd.DataFrame:
    """Build (from_stop, to_stop, weight) edges across all trips.

    Weight = number of trips traversing that consecutive-stop pair.
    Vectorized: groupby + shift, no Python loop.
    """
    if stop_times.empty:
        return pd.DataFrame(columns=["from_stop", "to_stop", "weight"])

    df = stop_times.sort_values(["trip_id", "stop_sequence"]).copy()
    df["to_stop"] = df.groupby("trip_id")["stop_id"].shift(-1)
    edges = df.dropna(subset=["to_stop"])[["stop_id", "to_stop"]]
    edges.columns = ["from_stop", "to_stop"]
    return (
        edges.groupby(["from_stop", "to_stop"], as_index=False)
             .size()
             .rename(columns={"size": "weight"})
    )
