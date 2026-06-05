"""
Network Graph — port of the original `main_.py` (GTFS network grapher).

Pulls stops, routes, rides, and ride_stops for a chosen date+time window+bbox,
builds a NetworkX graph weighted by service frequency, and renders it as a
matplotlib figure plus an interactive map of stops.

Uses `st.status` for step-by-step progress visible to the user, replacing the
legacy ad-hoc `st.write` / `st.info` calls scattered through the original.
"""
from __future__ import annotations

from datetime import date, time, timedelta
from typing import Final

import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

from api_client import StrideAPIError
from data_processor import BBox, GTFSFrames, clean_gtfs_frames, gtfs_to_edge_weights
from ui_components import SidebarState, get_client, report_api_error


# Default geo window — Tel Aviv center.
DEFAULT_BBOX: Final[BBox] = BBox(min_lat=32.05, max_lat=32.10, min_lon=34.76, max_lon=34.82)


def render(state: SidebarState) -> None:
    st.subheader("🌐 GTFS Network Graph")
    st.caption("Build a frequency-weighted graph from scheduled service.")

    params = _render_controls()
    if not st.button("🚀 Build network", type="primary"):
        st.info("Configure the date / time / bbox, then click **Build network**.")
        return

    frames = _fetch_and_clean(params)
    if frames is None:
        return

    _show_summary(frames)
    _render_graph(frames)
    _render_stop_map(frames)


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

class _NetworkParams:
    __slots__ = ("date_str", "t_from", "t_to", "bbox")

    def __init__(self, d: date, t_from: time, t_to: time, bbox: BBox) -> None:
        self.date_str = d.isoformat()
        self.t_from = t_from.strftime("%H:%M:%S")
        self.t_to = t_to.strftime("%H:%M:%S")
        self.bbox = bbox


def _render_controls() -> _NetworkParams:
    col1, col2 = st.columns(2)
    with col1:
        the_date = st.date_input("Service date", value=date(2026, 1, 13), key="ng_date")
    with col2:
        t_from, t_to = st.slider(
            "Time window",
            value=(time(6, 0), time(12, 0)),
            min_value=time(0, 0),
            max_value=time(23, 59),
            step=timedelta(minutes=30),
            format="HH:mm",
            key="ng_window",
        )

    st.markdown("**Bounding box**")
    c1, c2 = st.columns(2)
    with c1:
        min_lon = st.number_input("Min lon", value=DEFAULT_BBOX.min_lon, format="%.4f", key="ng_min_lon")
        min_lat = st.number_input("Min lat", value=DEFAULT_BBOX.min_lat, format="%.4f", key="ng_min_lat")
    with c2:
        max_lon = st.number_input("Max lon", value=DEFAULT_BBOX.max_lon, format="%.4f", key="ng_max_lon")
        max_lat = st.number_input("Max lat", value=DEFAULT_BBOX.max_lat, format="%.4f", key="ng_max_lat")

    return _NetworkParams(
        d=the_date,
        t_from=t_from,
        t_to=t_to,
        bbox=BBox(min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon),
    )


# ---------------------------------------------------------------------------
# Fetch & clean
# ---------------------------------------------------------------------------

def _fetch_and_clean(p: _NetworkParams) -> GTFSFrames | None:
    """Run the four API calls inside a single `st.status` block."""
    client = get_client()
    iso_from = f"{p.date_str}T{p.t_from}+02:00"
    iso_to = f"{p.date_str}T{p.t_to}+02:00"

    with st.status("Building network from Stride API...", expanded=True) as status:
        try:
            status.update(label="Fetching stops...")
            stops = client.list_all("gtfs_stops/list", {
                "date_from": p.date_str, "date_to": p.date_str,
            })
            status.write(f"• {len(stops):,} stops returned")

            status.update(label="Fetching routes...")
            routes = client.list_all("gtfs_routes/list", {
                "date_from": p.date_str, "date_to": p.date_str,
            })
            status.write(f"• {len(routes):,} routes returned")

            status.update(label="Fetching rides in time window...")
            rides = client.list_all("gtfs_rides/list", {
                "start_time_from": iso_from, "start_time_to": iso_to,
            })
            status.write(f"• {len(rides):,} rides returned")

            status.update(label="Fetching ride stops (this is the slow one)...")
            stop_times = client.list_all("gtfs_ride_stops/list", {
                "arrival_time_from": f"{p.date_str}T00:00:00+02:00",
                "arrival_time_to": f"{p.date_str}T23:59:59+02:00",
            })
            status.write(f"• {len(stop_times):,} ride-stops returned")

            status.update(label="Cleaning and aligning frames...")
            frames = clean_gtfs_frames(stops, routes, rides, stop_times, bbox=p.bbox)

        except StrideAPIError as e:
            status.update(label="API request failed", state="error", expanded=True)
            report_api_error(e, where="Network Graph")
            return None

        if frames is None:
            status.update(label="No usable data after cleaning", state="error")
            st.warning("Empty result. Try a wider bbox or a different date.")
            return None

        status.update(
            label=(
                f"Done · {len(frames.stops):,} stops · "
                f"{len(frames.routes):,} routes · "
                f"{len(frames.trips):,} trips · "
                f"{len(frames.stop_times):,} stop-times"
            ),
            state="complete",
        )
        return frames


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _show_summary(frames: GTFSFrames) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stops", f"{len(frames.stops):,}")
    c2.metric("Routes", f"{len(frames.routes):,}")
    c3.metric("Trips", f"{len(frames.trips):,}")
    c4.metric("Stop times", f"{len(frames.stop_times):,}")


def _build_graph(frames: GTFSFrames) -> nx.Graph:
    g = nx.Graph()
    for _, stop in frames.stops.iterrows():
        g.add_node(
            stop["stop_id"],
            pos=(float(stop["stop_lon"]), float(stop["stop_lat"])),
            name=stop.get("stop_name", f"Stop {stop['stop_id']}"),
        )
    edges = gtfs_to_edge_weights(frames.stop_times)
    for _, row in edges.iterrows():
        if g.has_node(row["from_stop"]) and g.has_node(row["to_stop"]):
            g.add_edge(row["from_stop"], row["to_stop"], weight=int(row["weight"]))
    return g


def _render_graph(frames: GTFSFrames) -> None:
    st.subheader("📈 Network plot")
    g = _build_graph(frames)
    if g.number_of_edges() == 0:
        st.warning("Graph has no edges — too little data to draw.")
        return

    pos = {n: g.nodes[n]["pos"] for n in g.nodes()}
    weights = [g[u][v]["weight"] for u, v in g.edges()]
    w_min, w_max = min(weights), max(weights)
    if w_max > w_min:
        widths = [2 + 6 * (w - w_min) / (w_max - w_min) for w in weights]
    else:
        widths = [3] * len(weights)

    fig, ax = plt.subplots(figsize=(12, 10))
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=40, node_color="#ff4d4f", alpha=0.85)
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#888", width=widths, alpha=0.55)

    # Label only the busiest hubs.
    hubs = sorted(g.degree, key=lambda kv: kv[1], reverse=True)[:8]
    labels = {n: (g.nodes[n]["name"] or "")[:20] for n, _ in hubs}
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=9, ax=ax)

    ax.set_title("Service-frequency-weighted stop network")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.2)
    st.pyplot(fig, clear_figure=True)

    st.markdown(
        f"**Edges:** {g.number_of_edges():,} · "
        f"**Avg degree:** {sum(dict(g.degree).values()) / max(g.number_of_nodes(), 1):.2f} · "
        f"**Density:** {nx.density(g):.4f}"
    )


def _render_stop_map(frames: GTFSFrames) -> None:
    st.subheader("📍 Stop locations")
    stops_for_map = frames.stops[["stop_lat", "stop_lon", "stop_name"]].rename(
        columns={"stop_lat": "lat", "stop_lon": "lon"}
    )
    st.map(stops_for_map, use_container_width=True)
