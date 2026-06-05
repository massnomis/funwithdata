"""
Raw API Explorer — pick an endpoint, edit JSON params, run, inspect the response.

A no-frills debug tool. Useful when adding new modes: try queries here first.
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from api_client import StrideAPIError, StrideValidationError
from ui_components import SidebarState, get_client, report_api_error


_ENDPOINTS = (
    "gtfs_agencies/list",
    "gtfs_routes/list",
    "gtfs_stops/list",
    "gtfs_rides/list",
    "gtfs_ride_stops/list",
    "siri_vehicle_locations/list",
    "siri_rides/list",
)


def render(_: SidebarState) -> None:
    st.subheader("🔍 API Explorer")
    st.caption("Send raw requests to the Stride API. Output is shown as JSON and as a table.")

    col_ep, col_params = st.columns([1, 2])
    endpoint = col_ep.selectbox("Endpoint", _ENDPOINTS, key="ax_endpoint")
    default_params = '{"limit": 5}'
    params_text = col_params.text_area(
        "Parameters (JSON)",
        value=default_params,
        key="ax_params",
        height=120,
        help="Object of query parameters, e.g. `{\"date_from\": \"2026-01-13\", \"limit\": 10}`",
    )

    if not st.button("▶️ Run query", type="primary"):
        st.info("Edit the params, then run.")
        return

    try:
        params = json.loads(params_text) if params_text.strip() else {}
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")
        return

    with st.spinner(f"Calling {endpoint}..."):
        try:
            data = get_client().get(endpoint, params)
        except StrideValidationError as e:
            report_api_error(e, where=endpoint)
            return
        except StrideAPIError as e:
            report_api_error(e, where=endpoint)
            return

    _render_response(data)


def _render_response(data) -> None:
    if isinstance(data, list):
        st.success(f"Got {len(data):,} records.")
        if data:
            tab_json, tab_table = st.tabs(["JSON", "Table"])
            with tab_json:
                # Show only the first 5 entries in JSON to avoid clogging the UI.
                st.json(data[:5])
                if len(data) > 5:
                    st.caption(f"Showing first 5 of {len(data):,} records.")
            with tab_table:
                st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
        else:
            st.info("Empty response.")
    else:
        st.json(data)
