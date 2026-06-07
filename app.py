"""
Stride Explorer — top-level entry point.

Responsibilities (intentionally small):
    1. Configure the Streamlit page
    2. Render the sidebar (delegated to ui_components)
    3. Dispatch to the chosen mode module

The mode modules own their own data fetching, layout, and state. Caching
policy lives in `ui_components`. This file should stay short.
"""
from __future__ import annotations

import streamlit as st

from modes import api_explorer, data_manager, live_traffic, network_graph, route_explorer
from ui_components import SidebarState, render_sidebar

# ---------------------------------------------------------------------------
# Page setup — must happen before any other Streamlit call.
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Stride Explorer",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Open Bus Stride API explorer · Israeli public transit · "
                 "Data by Hasadna.",
    },
)


# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------

# Mapping of sidebar labels to renderer callables. Adding a mode = adding a row.
_MODE_DISPATCH = {
    "📡 Live Traffic":   live_traffic.render,
    "🗺️ Route Explorer": route_explorer.render,
    "🌐 Network Graph":  network_graph.render,
    "💾 Data Manager":   data_manager.render,
    "🔍 API Explorer":   api_explorer.render,
}


def main() -> None:
    state: SidebarState = render_sidebar()
    renderer = _MODE_DISPATCH.get(state.mode)
    if renderer is None:
        st.error(f"Unknown mode: {state.mode}")
        st.stop()
    renderer(state)


if __name__ == "__main__":
    main()
