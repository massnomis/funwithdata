import streamlit as st
from ui_components import render_deck, build_user_pin_layer

st.write("map test")
render_deck(
    [build_user_pin_layer(32.0853, 34.7818, 150)],
    center_lat=32.0853, center_lon=34.7818, zoom=14, pitch=0,
)
