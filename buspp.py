import streamlit as st
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pydeck as pdk
from datetime import datetime, timedelta, timezone
import math
import json
import time
import random
import os
import pytz

# --- Configuration ---
API_BASE_URL = "https://open-bus-stride-api.hasadna.org.il"
st.set_page_config(page_title="Stride Ultimate Explorer",
                   layout="wide",
                   page_icon="🚌")

# --- Custom Styles ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .stMetricValue { font-size: 1.5rem !important; }
    </style>
    """,
            unsafe_allow_html=True)

# --- Helper Functions ---
def get_session():
    """Creates a requests session with retry logic and longer timeouts."""
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504, 429],
        allowed_methods=["GET"],
        respect_retry_after_header=True
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=100)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def fetch_api(endpoint, params=None):
    """Safe API fetcher with centralized debug handling."""
    session = get_session()
    if params is None:
        params = {}

    # Remove None values
    params = {k: v for k, v in params.items() if v is not None}
    
    # --- DEBUG: Show Request ---
    if st.session_state.get('debug_mode'):
        with st.expander(f"📤 REQ: {endpoint}", expanded=False):
            st.json(params)

    try:
        # Timeout increased to 120s to handle limit=-1 bulk downloads
        res = session.get(f"{API_BASE_URL}/{endpoint}",
                          params=params,
                          timeout=120)
        res.raise_for_status()
        
        # Check if response is valid JSON
        try:
            data = res.json()
        except json.JSONDecodeError:
            st.error(f"❌ Invalid JSON response from {endpoint}")
            return []

        # --- DEBUG: Show Response ---
        if st.session_state.get('debug_mode'):
            with st.expander(
                    f"📥 RES: {endpoint} (Count: {len(data) if isinstance(data, list) else 1})",
                    expanded=False):
                preview = data[:5] if isinstance(
                    data, list) and len(data) > 5 else data
                st.write(
                    f"Showing {len(preview)} of {len(data) if isinstance(data, list) else 1} items:"
                )
                st.json(preview)

        return data
    except requests.exceptions.ReadTimeout:
        st.error(
            f"⏱️ API Timeout ({endpoint}): The server is slow. Try reducing the radius or lookback."
        )
        return []
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 422:
            st.error(f"❌ Validation Error ({endpoint}): Check your parameters")
            try:
                error_details = e.response.json()
                st.error(f"Details: {error_details}")
            except:
                pass
        else:
            st.error(f"❌ HTTP Error ({endpoint}): {e}")
        return []
    except Exception as e:
        st.error(f"❌ API Error ({endpoint}): {e}")
        return []

def get_distinct_color(index):
    """Returns a bright, distinct color based on an index."""
    colors = [
        [0, 255, 150],  # Bright Green
        [255, 0, 100],  # Bright Red/Pink
        [0, 200, 255],  # Cyan
        [255, 200, 0],  # Gold
        [200, 100, 255],  # Purple
        [255, 100, 50],  # Orange
        [50, 50, 255],  # Blue
        [255, 255, 255]  # White
    ]
    return colors[index % len(colors)]

def haversine_bbox(lat, lon, radius_km):
    """Calculates a rough bounding box (min_lat, min_lon, max_lat, max_lon) given a center and radius."""
    lat_change = radius_km / 111.0
    lon_change = abs(radius_km / (111.0 * math.cos(math.radians(lat))))
    return {
        "min_lat": lat - lat_change,
        "max_lat": lat + lat_change,
        "min_lon": lon - lon_change,
        "max_lon": lon + lon_change
    }

def get_bearing_icon(bearing):
    """Converts a degree bearing to an emoji arrow and cardinal direction."""
    if pd.isna(bearing): return "❓ Unknown"
    try:
        b = int(bearing)
        if b >= 337 or b < 23: return "⬆️ N"
        if 23 <= b < 67: return "↗️ NE"
        if 67 <= b < 112: return "➡️ E"
        if 112 <= b < 157: return "↘️ SE"
        if 157 <= b < 202: return "⬇️ S"
        if 202 <= b < 247: return "↙️ SW"
        if 247 <= b < 292: return "⬅️ W"
        if 292 <= b < 337: return "↖️ NW"
    except:
        pass
    return "❓"

def get_bearing_icon_url(bearing):
    """Converts a degree bearing to an emoji arrow and cardinal direction."""
    if pd.isna(bearing): return None
    try:
        b = int(bearing)
        if b >= 337 or b < 23:
            return {
                "url": "https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/2b06.png",
                "width": 28,
                "height": 28,
                "anchorY": 28,
            }
        if 23 <= b < 67:
            return {
                "url": "https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/2197.png",
                "width": 28,
                "height": 28,
                "anchorY": 28,
            }
        if 67 <= b < 112:
            return {
                "url": "https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/27a1.png",
                "width": 28,
                "height": 28,
                "anchorY": 28,
            }
        if 112 <= b < 157:
            return {
                "url": "https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/2198.png",
                "width": 28,
                "height": 28,
                "anchorY": 28,
            }
        if 157 <= b < 202:
            return {
                "url": "https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/2b07.png",
                "width": 28,
                "height": 28,
                "anchorY": 28,
            }
        if 202 <= b < 247:
            return {
                "url": "https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/2199.png",
                "width": 28,
                "height": 28,
                "anchorY": 28,
            }
        if 247 <= b < 292:
            return {
                "url": "https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/2b05.png",
                "width": 28,
                "height": 28,
                "anchorY": 28,
            }
        if 292 <= b < 337:
            return {
                "url": "https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/2196.png",
                "width": 28,
                "height": 28,
                "anchorY": 28,
            }
    except:
        pass
    return None

# --- Data Management Helpers (Local Cache) ---
def get_local_file_path(filename):
    """Returns path to data/{today}/{filename}, creating dir if needed."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    folder_path = os.path.join("data", today)
    os.makedirs(folder_path, exist_ok=True)
    return os.path.join(folder_path, filename)

def fetch_all_routes_today():
    """Fetches ALL active routes for the current day using limit=-1."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    params = {"date_from": today, "date_to": today, "limit": -1}
    return fetch_api("gtfs_routes/list", params)

def fetch_all_stops_today():
    """Fetches ALL active stops for the current day using limit=-1."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    params = {"date_from": today, "date_to": today, "limit": -1}
    return fetch_api("gtfs_stops/list", params)

def get_or_fetch_routes():
    """Checks local disk for routes.json; otherwise fetches and saves."""
    file_path = get_local_file_path("routes.json")

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f), "disk", file_path

    # Not found, fetch from API
    data = fetch_all_routes_today()
    if data:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return data, "api", file_path

    return [], "error", None

def get_or_fetch_agencies():
    """Checks local disk for agencies.json; otherwise fetches and saves."""
    file_path = get_local_file_path("agencies.json")

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f), "disk", file_path

    # Agencies are few, limit=-1 is safest
    data = fetch_api("gtfs_agencies/list", {"limit": -1})
    if data:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return data, "api", file_path

    return [], "error", None

def get_or_fetch_stops():
    """Checks local disk for stops.json; otherwise fetches and saves."""
    file_path = get_local_file_path("stops.json")

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f), "disk", file_path

    # Not found, fetch from API
    data = fetch_all_stops_today()
    if data:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return data, "api", file_path

    return [], "error", None

def get_agency_lookup_local():
    """Returns a dict mapping operator_ref -> agency_name using local/cached data."""
    agencies, _, _ = get_or_fetch_agencies()
    if not agencies:
        return {}
    # Key must be string to match enriched data
    return {
        str(a['operator_ref']): a['agency_name']
        for a in agencies if a.get('operator_ref')
    }

def get_master_routes_df():
    """
    Loads routes and merges with agency data to create a master DataFrame.
    """
    # 1. Load Routes
    routes_data, _, _ = get_or_fetch_routes()
    if not routes_data:
        return pd.DataFrame()

    df_routes = pd.DataFrame(routes_data)

    # 2. Load Agencies
    agency_lookup = get_agency_lookup_local()

    # 3. Enrich Routes with Agency Name
    if 'operator_ref' in df_routes.columns:
        # Cast to string for lookup
        df_routes['operator_ref_str'] = df_routes['operator_ref'].astype(str)
        df_routes['agency_name'] = df_routes['operator_ref_str'].apply(
            lambda x: agency_lookup.get(x, f"Unknown ({x})"))
    else:
        df_routes['agency_name'] = "Unknown"

    return df_routes

def fetch_live_siri_data(lat, lon, radius_km, lookback_min=10):
    """
    Fetches real-time vehicle locations (SIRI) within a bounding box.
    """
    bbox = haversine_bbox(lat, lon, radius_km)
    
    # Use timezone-aware timestamps
    israel_tz = pytz.timezone('Asia/Jerusalem')
    now = datetime.now(israel_tz)
    from_time = now - timedelta(minutes=lookback_min)
    
    params = {
        "recorded_at_time_from": from_time.isoformat(),
        "lat__greater_or_equal": bbox["min_lat"],
        "lat__lower_or_equal": bbox["max_lat"],
        "lon__greater_or_equal": bbox["min_lon"],
        "lon__lower_or_equal": bbox["max_lon"],
        "limit": -1,
        "order_by": "recorded_at_time desc"
    }

    return fetch_api("siri_vehicle_locations/list", params)

def enrich_siri_data_with_local(siri_df, master_routes_df):
    """
    Enriches SIRI data using the pre-loaded Master Routes DataFrame.
    """
    if siri_df.empty or master_routes_df.empty:
        return siri_df

    # 1. Build Lookups from Master Data
    id_lookup = master_routes_df.set_index('id')[[
        'route_short_name', 'route_long_name', 'agency_name'
    ]].to_dict('index')

    # B. Fallback Line Ref Lookup (Generic)
    master_routes_df['line_ref_str'] = master_routes_df['line_ref'].astype(str)
    master_routes_df.sort_values(by='id', ascending=False, inplace=True)
    unique_routes = master_routes_df.drop_duplicates(subset=['line_ref_str'])
    ref_lookup = unique_routes.set_index('line_ref_str')[[
        'route_short_name', 'route_long_name', 'agency_name', 'id'
    ]].to_dict('index')

    def apply_enrichment(row):
        # 1. Try Specific GTFS Route ID
        specific_route_id = row.get('gtfs_ride__gtfs_route_id')
        if specific_route_id and specific_route_id in id_lookup:
            info = id_lookup[specific_route_id]
            return pd.Series([
                info['route_short_name'], info['route_long_name'],
                info['agency_name'], specific_route_id
            ])

        # 2. Try SIRI Line Ref
        ref = str(row.get('siri_route__line_ref') or row.get('line_ref'))

        if ref and ref in ref_lookup:
            info = ref_lookup[ref]
            return pd.Series([
                info['route_short_name'], info['route_long_name'],
                info['agency_name'], info['id']
            ])

        return pd.Series(['Unknown', 'Unknown', 'Unknown', None])

    # Apply columns
    siri_df[[
        'route_short_name', 'route_long_name', 'agency_name', 'gtfs_route_id'
    ]] = siri_df.apply(apply_enrichment, axis=1)

    return siri_df

def get_timetable_details(route_id):
    """
    Fetches the shape (path) and stops for a specific GTFS route ID.
    """
    try:
        route_id = int(float(route_id))
    except (ValueError, TypeError):
        return None, f"Invalid Route ID: {route_id}"

    # Use Israel timezone for date ranges
    israel_tz = pytz.timezone('Asia/Jerusalem')
    
    # Get today's date in Israel timezone
    today = datetime.now(israel_tz).date()
    
    # Step 1: Get scheduled rides for today
    ride_params = {
        "gtfs_route__date_from": today.isoformat(),
        "gtfs_route__date_to": today.isoformat(),
        "gtfs_route_id": route_id,
        "order_by": "start_time asc",
        "limit": 1
    }

    rides = fetch_api("gtfs_rides/list", ride_params)

    if not rides:
        return None, "No scheduled rides found for today."

    ride = rides[0]
    ride_id = ride['id']
    
    # Step 2: Get stops for this ride
    stops_params = {
        "gtfs_ride_ids": str(ride_id),
        "order_by": "stop_sequence",
        "limit": 1000,
    }

    stops_data = fetch_api("gtfs_ride_stops/list", stops_params)

    if not stops_data:
        return None, "No stops linked."

    path_data = []
    final_stops = []

    for item in stops_data:
        if item.get('gtfs_ride_id') != ride_id:
            continue

        lat = item.get('gtfs_stop__lat')
        lon = item.get('gtfs_stop__lon')
        name = item.get('gtfs_stop__name', 'Unknown')
        code = item.get('gtfs_stop__code')

        if lat and lon:
            coords = [lon, lat]
            path_data.append(coords)
            final_stops.append({
                "name": name,
                "code": code,
                "time": item.get('arrival_time') or "N/A",
                "seq": item.get('stop_sequence'),
                "coordinates": coords
            })

    if not final_stops:
        return None, "No valid coordinates."

    final_stops.sort(key=lambda x: x['seq'])
    path_coords = [s['coordinates'] for s in final_stops]

    return {"path": path_coords, "stops": final_stops, "ride_id": ride_id}, None

def sanitize_siri_data(siri_df):
    """Clean and validate SIRI data."""
    if siri_df.empty:
        return siri_df
    
    # Ensure required columns exist
    required_cols = ['lat', 'lon']
    for col in required_cols:
        if col not in siri_df.columns:
            siri_df[col] = None
    
    # Convert numeric columns
    numeric_cols = ['lat', 'lon', 'bearing', 'velocity']
    for col in numeric_cols:
        if col in siri_df.columns:
            siri_df[col] = pd.to_numeric(siri_df[col], errors='coerce')
    
    # Drop invalid rows
    siri_df = siri_df.dropna(subset=['lat', 'lon'])
    
    # Filter out unrealistic coordinates
    siri_df = siri_df[
        (siri_df['lat'].between(29.0, 34.0)) &
        (siri_df['lon'].between(34.0, 36.0))
    ]
    
    # Remove duplicates
    siri_df = siri_df.drop_duplicates(
        subset=['siri_ride__vehicle_ref', 'recorded_at_time'], 
        keep='last'
    )
    
    return siri_df

def create_map_layers(siri_df, live_paths, user_lat, user_lon, viz_path_width, viz_pin_radius):
    """Create optimized map layers for better performance."""
    layers = []
    
    # 1. Bus Location Layer (Optimized)
    if not siri_df.empty:
        # Filter out buses with invalid locations
        valid_buses = siri_df.dropna(subset=['lat', 'lon']).copy()
        
        # Add icon data
        valid_buses['icon_data'] = valid_buses['bearing'].apply(get_bearing_icon_url)
        
        # Create Icon Layer
        layers.append(
            pdk.Layer(
                "IconLayer",
                data=valid_buses.to_dict('records'),
                get_icon="icon_data",
                get_size=4,
                size_scale=5,
                get_position='[lon, lat]',
                pickable=True,
            )
        )
    
    # 2. Route Paths Layer (Only if needed)
    if live_paths and len(live_paths) <= 50:
        path_layer_data = []
        for path_info in live_paths[:50]:
            if len(path_info.get('path', [])) >= 2:
                path_layer_data.append({
                    "path": path_info['path'],
                    "color": path_info.get('color', [0, 255, 0])
                })
        
        if path_layer_data:
            layers.append(
                pdk.Layer(
                    "PathLayer",
                    data=path_layer_data,
                    get_path="path",
                    get_width=viz_path_width,
                    get_color="color",
                    width_min_pixels=2,
                    pickable=False
                )
            )
    
    # 3. User Location
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=[{"lat": user_lat, "lon": user_lon}],
            get_position='[lon, lat]',
            get_radius=viz_pin_radius,
            get_fill_color=[255, 0, 0, 180],
            stroked=True,
            get_line_color=[255, 255, 255],
            get_line_width=2,
        )
    )
    
    return layers

def draw_route_paths(route_ids):
    """Draw multiple route paths with error handling."""
    all_paths = []
    all_stops = []
    errors = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, r_id in enumerate(route_ids):
        status_text.text(f"Fetching route {i+1}/{len(route_ids)} (ID: {r_id})...")
        
        try:
            data, err = get_timetable_details(r_id)
            if err:
                errors.append(f"Route {r_id}: {err}")
            elif data:
                if data.get('path'):
                    all_paths.append({
                        "path": data['path'],
                        "color": get_distinct_color(i)
                    })
                if data.get('stops'):
                    for s in data['stops']:
                        s['color'] = get_distinct_color(i)
                    all_stops.extend(data['stops'])
        except Exception as e:
            errors.append(f"Route {r_id}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(route_ids))
    
    progress_bar.empty()
    status_text.empty()
    
    # Show errors if any
    if errors:
        with st.expander("⚠️ Some routes had issues", expanded=False):
            for error in errors[:5]:
                st.error(error)
    
    return all_paths, all_stops

def check_api_health():
    """Check if the API is available and responsive."""
    try:
        response = requests.get(f"{API_BASE_URL}/gtfs_routes/list", 
                               params={"limit": 1}, 
                               timeout=10)
        return response.status_code == 200
    except:
        return False

def optimize_dataframe_operations(df):
    """Optimize DataFrame for memory and performance."""
    if df.empty:
        return df
    
    # Downcast numeric columns
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Convert object columns to categorical where appropriate
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype('category')
    
    return df

@st.cache_data(ttl=3600)
def get_filtered_routes(search_mode, search_query=None, selected_agency=None):
    """Cache filtered routes to improve performance."""
    df_routes = get_master_routes_df()
    
    if df_routes.empty:
        return pd.DataFrame()
    
    if search_mode == "Search by Line Number" and search_query:
        filtered = df_routes[
            (df_routes['route_short_name'] == search_query) |
            (df_routes['route_long_name'].str.contains(search_query, case=False, na=False))
        ]
    elif search_mode == "Browse by Operator" and selected_agency:
        filtered = df_routes[df_routes['agency_name'] == selected_agency]
    else:
        filtered = pd.DataFrame()
    
    return filtered

# --- Sidebar Controls ---
with st.sidebar:
    st.title("🎛️ Control Panel")

    app_mode = st.radio("Select Mode", [
        "📡 Live Traffic (SIRI)", "🗺️ Route Explorer", "💾 Bulk Data Manager",
        "🔍 API Data Explorer"
    ])

    st.divider()

    # --- VISUALIZATION SETTINGS ---
    with st.expander("🎨 Visualization Settings", expanded=False):
        viz_dot_radius = st.slider("Dot Radius (Meters)", 10, 500, 50)
        viz_path_width = st.slider("Path Width (Meters)", 5, 200, 30)
        viz_arrow_size = st.slider("Arrow Size", 10, 100, 45)
        viz_elevation_scale = st.slider("3D Speed Scale", 0, 50, 10)
        viz_pin_radius = st.slider("User Pin Radius", 50, 1000, 150)
        viz_max_paths = st.slider(
            "Max Live Paths Limit",
            10,
            500,
            50,
            help="How many route lines to draw simultaneously in Live Mode."
        )

    # Debug Toggle
    st.toggle("🐞 Enable Debug Mode",
              key="debug_mode",
              help="Show API requests and responses.")

    if app_mode not in ["🔍 API Data Explorer", "💾 Bulk Data Manager"]:
        st.subheader("📍 My Area")
        city_preset = st.selectbox(
            "Quick Select",
            ["Tel Aviv", "Jerusalem", "Haifa", "Be'er Sheva", "Custom"])

        coords = {
            "Tel Aviv": (32.0853, 34.7818),
            "Jerusalem": (31.7683, 35.2137),
            "Haifa": (32.7940, 34.9896),
            "Be'er Sheva": (31.2529, 34.7914),
            "Custom": (32.0, 34.8)
        }

        default_lat, default_lon = coords[city_preset]

        if city_preset == "Custom":
            user_lat = st.number_input("Latitude",
                                       value=default_lat,
                                       format="%.4f")
            user_lon = st.number_input("Longitude",
                                       value=default_lon,
                                       format="%.4f")
        else:
            user_lat, user_lon = default_lat, default_lon

        if app_mode == "📡 Live Traffic (SIRI)":
            radius = st.slider("Scan Radius (km)", 0.5, 10.0, 2.0)
            lookback = st.slider(
                "Lookback Minutes",
                1,
                30,
                10,
                help="How far back to check for vehicle pings.")

    st.info("Data provided by Open Bus Stride API (Hasadna)")

# --- Main App Logic ---
st.title("🚌 Stride Ultimate Explorer")

# API Health Check
if not check_api_health():
    st.error("⚠️ Open Bus API is currently unavailable or slow. Some features may not work properly.")
    st.info("You can still use cached data or try again later.")

# --- Load Master Data Once ---
if app_mode in ["📡 Live Traffic (SIRI)", "🗺️ Route Explorer"]:
    if 'master_routes' not in st.session_state or st.session_state[
            'master_routes'].empty:
        with st.spinner("Loading local route database..."):
            st.session_state['master_routes'] = get_master_routes_df()

if app_mode == "📡 Live Traffic (SIRI)":
    st.markdown(
        f"### Real-Time Bus Activity near {city_preset if city_preset != 'Custom' else 'Selected Area'}"
    )

    col_ctrl, col_refresh = st.columns([4, 1])
    with col_refresh:
        if st.button("🔄 Refresh Live Data",
                     type="primary",
                     use_container_width=True):
            st.cache_data.clear()

    with st.spinner("Ping satellites... fetching SIRI data..."):
        raw_siri = fetch_live_siri_data(user_lat, user_lon, radius, lookback)

    if raw_siri:
        siri_df = pd.DataFrame(raw_siri)
        siri_df = sanitize_siri_data(siri_df)

        # Enrich Data using MASTER DataFrame
        if not st.session_state['master_routes'].empty:
            siri_df = enrich_siri_data_with_local(
                siri_df, st.session_state['master_routes'])

        # Add Compass Direction and Angle for visual
        if 'bearing' in siri_df.columns:
            siri_df['compass_direction'] = siri_df['bearing'].apply(
                get_bearing_icon)
            siri_df['bearing_angle'] = 90 - siri_df['bearing'].fillna(0)
        else:
            siri_df['compass_direction'] = "?"
            siri_df['bearing_angle'] = 0

        # Calculate Z-height for arrows
        siri_df['z_height'] = 200 + (siri_df['velocity'].fillna(0) * 10)

        # Date Conversion
        if 'recorded_at_time' in siri_df.columns:
            siri_df['recorded_at_time'] = pd.to_datetime(
                siri_df['recorded_at_time'], errors='coerce')

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Active Buses", len(siri_df))
        m2.metric(
            "Avg Speed", f"{siri_df['velocity'].mean():.1f} km/h"
            if 'velocity' in siri_df and not siri_df['velocity'].isnull().all() else "N/A")
        m3.metric(
            "Max Speed", f"{siri_df['velocity'].max():.1f} km/h"
            if 'velocity' in siri_df and not siri_df['velocity'].isnull().all() else "N/A")
        m4.metric(label="Last Update",
                  value=(datetime.now() +
                         timedelta(hours=2)).strftime("%H:%M:%S"),
                  delta="Live Data")
        
        st.markdown("### 🚦 Traffic Control")

        available_lines = sorted(
            siri_df['route_short_name'].astype(str).unique())
        selected_lines = st.multiselect(
            "Filter Active Buses (Select Lines)",
            options=available_lines,
            placeholder="Show all lines or select specific ones...")

        # Toggle for Route Paths
        show_paths = st.checkbox("Show Route Paths (Overlay)",
                                 value=False,
                                 help="Draws the lines for active buses.")

        # Route Shapes Logic
        live_paths = []
        route_ids_to_draw = []

        if selected_lines:
            siri_df = siri_df[siri_df['route_short_name'].astype(str).isin(
                selected_lines)]
            route_ids_to_draw = siri_df['gtfs_route_id'].dropna().unique().tolist()
        elif show_paths:
            unique_routes = siri_df['gtfs_route_id'].dropna().unique().tolist()
            route_ids_to_draw = unique_routes[:viz_max_paths]

            if len(unique_routes) > viz_max_paths:
                st.toast(
                    f"⚠️ Limiting route paths to top {viz_max_paths} lines (out of {len(unique_routes)}) for performance."
                )

        if route_ids_to_draw:
            with st.spinner(
                    f"Fetching route shapes for {len(route_ids_to_draw)} variants..."
            ):
                for i, r_id in enumerate(route_ids_to_draw):
                    route_details, _ = get_timetable_details(r_id)
                    if route_details and route_details.get('path'):
                        live_paths.append({
                            "path": route_details['path'],
                            "color": get_distinct_color(i)
                        })

        # Create Map Layers
        layers = create_map_layers(siri_df, live_paths, user_lat, user_lon, viz_path_width, viz_pin_radius)

        # Final Map Output
        st.pydeck_chart(
            pdk.Deck(
                map_style=None,
                initial_view_state=pdk.ViewState(latitude=user_lat,
                                                 longitude=user_lon,
                                                 zoom=13,
                                                 pitch=45),
                layers=layers,
                tooltip={
                    "html": """
                    <div style="font-family: sans-serif; padding: 5px;">
                        <b>Line:</b> {route_short_name}<br/>
                        <b>Destination:</b> {route_long_name}<br/>
                        <b>Speed:</b> {velocity} km/h<br/>
                        <b>Bearing:</b> {bearing}°
                    </div>
                """,
                    "style": {
                        "backgroundColor": "rgba(0, 50, 100, 0.8)",
                        "color": "white",
                        "fontSize": "14px",
                        "zIndex": 1000
                    }
                }))
        
        st.caption(
            "🔴 Red Pin = Your Search Location | 🚌 = Bus Location | ➤ = Direction | 🟩 Green Line = Route Path"
        )

        st.subheader("📊 Live Fleet Monitor")

        display_df = optimize_dataframe_operations(siri_df.copy())

        priority_cols = [
            'route_short_name', 'agency_name', 'velocity', 'compass_direction',
            'recorded_at_time', 'route_long_name', 'lon', 'lat', 'line_ref',
            'operator_ref', 'bearing', 'distance_from_journey_start',
            'siri_ride_stop_id'
        ]

        remaining_cols = [
            c for c in display_df.columns if c not in priority_cols
        ]
        final_order = [c for c in priority_cols if c in display_df.columns
                       ] + sorted(remaining_cols)

        st.data_editor(display_df[final_order],
                       column_config={
                           "route_short_name":
                           st.column_config.TextColumn("Line", width="small"),
                           "agency_name":
                           st.column_config.TextColumn("Operator",
                                                       width="medium"),
                           "velocity":
                           st.column_config.ProgressColumn("Speed (km/h)",
                                                           format="%d",
                                                           min_value=0,
                                                           max_value=120),
                           "compass_direction":
                           st.column_config.TextColumn("Heading",
                                                       width="small"),
                           "recorded_at_time":
                           st.column_config.DatetimeColumn("Last Signal",
                                                           format="HH:mm:ss"),
                           "route_long_name":
                           st.column_config.TextColumn("Route Description",
                                                       width="large"),
                           "lat":
                           st.column_config.NumberColumn("Latitude",
                                                         format="%.5f"),
                           "lon":
                           st.column_config.NumberColumn("Longitude",
                                                         format="%.5f"),
                       },
                       hide_index=True,
                       use_container_width=True)

    else:
        st.warning(
            "No active buses found. Try: 1. Increasing radius 2. Increasing lookback time 3. Selecting a different city."
        )

elif app_mode == "🗺️ Route Explorer":
    st.markdown("### 🗺️ Unified Route Explorer")
    st.info("Explore planned routes by Line Number OR Operator.")

    df_routes = st.session_state.get('master_routes', pd.DataFrame())

    if df_routes.empty:
        st.warning(
            "No route data found. Please go to 'Bulk Data Manager' and click 'Update Routes'."
        )
    else:
        search_mode = st.radio("Search Method",
                               ["Search by Line Number", "Browse by Operator"],
                               horizontal=True)

        filtered_routes = pd.DataFrame()

        if search_mode == "Search by Line Number":
            search_query = st.text_input("Enter Line Number (e.g. 480, 5)",
                                         placeholder="Type line number...")
            if search_query:
                filtered_routes = get_filtered_routes(search_mode, search_query)

        elif search_mode == "Browse by Operator":
            valid_agencies = sorted([
                a for a in df_routes['agency_name'].unique()
                if a and isinstance(a, str)
            ])
            selected_agency = st.selectbox("Select Operator",
                                           options=valid_agencies)

            if selected_agency:
                filtered_routes = get_filtered_routes(search_mode, None, selected_agency)

        # Display & Select Routes
        if not filtered_routes.empty:
            st.success(f"Found {len(filtered_routes)} routes.")

            if search_mode == "Browse by Operator":
                with st.expander("📋 View Route Table", expanded=True):
                    cols_to_show = [
                        'route_short_name', 'agency_name', 'route_long_name',
                        'id'
                    ]
                    valid_cols = [
                        c for c in cols_to_show if c in filtered_routes.columns
                    ]
                    st.dataframe(filtered_routes[valid_cols],
                                 use_container_width=True)

            filtered_routes['display_name'] = filtered_routes.apply(
                lambda x:
                f"Line {x['route_short_name']} | {x['route_long_name']} (ID: {x['id']})",
                axis=1)

            selected_route_keys = st.multiselect(
                "Select Routes to Map",
                options=filtered_routes['display_name'].tolist())

            if st.button("Draw Selected Routes") and selected_route_keys:
                selected_ids = filtered_routes[filtered_routes[
                    'display_name'].isin(selected_route_keys)]['id'].tolist()
                
                all_paths, all_stops = draw_route_paths(selected_ids)

                if all_stops:
                    path_layer = pdk.Layer("PathLayer",
                                           data=all_paths,
                                           get_path="path",
                                           get_width=viz_path_width,
                                           get_color="color",
                                           width_min_pixels=3)
                    stops_layer = pdk.Layer("ScatterplotLayer",
                                            data=all_stops,
                                            get_position="coordinates",
                                            get_color="color",
                                            get_radius=viz_dot_radius * 2,
                                            pickable=True)

                    start_pos = all_stops[0]['coordinates']
                    view_state = pdk.ViewState(latitude=start_pos[1],
                                               longitude=start_pos[0],
                                               zoom=11)

                    st.pydeck_chart(
                        pdk.Deck(map_style=None,
                                 initial_view_state=view_state,
                                 layers=[path_layer, stops_layer],
                                 tooltip={
                                     "html": """
                                     <div style="font-family: sans-serif; padding: 5px;">
                                         <b>Stop:</b> {name}<br/>
                                         <b>Code:</b> {code}<br/>
                                         <b>Sequence:</b> {seq}
                                     </div>
                                 """,
                                     "style": {
                                         "backgroundColor": "rgba(0, 50, 100, 0.8)",
                                         "color": "white",
                                         "fontSize": "14px",
                                         "zIndex": 1000
                                     }
                                 }))
                else:
                    st.error("No valid map data found for these routes.")

        elif search_mode == "Search by Line Number" and search_query:
            st.warning("No routes found matching your search.")

elif app_mode == "💾 Bulk Data Manager":
    st.markdown("### 💾 Bulk Data Manager")
    st.write("Manage local data cache for faster querying.")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📥 Update Routes (Today)"):
            with st.spinner("Updating route cache..."):
                start_time = time.time()
                data = fetch_all_routes_today()
                if data:
                    path = get_local_file_path("routes.json")
                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    elapsed = time.time() - start_time
                    st.success(f"Fetched {len(data)} routes in {elapsed:.2f}s")
                    if 'master_routes' in st.session_state:
                        del st.session_state['master_routes']

    with col2:
        if st.button("📥 Update Stops (Global)"):
            with st.spinner("Updating stop cache..."):
                start_time = time.time()
                data = fetch_all_stops_today()
                if data:
                    path = get_local_file_path("stops.json")
                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    elapsed = time.time() - start_time
                    st.success(f"Fetched {len(data)} stops in {elapsed:.2f}s")

    with col3:
        if st.button("📥 Update Agencies"):
            with st.spinner("Updating agency cache..."):
                start_time = time.time()
                data = fetch_api("gtfs_agencies/list", {"limit": -1})
                if data:
                    path = get_local_file_path("agencies.json")
                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    elapsed = time.time() - start_time
                    st.success(
                        f"Fetched {len(data)} agencies in {elapsed:.2f}s")

    # Check status
    st.markdown("---")
    st.subheader("Current Cache Status")

    for fname in ["routes.json", "stops.json", "agencies.json"]:
        fpath = get_local_file_path(fname)
        if os.path.exists(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            st.info(f"✅ {fname}: {size_mb:.2f} MB")
        else:
            st.warning(f"❌ {fname}: Not cached")

elif app_mode == "🔍 API Data Explorer":
    st.markdown("### 🔍 Stride API Explorer")

    tab1, tab2, tab3 = st.tabs(
        ["🏢 Operators", "🚌 Route Catalog", "🛠️ Raw Query"])

    with tab1:
        agencies, _, _ = get_or_fetch_agencies()
        if agencies:
            df = pd.DataFrame(agencies)
            desired_cols = ['agency_name', 'operator_ref', 'id']
            existing_cols = [c for c in desired_cols if c in df.columns]
            st.dataframe(df[existing_cols], use_container_width=True)
        else:
            st.warning("No agencies found.")

    with tab2:
        routes, _, _ = get_or_fetch_routes()
        if routes:
            df_routes = pd.DataFrame(routes)
            search = st.text_input("Search Routes (Local Cache)",
                                   placeholder="Line 480")
            if search:
                if search.isdigit():
                    res = df_routes[df_routes['route_short_name'] == search]
                else:
                    res = df_routes[df_routes['route_long_name'].str.contains(
                        search, na=False)]
                st.dataframe(res[[
                    'route_short_name', 'route_long_name', 'line_ref', 'id'
                ]],
                             use_container_width=True)
            else:
                st.dataframe(df_routes.head(100), use_container_width=True)

    with tab3:
        col_ep, col_param = st.columns([1, 2])
        endpoint = col_ep.selectbox("Endpoint", [
            "gtfs_agencies/list", "gtfs_routes/list", "gtfs_stops/list",
            "siri_vehicle_locations/list"
        ])
        params_txt = col_param.text_area("Parameters (JSON)", '{"limit": 5}')
        if st.button("Run Raw Query"):
            try:
                res = fetch_api(endpoint, json.loads(params_txt))
                st.json(res)
            except Exception as e:
                st.error(e)
