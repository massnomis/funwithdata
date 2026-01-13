import pandas as pd
import urbanaccess as ua
import urbanaccess.plot
import streamlit as st
import matplotlib.pyplot as plt
import requests
from urbanaccess.gtfs.gtfsfeeds_dataframe import gtfsfeeds_dfs

# --- CONFIGURATION ---
DATE_TARGET = '2026-01-13'
TIME_WINDOW_START = '06:00:00' 
TIME_WINDOW_END = '12:00:00'   # 6 hour window
BBOX = (34.76, 32.05, 34.82, 32.10) # Tel Aviv Center

# --- 1. DATA FETCHING (STRIDE API) ---
@st.cache_data
def get_stride_data(endpoint, params):
    """
    Fetches data from Stride API with automatic pagination.
    Loops until all data matching the filters is retrieved.
    """
    base_url = f"https://open-bus-stride-api.hasadna.org.il/{endpoint}/list"
    all_records = []
    offset = 0
    BATCH_SIZE = 1000  # Safe chunk size for Stride API
    
    # If a limit was passed by the user, respect it as a hard cap. 
    # Otherwise, fetch everything (set a very high safety ceiling).
    max_total_limit = params.pop('limit', 500000)
    
    status_text = st.empty()
    status_text.text(f"📡 {endpoint}: Initializing...")

    while len(all_records) < max_total_limit:
        current_params = params.copy()
        current_params['offset'] = offset
        current_params['limit'] = BATCH_SIZE 
        
        try:
            r = requests.get(base_url, params=current_params, timeout=60)
            if r.status_code != 200:
                st.error(f"API Error {endpoint}: {r.status_code} - {r.text}")
                break
                
            data = r.json()
            if not data:
                break
                
            all_records.extend(data)
            status_text.text(f"📡 {endpoint}: Fetched {len(all_records)} records...")
            
            # Stop if we got fewer records than requested (end of data)
            if len(data) < BATCH_SIZE:
                break
                
            offset += BATCH_SIZE
            
        except Exception as e:
            st.error(f"Request failed for {endpoint}: {e}")
            break
            
    status_text.empty() # Clear the loading message
    return pd.DataFrame(all_records)

def fetch_network_data(date, bbox):
    st.info(f"📡 Fetching Stride Data for {date}...")
    
    # A. STOPS (Spatial Filter)
    stops = get_stride_data("gtfs_stops", {
        "min_lon": bbox[0], "min_lat": bbox[1],
        "max_lon": bbox[2], "max_lat": bbox[3],
        "limit": 50000 # High cap, pagination handles the rest
    })
    
    # B. ROUTES (Date Filter)
    routes = get_stride_data("gtfs_routes", {
        "date_from": date, "date_to": date, 
        "limit": 10000
    })

    # C. TRIPS (Mapped from gtfs_rides)
    trips = get_stride_data("gtfs_rides", {
        "start_time_from": f"{date}T{TIME_WINDOW_START}+02:00",
        "start_time_to": f"{date}T{TIME_WINDOW_END}+02:00",
        "gtfs_route__date_from": date,
        "gtfs_route__date_to": date,
        "limit": 50000
    })

    # D. STOP TIMES (Mapped from gtfs_ride_stops)
    # This is the heavy one. Pagination is CRITICAL here.
    stop_times = get_stride_data("gtfs_ride_stops", {
        "arrival_time_from": f"{date}T{TIME_WINDOW_START}+02:00",
        "arrival_time_to": f"{date}T{TIME_WINDOW_END}+02:00",
        "limit": 500000 # Very high cap
    })

    return stops, routes, trips, stop_times

# --- 2. DATA CLEANING & TRANSFORMATION ---
def clean_gtfs_data(stops, routes, trips, stop_times, date_str):
    st.write("🧹 Cleaning and linking data...")
    
    if any(df.empty for df in [stops, routes, trips, stop_times]):
        st.error("❌ Missing data from API (One or more tables are empty).")
        return None, None, None, None, None

    # --- RENAME COLUMNS TO GTFS STANDARD ---
    stops = stops.rename(columns={'id': 'stop_id', 'lat': 'stop_lat', 'lon': 'stop_lon', 'name': 'stop_name', 'code': 'stop_code'})
    routes = routes.rename(columns={'id': 'route_id', 'short_name': 'route_short_name', 'long_name': 'route_long_name'})
    trips = trips.rename(columns={'id': 'trip_id', 'gtfs_route_id': 'route_id'})
    stop_times = stop_times.rename(columns={'gtfs_ride_id': 'trip_id', 'gtfs_stop_id': 'stop_id', 'stop_sequence': 'stop_sequence'})

    # --- STRING ID ENFORCEMENT ---
    def to_str(df, col):
        if col in df.columns: df[col] = df[col].astype(str).str.strip()
        return df

    for df in [stops, routes, trips, stop_times]:
        df = to_str(df, 'stop_id')
        df = to_str(df, 'route_id')
        df = to_str(df, 'trip_id')
    
    # --- DEDUPLICATION ---
    stops = stops.drop_duplicates(subset=['stop_id'], keep='first')
    routes = routes.drop_duplicates(subset=['route_id'], keep='first')
    trips = trips.drop_duplicates(subset=['trip_id'], keep='first')
    stop_times = stop_times.drop_duplicates(subset=['trip_id', 'stop_sequence'], keep='first')

    # --- ROBUST TIME CONVERSION ---
    # This method calculates seconds relative to the START of the target day.
    # It correctly handles trips that cross midnight (e.g., 25:00:00).
    
    base_date_dt = pd.to_datetime(date_str)
    
    def to_seconds_robust(t_str):
        if pd.isna(t_str): return None
        try:
            # Stride returns ISO: 2026-01-13T08:00:00+02:00
            dt = pd.to_datetime(t_str)
            
            # Normalize to the target date's midnight (keeping timezone info from dt)
            # This ensures 2026-01-14T00:01:00 becomes > 86400 seconds
            base_reference = dt.normalize().replace(
                year=base_date_dt.year, month=base_date_dt.month, day=base_date_dt.day
            )
            
            # Calculate total seconds difference
            diff = (dt - base_reference).total_seconds()
            
            # If the date is actually the next day, add 24 hours (86400 seconds)
            # Stride gives full datetime, so (dt - base_reference) might handle this,
            # but let's be explicit if dates differ.
            days_diff = (dt.date() - base_date_dt.date()).days
            if days_diff > 0:
                 # Adjust base reference to be the start of the query date
                 # This logic simplifies to: just subtract query_date_midnight from timestamp
                 pass 
            
            # Simplified: (Timestamp - Midnight_of_Query_Date) in seconds
            # We construct a timezone-aware midnight for the query date
            query_midnight = pd.Timestamp(date_str).tz_localize(dt.tz)
            seconds = (dt - query_midnight).total_seconds()
            
            return int(seconds)
        except Exception:
            return None

    def clean_time_str(t_str):
        if pd.isna(t_str): return None
        t_str = str(t_str)
        if 'T' in t_str: t_str = t_str.split('T')[-1]
        if '+' in t_str: t_str = t_str.split('+')[0]
        return t_str[:8] # HH:MM:SS

    stop_times['departure_time'] = stop_times['departure_time'].fillna(stop_times['arrival_time'])
    
    # Calculate seconds using the robust method
    stop_times['arrival_time_sec'] = stop_times['arrival_time'].apply(to_seconds_robust)
    stop_times['departure_time_sec'] = stop_times['departure_time'].apply(to_seconds_robust)
    
    # Clean string format for display
    stop_times['arrival_time'] = stop_times['arrival_time'].apply(clean_time_str)
    stop_times['departure_time'] = stop_times['departure_time'].apply(clean_time_str)
    
    # Drop rows where time calc failed
    stop_times = stop_times.dropna(subset=['arrival_time_sec', 'departure_time_sec'])

    # --- FILTERING ORPHANS ---
    valid_trips = set(trips['trip_id'])
    stop_times = stop_times[stop_times['trip_id'].isin(valid_trips)].copy()
    
    valid_trips_with_stops = set(stop_times['trip_id'])
    trips = trips[trips['trip_id'].isin(valid_trips_with_stops)].copy()
    
    valid_stops = set(stop_times['stop_id'])
    stops = stops[stops['stop_id'].isin(valid_stops)].copy()

    valid_routes = set(trips['route_id'])
    routes = routes[routes['route_id'].isin(valid_routes)].copy()

    # --- URBAN ACCESS METADATA ---
    for df in [stops, routes, trips, stop_times]:
        df['unique_agency_id'] = 'stride_agency'
        df['unique_feed_id'] = 'stride_feed'

    # --- CALENDAR ---
    trips['service_id'] = 'daily_service'
    calendar = pd.DataFrame({
        'service_id': ['daily_service'],
        'monday': [1], 'tuesday': [1], 'wednesday': [1], 'thursday': [1], 
        'friday': [1], 'saturday': [1], 'sunday': [1],
        'start_date': [date_str.replace('-', '')],
        'end_date': [date_str.replace('-', '')],
        'unique_agency_id': ['stride_agency'],
        'unique_feed_id': ['stride_feed']
    })

    stop_times['stop_sequence'] = pd.to_numeric(stop_times['stop_sequence'], errors='coerce').fillna(0).astype(int)
    stop_times = stop_times.sort_values(['trip_id', 'stop_sequence'])
    
    if 'route_type' not in routes.columns:
        routes['route_type'] = 3
    else:
         routes['route_type'] = pd.to_numeric(routes['route_type'], errors='coerce').fillna(3).astype(int)

    stops = stops.reset_index(drop=True)
    routes = routes.reset_index(drop=True)
    trips = trips.reset_index(drop=True)
    stop_times = stop_times.reset_index(drop=True)
    calendar = calendar.reset_index(drop=True)

    return stops, routes, trips, stop_times, calendar

# --- 3. DATA VALIDATION (VERIFY YOUR DATA) ---
def validate_data(stop_times, stops):
    st.divider()
    st.subheader("🕵️ Data Health Check")
    
    # 1. Check a sample trip
    sample_trip_id = stop_times['trip_id'].iloc[0]
    st.write(f"**Sample Trip Inspection ({sample_trip_id}):**")
    sample_data = stop_times[stop_times['trip_id'] == sample_trip_id][['stop_sequence', 'stop_id', 'arrival_time', 'arrival_time_sec']]
    
    # Calculate delta between stops to verify 'seconds'
    sample_data['delta_sec'] = sample_data['arrival_time_sec'].diff()
    st.dataframe(sample_data)
    
    # 2. Check for logic errors
    negative_travel = stop_times[stop_times['departure_time_sec'] > stop_times['arrival_time_sec']]
    if not negative_travel.empty:
        st.error(f"⚠️ Found {len(negative_travel)} rows where Departure < Arrival!")
    else:
        st.success("✅ Time logic is valid (Departure >= Arrival)")
        
    st.write(f"**Seconds Range:** Min: {stop_times['arrival_time_sec'].min()}, Max: {stop_times['arrival_time_sec'].max()}")
    st.divider()

# --- 4. MAIN EXECUTION ---
st.title("🚌 GTFS Network Grapher")

raw_stops, raw_routes, raw_trips, raw_st = fetch_network_data(DATE_TARGET, BBOX)

if raw_trips is not None and not raw_trips.empty:
    stops, routes, trips, stop_times, calendar = clean_gtfs_data(
        raw_stops, raw_routes, raw_trips, raw_st, DATE_TARGET
    )

    if stops is not None and not stops.empty:
        # RUN VALIDATION
        validate_data(stop_times, stops)
        
        st.success(f"✅ Data Ready: {len(stops)} stops, {len(trips)} trips, {len(stop_times)} stop_times.")
        
        ua_gtfs = gtfsfeeds_dfs
        ua_gtfs.stops = stops
        ua_gtfs.routes = routes
        ua_gtfs.trips = trips
        ua_gtfs.stop_times = stop_times
        ua_gtfs.calendar = calendar
        
        try:
            ua_net = ua.gtfs.network.create_transit_net(
                gtfsfeeds_dfs=ua_gtfs,
                day='tuesday', 
                timerange=['05:00:00', '13:00:00'],
                calendar_dates_lookup=None
            )
            
            st.write("🗺️ Generating Plot...")
            fig, ax = ua.plot.plot_net(
                nodes=ua_net.net_nodes,
                edges=ua_net.net_edges,
                bbox=BBOX,
                fig_height=6,
                edge_linewidth=1
            )
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"UrbanAccess Error: {e}")
            st.write("Check data integrity:", stop_times[['trip_id', 'arrival_time_sec', 'departure_time_sec']].head())
    else:
        st.warning("No valid data after cleaning.")
else:
    st.warning("No trips found in initial fetch.")
