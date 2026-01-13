import pandas as pd
import urbanaccess as ua
import streamlit as st
import matplotlib.pyplot as plt
import requests
from datetime import datetime
import pytz

# --- CONFIGURATION ---
DATE_TARGET = '2026-01-13'
TIME_WINDOW_START = '06:00:00' 
TIME_WINDOW_END = '12:00:00'
BBOX = (34.76, 32.05, 34.82, 32.10)  # Tel Aviv Center: (min_lon, min_lat, max_lon, max_lat)

# --- 1. DATA FETCHING (STRIDE API) ---
@st.cache_data
def get_stride_data(endpoint, params):
    """Fetches data from Stride API with proper error handling."""
    base_url = f"https://open-bus-stride-api.hasadna.org.il/{endpoint}"
    all_records = []
    offset = 0
    BATCH_SIZE = 1000
    
    status_text = st.empty()
    status_text.text(f"📡 {endpoint}: Initializing...")
    
    while True:
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
            
            if len(data) < BATCH_SIZE:
                break
                
            offset += BATCH_SIZE
            
        except Exception as e:
            st.error(f"Request failed for {endpoint}: {e}")
            break
            
    status_text.empty()
    return pd.DataFrame(all_records)

def fetch_network_data(date, bbox):
    """Fetch Stride data with proper filtering."""
    st.info(f"📡 Fetching Stride Data for {date}...")
    
    # Use Israel timezone
    israel_tz = pytz.timezone('Asia/Jerusalem')
    
    # A. STOPS - First get stops within bounding box
    stops = get_stride_data("gtfs_stops/list", {
        "date_from": date,
        "date_to": date,
        "limit": 50000
    })
    
    # Filter stops by bounding box after fetching (since API doesn't support bbox directly)
    if not stops.empty:
        stops = stops[
            (stops['lon'] >= bbox[0]) & (stops['lon'] <= bbox[2]) &
            (stops['lat'] >= bbox[1]) & (stops['lat'] <= bbox[3])
        ]
    
    # B. ROUTES - Get routes for the date
    routes = get_stride_data("gtfs_routes/list", {
        "date_from": date, 
        "date_to": date, 
        "limit": 10000
    })
    
    # C. RIDES/TRIPS - Get rides within time window
    trips = get_stride_data("gtfs_rides/list", {
        "start_time_from": f"{date}T{TIME_WINDOW_START}+02:00",
        "start_time_to": f"{date}T{TIME_WINDOW_END}+02:00",
        "limit": 50000
    })
    
    # D. STOP TIMES - Get stop times for our filtered stops and trips
    if not stops.empty and not trips.empty:
        # Get stop IDs and trip IDs
        stop_ids = stops['id'].astype(str).tolist()
        trip_ids = trips['id'].astype(str).tolist()
        
        # Need to fetch in batches due to API limits
        stop_times_batches = []
        for i in range(0, len(trip_ids), 50):  # Process 50 trips at a time
            batch_ids = trip_ids[i:i+50]
            batch = get_stride_data("gtfs_ride_stops/list", {
                "gtfs_ride_ids": ",".join(batch_ids),
                "gtfs_stop_ids": ",".join(stop_ids[:100]),  # Limit stops
                "limit": 1000
            })
            stop_times_batches.append(batch)
        
        stop_times = pd.concat(stop_times_batches, ignore_index=True) if stop_times_batches else pd.DataFrame()
    else:
        stop_times = pd.DataFrame()
    
    return stops, routes, trips, stop_times

# --- 2. DATA CLEANING & TRANSFORMATION ---
def clean_gtfs_data(stops, routes, trips, stop_times, date_str):
    """Clean and transform Stride data to UrbanAccess format."""
    st.write("🧹 Cleaning and linking data...")
    
    # Check for empty data
    if stops.empty or routes.empty or trips.empty or stop_times.empty:
        st.error("❌ One or more data tables are empty!")
        return None, None, None, None, None
    
    # --- RENAME COLUMNS TO GTFS STANDARD ---
    # Stops
    stops = stops.rename(columns={
        'id': 'stop_id',
        'lat': 'stop_lat',
        'lon': 'stop_lon',
        'name': 'stop_name',
        'code': 'stop_code'
    })
    
    # Routes
    routes = routes.rename(columns={
        'id': 'route_id',
        'route_short_name': 'route_short_name',
        'route_long_name': 'route_long_name'
    })
    
    # Trips (from gtfs_rides)
    trips = trips.rename(columns={
        'id': 'trip_id',
        'gtfs_route_id': 'route_id'
    })
    
    # Stop times
    stop_times = stop_times.rename(columns={
        'gtfs_ride_id': 'trip_id',
        'gtfs_stop_id': 'stop_id'
    })
    
    # Ensure IDs are strings
    stops['stop_id'] = stops['stop_id'].astype(str)
    routes['route_id'] = routes['route_id'].astype(str)
    trips['trip_id'] = trips['trip_id'].astype(str)
    trips['route_id'] = trips['route_id'].astype(str)
    stop_times['trip_id'] = stop_times['trip_id'].astype(str)
    stop_times['stop_id'] = stop_times['stop_id'].astype(str)
    
    # --- TIME CONVERSION ---
    def parse_time(time_str):
        """Parse time string to seconds since midnight."""
        if pd.isna(time_str):
            return None
        
        try:
            # Handle ISO format: "2026-01-13T08:00:00+02:00"
            if 'T' in str(time_str):
                dt = pd.to_datetime(time_str)
                return dt.hour * 3600 + dt.minute * 60 + dt.second
            # Handle HH:MM:SS format
            else:
                parts = str(time_str).split(':')
                if len(parts) >= 3:
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        except:
            pass
        return None
    
    # Convert times
    stop_times['arrival_time_sec'] = stop_times['arrival_time'].apply(parse_time)
    stop_times['departure_time_sec'] = stop_times['departure_time'].apply(parse_time)
    
    # Remove rows with invalid times
    stop_times = stop_times.dropna(subset=['arrival_time_sec', 'departure_time_sec'])
    
    # Format times as HH:MM:SS for display
    def sec_to_hhmmss(seconds):
        if pd.isna(seconds):
            return ""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    stop_times['arrival_time'] = stop_times['arrival_time_sec'].apply(sec_to_hhmmss)
    stop_times['departure_time'] = stop_times['departure_time_sec'].apply(sec_to_hhmmss)
    
    # --- FILTER ORPHANED DATA ---
    # Get valid IDs
    valid_stop_ids = set(stop_times['stop_id'])
    valid_trip_ids = set(stop_times['trip_id'])
    valid_route_ids = set(trips[trips['trip_id'].isin(valid_trip_ids)]['route_id'])
    
    # Filter data
    stops = stops[stops['stop_id'].isin(valid_stop_ids)].copy()
    trips = trips[trips['trip_id'].isin(valid_trip_ids)].copy()
    routes = routes[routes['route_id'].isin(valid_route_ids)].copy()
    
    # Add required columns for UrbanAccess
    stops['unique_agency_id'] = 'stride_agency'
    stops['unique_feed_id'] = 'stride_feed'
    
    routes['unique_agency_id'] = 'stride_agency'
    routes['unique_feed_id'] = 'stride_feed'
    routes['agency_id'] = 'stride_agency'
    
    trips['unique_agency_id'] = 'stride_agency'
    trips['unique_feed_id'] = 'stride_feed'
    trips['service_id'] = 'weekday_service'  # Default service
    
    stop_times['unique_agency_id'] = 'stride_agency'
    stop_times['unique_feed_id'] = 'stride_feed'
    
    # Add required columns
    if 'stop_sequence' not in stop_times.columns:
        stop_times['stop_sequence'] = range(len(stop_times))
    
    if 'pickup_type' not in stop_times.columns:
        stop_times['pickup_type'] = 0
    
    if 'drop_off_type' not in stop_times.columns:
        stop_times['drop_off_type'] = 0
    
    if 'route_type' not in routes.columns:
        routes['route_type'] = 3  # Bus
    
    # Create calendar
    calendar = pd.DataFrame({
        'service_id': ['weekday_service'],
        'monday': [1], 'tuesday': [1], 'wednesday': [1], 'thursday': [1],
        'friday': [0], 'saturday': [0], 'sunday': [0],  # Adjust based on your date
        'start_date': [date_str.replace('-', '')],
        'end_date': [date_str.replace('-', '')],
        'unique_agency_id': ['stride_agency'],
        'unique_feed_id': ['stride_feed']
    })
    
    return stops, routes, trips, stop_times, calendar

# --- 3. VALIDATION ---
def validate_data(stops, routes, trips, stop_times):
    """Validate the data before processing."""
    st.divider()
    st.subheader("🕵️ Data Validation")
    
    st.write(f"**Stops:** {len(stops)} records")
    st.write(f"**Routes:** {len(routes)} records")
    st.write(f"**Trips:** {len(trips)} records")
    st.write(f"**Stop Times:** {len(stop_times)} records")
    
    if not stops.empty:
        st.write(f"**Stop Bounding Box:** Lat: {stops['stop_lat'].min():.4f}-{stops['stop_lat'].max():.4f}, "
                f"Lon: {stops['stop_lon'].min():.4f}-{stops['stop_lon'].max():.4f}")
    
    if not stop_times.empty:
        # Check sample
        sample_trip = stop_times.iloc[0]['trip_id']
        sample_data = stop_times[stop_times['trip_id'] == sample_trip][
            ['stop_sequence', 'stop_id', 'arrival_time', 'departure_time']
        ].head()
        st.write(f"**Sample Trip ({sample_trip}):**")
        st.dataframe(sample_data)
    
    st.divider()

# --- 4. URBAN ACCESS PROCESSING ---
def create_urbanaccess_network(stops, routes, trips, stop_times, calendar):
    """Create UrbanAccess network from GTFS data."""
    try:
        st.write("🔧 Creating UrbanAccess network...")
        
        # Create a GTFS feed object
        class GTFSFeed:
            def __init__(self):
                self.stops = stops
                self.routes = routes
                self.trips = trips
                self.stop_times = stop_times
                self.calendar = calendar
        
        # Create feed object
        feed = GTFSFeed()
        
        # Load into UrbanAccess (simplified approach)
        loaded_feeds = ua.gtfsfeeds.load(
            feeds={'stride_feed': feed},
            verbose=True,
            bbox=BBOX,
            remove_stops_outsidebbox=True,
            append_definitions=False
        )
        
        # Create network
        st.write("🔄 Creating transit network...")
        ua_net = ua.gtfs.network.create_transit_net(
            gtfsfeeds_dfs=loaded_feeds,
            day='tuesday',  # Adjust based on your date
            timerange=[TIME_WINDOW_START, TIME_WINDOW_END],
            calendar_dates_lookup=None
        )
        
        return ua_net
        
    except Exception as e:
        st.error(f"❌ UrbanAccess processing error: {e}")
        return None

# --- 5. MAIN APP ---
st.title("🚌 GTFS Network Grapher with UrbanAccess")

# Configuration sidebar
with st.sidebar:
    st.header("Configuration")
    DATE_TARGET = st.date_input("Select Date", value=datetime(2026, 1, 13))
    TIME_WINDOW_START = st.text_input("Start Time", "06:00:00")
    TIME_WINDOW_END = st.text_input("End Time", "12:00:00")
    
    st.subheader("Bounding Box")
    col1, col2 = st.columns(2)
    with col1:
        min_lon = st.number_input("Min Longitude", value=34.76, format="%.4f")
        min_lat = st.number_input("Min Latitude", value=32.05, format="%.4f")
    with col2:
        max_lon = st.number_input("Max Longitude", value=34.82, format="%.4f")
        max_lat = st.number_input("Max Latitude", value=32.10, format="%.4f")
    
    BBOX = (min_lon, min_lat, max_lon, max_lat)

# Fetch data
st.header("1. Data Fetching")
if st.button("📡 Fetch Data from Stride API"):
    raw_stops, raw_routes, raw_trips, raw_stop_times = fetch_network_data(
        str(DATE_TARGET), BBOX
    )
    
    if not raw_stops.empty:
        # Clean data
        st.header("2. Data Cleaning")
        stops, routes, trips, stop_times, calendar = clean_gtfs_data(
            raw_stops, raw_routes, raw_trips, raw_stop_times, str(DATE_TARGET)
        )
        
        if stops is not None:
            # Validate
            validate_data(stops, routes, trips, stop_times)
            
            # Show preview
            with st.expander("📊 Data Preview"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Stops Preview:**")
                    st.dataframe(stops.head())
                with col2:
                    st.write("**Stop Times Preview:**")
                    st.dataframe(stop_times.head())
            
            st.success(f"✅ Data ready: {len(stops)} stops, {len(trips)} trips")
            
            # Process with UrbanAccess
            st.header("3. Network Creation")
            ua_net = create_urbanaccess_network(stops, routes, trips, stop_times, calendar)
            
            if ua_net is not None:
                st.header("4. Network Visualization")
                
                # Plot the network
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Plot edges
                for idx, edge in ua_net.net_edges.iterrows():
                    # This is simplified - you'll need to adjust based on UrbanAccess structure
                    pass
                
                # Plot nodes (stops)
                ax.scatter(stops['stop_lon'], stops['stop_lat'], 
                          c='red', s=20, alpha=0.6, label='Stops')
                
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                ax.set_title(f"Transit Network - {DATE_TARGET}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # Alternative: Show data on map
                st.subheader("📍 Interactive Map")
                map_data = stops[['stop_lat', 'stop_lon', 'stop_name']].copy()
                st.map(map_data, use_container_width=True)
                
            else:
                st.warning("Network creation failed. Showing raw data instead.")
                
                # Fallback: Show stops on map
                st.subheader("📍 Stops Map")
                st.map(stops[['stop_lat', 'stop_lon']])
        else:
            st.error("Data cleaning failed!")
    else:
        st.warning("No data found for the specified parameters!")
else:
    st.info("Click 'Fetch Data from Stride API' to begin.")
