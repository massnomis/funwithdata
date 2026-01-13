import pandas as pd
import urbanaccess as ua
import streamlit as st
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
import pytz
import warnings
warnings.filterwarnings('ignore')

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
            r = requests.get(base_url, params=current_params, timeout=120)
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
    
    # A. STOPS - Get all stops for the date
    stops = get_stride_data("gtfs_stops/list", {
        "date_from": date,
        "date_to": date,
        "limit": 50000
    })
    
    # Filter stops by bounding box
    if not stops.empty:
        stops = stops[
            (stops['lon'] >= bbox[0]) & (stops['lon'] <= bbox[2]) &
            (stops['lat'] >= bbox[1]) & (stops['lat'] <= bbox[3])
        ].copy()
    
    if stops.empty:
        st.warning("No stops found in the bounding box!")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # B. ROUTES - Get routes for the date
    routes = get_stride_data("gtfs_routes/list", {
        "date_from": date, 
        "date_to": date, 
        "limit": 10000
    })
    
    if routes.empty:
        st.warning("No routes found for the date!")
        return stops, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # C. RIDES/TRIPS - Get rides within time window
    rides = get_stride_data("gtfs_rides/list", {
        "start_time_from": f"{date}T{TIME_WINDOW_START}+02:00",
        "start_time_to": f"{date}T{TIME_WINDOW_END}+02:00",
        "limit": 50000
    })
    
    if rides.empty:
        st.warning("No rides found in the time window!")
        return stops, routes, pd.DataFrame(), pd.DataFrame()
    
    # Get ride IDs for our rides
    ride_ids = rides['id'].astype(str).unique().tolist()
    
    # D. STOP TIMES - Use time range filtering as required by API
    # Create time range for the entire date to ensure we get all data
    start_time = f"{date}T00:00:00+02:00"
    end_time = f"{date}T23:59:59+02:00"
    
    st.write(f"🔍 Fetching stop times for {len(ride_ids)} rides...")
    
    # We need to fetch stop times for our rides. Since the API requires time range,
    # we'll use the entire day and then filter by our ride IDs
    stop_times = get_stride_data("gtfs_ride_stops/list", {
        "arrival_time_from": start_time,
        "arrival_time_to": end_time,
        "limit": 100000  # Increased limit for broader time range
    })
    
    # Filter stop_times to only include our rides and stops
    if not stop_times.empty:
        # Convert IDs to strings for comparison
        stop_times['gtfs_ride_id'] = stop_times['gtfs_ride_id'].astype(str)
        stop_times['gtfs_stop_id'] = stop_times['gtfs_stop_id'].astype(str)
        
        # Get stop IDs as strings
        stop_ids = stops['id'].astype(str).unique().tolist()
        
        # Filter by ride IDs and stop IDs
        stop_times = stop_times[
            stop_times['gtfs_ride_id'].isin(ride_ids) &
            stop_times['gtfs_stop_id'].isin(stop_ids)
        ].copy()
    
    return stops, routes, rides, stop_times

# --- 2. DATA CLEANING & TRANSFORMATION ---
def clean_gtfs_data(stops, routes, rides, stop_times, date_str):
    """Clean and transform Stride data to UrbanAccess format."""
    st.write("🧹 Cleaning and linking data...")
    
    # Check for empty data
    if stops.empty or routes.empty or rides.empty or stop_times.empty:
        st.error("❌ One or more data tables are empty!")
        st.write(f"Stops: {len(stops)}, Routes: {len(routes)}, Rides: {len(rides)}, Stop Times: {len(stop_times)}")
        return None, None, None, None, None
    
    # --- RENAME COLUMNS TO GTFS STANDARD ---
    # Stops
    stops = stops.rename(columns={
        'id': 'stop_id',
        'lat': 'stop_lat',
        'lon': 'stop_lon',
        'name': 'stop_name',
        'code': 'stop_code',
        'city': 'stop_city'
    })
    
    # Ensure required columns exist
    if 'stop_city' not in stops.columns:
        stops['stop_city'] = 'Unknown'
    
    # Routes
    routes = routes.rename(columns={
        'id': 'route_id',
        'route_short_name': 'route_short_name',
        'route_long_name': 'route_long_name',
        'operator_ref': 'agency_id'
    })
    
    # Add agency_name if not present
    if 'agency_name' not in routes.columns:
        routes['agency_name'] = 'Unknown Agency'
    
    # Rides (trips in GTFS)
    rides = rides.rename(columns={
        'id': 'trip_id',
        'gtfs_route_id': 'route_id',
        'start_time': 'trip_start_time',
        'end_time': 'trip_end_time'
    })
    
    # Stop times
    stop_times = stop_times.rename(columns={
        'id': 'stop_time_id',
        'gtfs_ride_id': 'trip_id',
        'gtfs_stop_id': 'stop_id',
        'arrival_time': 'arrival_time_raw',
        'departure_time': 'departure_time_raw',
        'stop_sequence': 'stop_sequence'
    })
    
    # Ensure IDs are strings
    stops['stop_id'] = stops['stop_id'].astype(str)
    routes['route_id'] = routes['route_id'].astype(str)
    rides['trip_id'] = rides['trip_id'].astype(str)
    rides['route_id'] = rides['route_id'].astype(str)
    stop_times['trip_id'] = stop_times['trip_id'].astype(str)
    stop_times['stop_id'] = stop_times['stop_id'].astype(str)
    
    # --- TIME CONVERSION ---
    def parse_time_to_seconds(time_str):
        """Parse time string to seconds since midnight."""
        if pd.isna(time_str):
            return None
        
        try:
            # Handle ISO format: "2026-01-13T08:00:00+02:00"
            if 'T' in str(time_str):
                # Extract just the time part
                time_part = str(time_str).split('T')[1].split('+')[0]
                hours, minutes, seconds = map(int, time_part.split(':'))
                return hours * 3600 + minutes * 60 + seconds
            # Handle HH:MM:SS format
            else:
                parts = str(time_str).split(':')
                if len(parts) >= 3:
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(float(parts[2]))
        except Exception as e:
            return None
        return None
    
    # Convert times to seconds since midnight
    stop_times['arrival_time_sec'] = stop_times['arrival_time_raw'].apply(parse_time_to_seconds)
    stop_times['departure_time_sec'] = stop_times['departure_time_raw'].apply(parse_time_to_seconds)
    
    # Handle missing departure times
    stop_times['departure_time_sec'] = stop_times['departure_time_sec'].fillna(stop_times['arrival_time_sec'])
    
    # Remove rows with invalid times
    stop_times = stop_times.dropna(subset=['arrival_time_sec', 'departure_time_sec']).copy()
    
    # Format times as HH:MM:SS for GTFS
    def sec_to_hhmmss(seconds):
        if pd.isna(seconds):
            return "00:00:00"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    stop_times['arrival_time'] = stop_times['arrival_time_sec'].apply(sec_to_hhmmss)
    stop_times['departure_time'] = stop_times['departure_time_sec'].apply(sec_to_hhmmss)
    
    # Ensure stop_sequence is integer
    if 'stop_sequence' not in stop_times.columns:
        # Create stop sequence by grouping by trip_id and ordering by arrival time
        stop_times = stop_times.sort_values(['trip_id', 'arrival_time_sec']).copy()
        stop_times['stop_sequence'] = stop_times.groupby('trip_id').cumcount() + 1
    else:
        stop_times['stop_sequence'] = pd.to_numeric(stop_times['stop_sequence'], errors='coerce').fillna(0).astype(int)
    
    # --- FILTER ORPHANED DATA ---
    # Get valid IDs
    valid_stop_ids = set(stop_times['stop_id'])
    valid_trip_ids = set(stop_times['trip_id'])
    valid_route_ids = set(rides[rides['trip_id'].isin(valid_trip_ids)]['route_id'])
    
    # Filter data
    stops = stops[stops['stop_id'].isin(valid_stop_ids)].copy()
    trips = rides[rides['trip_id'].isin(valid_trip_ids)].copy()
    routes = routes[routes['route_id'].isin(valid_route_ids)].copy()
    
    # Check if we have data after filtering
    if stops.empty or trips.empty or routes.empty:
        st.error("❌ After filtering, one or more tables became empty!")
        return None, None, None, None, None
    
    # --- ADD REQUIRED COLUMNS FOR URBANACCESS ---
    # Agency table
    agency = pd.DataFrame({
        'agency_id': routes['agency_id'].unique(),
        'agency_name': 'Stride Bus Agency',
        'agency_url': 'https://open-bus.hasadna.org.il/',
        'agency_timezone': 'Asia/Jerusalem',
        'agency_lang': 'he',
        'agency_phone': ''
    })
    
    # Add required columns to stops
    stops['unique_agency_id'] = 'stride_agency'
    stops['unique_feed_id'] = 'stride_feed'
    stops['zone_id'] = '1'
    
    # Add required columns to routes
    routes['unique_agency_id'] = 'stride_agency'
    routes['unique_feed_id'] = 'stride_feed'
    if 'route_type' not in routes.columns:
        routes['route_type'] = 3  # Bus
    
    # Add required columns to trips
    trips['unique_agency_id'] = 'stride_agency'
    trips['unique_feed_id'] = 'stride_feed'
    trips['service_id'] = 'weekday_service'
    trips['trip_headsign'] = trips.get('trip_headsign', 'Unknown Destination')
    trips['direction_id'] = 0
    
    # Add required columns to stop_times
    stop_times['unique_agency_id'] = 'stride_agency'
    stop_times['unique_feed_id'] = 'stride_feed'
    stop_times['pickup_type'] = 0
    stop_times['drop_off_type'] = 0
    stop_times['shape_dist_traveled'] = 0
    
    # Create calendar
    # Determine day of week for the target date
    target_date = pd.to_datetime(date_str)
    day_of_week = target_date.dayofweek  # 0=Monday, 6=Sunday
    
    calendar = pd.DataFrame({
        'service_id': ['weekday_service'],
        'monday': [1 if day_of_week == 0 else 0],
        'tuesday': [1 if day_of_week == 1 else 0],
        'wednesday': [1 if day_of_week == 2 else 0],
        'thursday': [1 if day_of_week == 3 else 0],
        'friday': [1 if day_of_week == 4 else 0],
        'saturday': [1 if day_of_week == 5 else 0],
        'sunday': [1 if day_of_week == 6 else 0],
        'start_date': [date_str.replace('-', '')],
        'end_date': [date_str.replace('-', '')]
    })
    
    return stops, routes, trips, stop_times, calendar, agency

# --- 3. VALIDATION ---
def validate_data(stops, routes, trips, stop_times):
    """Validate the data before processing."""
    st.divider()
    st.subheader("🕵️ Data Validation")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Stops", len(stops))
    with col2:
        st.metric("Routes", len(routes))
    with col3:
        st.metric("Trips", len(trips))
    with col4:
        st.metric("Stop Times", len(stop_times))
    
    if not stops.empty:
        st.write(f"**Stop Bounding Box:** Lat: {stops['stop_lat'].min():.4f}-{stops['stop_lat'].max():.4f}, "
                f"Lon: {stops['stop_lon'].min():.4f}-{stops['stop_lon'].max():.4f}")
    
    if not stop_times.empty:
        # Check time ranges
        min_time = sec_to_hhmmss(stop_times['arrival_time_sec'].min())
        max_time = sec_to_hhmmss(stop_times['arrival_time_sec'].max())
        st.write(f"**Time Range:** {min_time} to {max_time}")
        
        # Check a sample trip
        sample_trip = stop_times.iloc[0]['trip_id']
        sample_data = stop_times[stop_times['trip_id'] == sample_trip][
            ['stop_sequence', 'stop_id', 'arrival_time', 'departure_time']
        ].head()
        st.write(f"**Sample Trip ({sample_trip}):**")
        st.dataframe(sample_data)
    
    st.divider()
    return True

def sec_to_hhmmss(seconds):
    """Helper function to convert seconds to HH:MM:SS format."""
    if pd.isna(seconds):
        return "00:00:00"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# --- 4. SIMPLE VISUALIZATION (Fallback) ---
def create_simple_visualization(stops, stop_times):
    """Create a simple visualization of the transit network."""
    try:
        # Create a simple graph
        import networkx as nx
        
        st.subheader("🗺️ Network Visualization")
        
        # Create graph
        G = nx.Graph()
        
        # Add stops as nodes with positions
        for _, stop in stops.iterrows():
            G.add_node(stop['stop_id'], 
                      pos=(stop['stop_lon'], stop['stop_lat']),
                      name=stop.get('stop_name', f"Stop {stop['stop_id']}"),
                      lat=stop['stop_lat'],
                      lon=stop['stop_lon'])
        
        # Add edges between consecutive stops in trips
        edges_added = set()
        edge_weights = {}
        
        # Group stop_times by trip and sort by sequence
        for trip_id, trip_data in stop_times.groupby('trip_id'):
            trip_data = trip_data.sort_values('stop_sequence')
            stops_in_trip = trip_data['stop_id'].tolist()
            
            # Add edges between consecutive stops
            for i in range(len(stops_in_trip) - 1):
                edge = (stops_in_trip[i], stops_in_trip[i + 1])
                if edge not in edges_added:
                    edges_added.add(edge)
                    edge_weights[edge] = 1
                else:
                    edge_weights[edge] += 1
        
        # Add weighted edges to graph
        for (u, v), weight in edge_weights.items():
            G.add_edge(u, v, weight=weight)
        
        # Plot the graph
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Get positions
        pos = {node: (G.nodes[node]['lon'], G.nodes[node]['lat']) for node in G.nodes()}
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=50, node_color='red', alpha=0.7)
        
        # Draw edges with weights
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        # Normalize weights for line width
        if weights:
            min_weight = min(weights)
            max_weight = max(weights)
            if max_weight > min_weight:
                widths = [2 + 8 * (w - min_weight) / (max_weight - min_weight) for w in weights]
            else:
                widths = [4] * len(weights)
        else:
            widths = [2] * len(edges)
        
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', width=widths, alpha=0.5)
        
        # Add labels for major stops (hubs)
        hubs = [node for node in G.nodes() if G.degree(node) > 2]
        hub_labels = {node: G.nodes[node]['name'][:15] for node in hubs}
        nx.draw_networkx_labels(G, pos, hub_labels, font_size=8, ax=ax)
        
        # Set plot properties
        ax.set_title(f"Transit Network - {DATE_TARGET}", fontsize=16)
        ax.set_xlabel("Longitude", fontsize=12)
        ax.set_ylabel("Latitude", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.scatter([], [], c='red', s=50, label='Bus Stops')
        ax.plot([], [], c='gray', linewidth=4, label='Bus Routes')
        ax.legend(loc='upper right')
        
        st.pyplot(fig)
        
        # Show statistics
        st.write(f"**Network Statistics:**")
        st.write(f"- Number of stops: {G.number_of_nodes()}")
        st.write(f"- Number of route segments: {G.number_of_edges()}")
        st.write(f"- Network density: {nx.density(G):.4f}")
        
        return True
        
    except Exception as e:
        st.error(f"❌ Visualization error: {e}")
        return False

# --- 5. MAIN APP ---
st.title("🚌 GTFS Network Grapher with Stride API")

# Configuration sidebar
with st.sidebar:
    st.header("Configuration")
    DATE_TARGET = st.date_input("Select Date", value=datetime(2026, 1, 13))
    TIME_WINDOW_START = st.text_input("Start Time", "06:00:00")
    TIME_WINDOW_END = st.text_input("End Time", "12:00:00")
    
    st.subheader("Bounding Box (Tel Aviv Center)")
    col1, col2 = st.columns(2)
    with col1:
        min_lon = st.number_input("Min Longitude", value=34.76, format="%.4f")
        min_lat = st.number_input("Min Latitude", value=32.05, format="%.4f")
    with col2:
        max_lon = st.number_input("Max Longitude", value=34.82, format="%.4f")
        max_lat = st.number_input("Max Latitude", value=32.10, format="%.4f")
    
    BBOX = (min_lon, min_lat, max_lon, max_lat)
    
    st.divider()
    st.caption("Using Stride Open Bus API")

# Main execution
if st.button("🚀 Fetch and Visualize Data"):
    DATE_TARGET = str(DATE_TARGET)
    
    with st.spinner("Fetching data from Stride API..."):
        raw_stops, raw_routes, raw_rides, raw_stop_times = fetch_network_data(DATE_TARGET, BBOX)
    
    if not raw_stops.empty:
        # Clean data
        with st.spinner("Cleaning and transforming data..."):
            result = clean_gtfs_data(raw_stops, raw_routes, raw_rides, raw_stop_times, DATE_TARGET)
            
            if result[0] is not None:
                stops, routes, trips, stop_times, calendar, agency = result
                
                # Validate
                if validate_data(stops, routes, trips, stop_times):
                    # Show data preview
                    with st.expander("📊 Data Preview"):
                        tab1, tab2, tab3, tab4 = st.tabs(["Stops", "Routes", "Trips", "Stop Times"])
                        with tab1:
                            st.dataframe(stops[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']].head())
                        with tab2:
                            st.dataframe(routes[['route_id', 'route_short_name', 'route_long_name', 'agency_id']].head())
                        with tab3:
                            st.dataframe(trips[['trip_id', 'route_id', 'service_id']].head())
                        with tab4:
                            st.dataframe(stop_times[['trip_id', 'stop_id', 'stop_sequence', 'arrival_time', 'departure_time']].head())
                    
                    # Try UrbanAccess if available
                    try:
                        st.subheader("🔧 Creating UrbanAccess Network")
                        
                        # Create a dictionary of dataframes for UrbanAccess
                        gtfs_data = {
                            'stops': stops,
                            'routes': routes,
                            'trips': trips,
                            'stop_times': stop_times,
                            'calendar': calendar,
                            'agency': agency
                        }
                        
                        # Load into UrbanAccess
                        ua.gtfsfeeds.load_feed(gtfs_data, feed_name='stride_feed')
                        
                        # Create network
                        ua_net = ua.gtfs.network.create_transit_net(
                            gtfsfeeds=gtfs_data,
                            day=DATE_TARGET,
                            timerange=[TIME_WINDOW_START, TIME_WINDOW_END]
                        )
                        
                        # Plot with UrbanAccess
                        fig, ax = plt.subplots(figsize=(12, 10))
                        ua.plot.plot_net(
                            nodes=ua_net.network_nodes,
                            edges=ua_net.network_edges,
                            bbox=BBOX,
                            fig_height=10,
                            margin=0.02,
                            edge_linewidth=0.5,
                            edge_alpha=0.6,
                            node_color='red',
                            node_size=5,
                            node_alpha=0.8,
                            node_edgecolor='none'
                        )
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.warning(f"UrbanAccess failed: {e}")
                        st.info("Falling back to simple visualization...")
                        
                        # Use simple visualization
                        create_simple_visualization(stops, stop_times)
                    
                    # Show map view
                    st.subheader("📍 Interactive Map View")
                    map_data = stops[['stop_lat', 'stop_lon', 'stop_name']].copy()
                    st.map(map_data, use_container_width=True)
                    
                else:
                    st.error("Data validation failed!")
            else:
                st.error("Data cleaning failed!")
    else:
        st.error("No data found! Try adjusting the date or bounding box.")
else:
    st.info("👈 Configure the parameters in the sidebar and click 'Fetch and Visualize Data' to begin.")
    st.write("This app fetches bus transit data from the Stride Open Bus API and visualizes the network.")
    
    # Show example of what will be fetched
    st.write("**Example Query:**")
    st.code(f"""
    Date: {DATE_TARGET}
    Time Window: {TIME_WINDOW_START} to {TIME_WINDOW_END}
    Bounding Box: {BBOX}
    """)
