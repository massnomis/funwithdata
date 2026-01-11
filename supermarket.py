import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import kagglehub
import os
import gc 
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# --- CONFIG & AUTH ---
# Credentials provided by user
os.environ["KAGGLE_USERNAME"] = "massnomis" 
os.environ["KAGGLE_KEY"] = "KGAT_42bfd4b0352d16e74bca88ea7af74199"

st.set_page_config(page_title="Israel Grocery Analytics", layout="wide", page_icon="🛒")

# --- UTILS & DATA LOADING ---

def get_file_path(all_files, chain_key, file_type):
    """
    Helper to find the best matching file for a chain and type (Price/Store).
    Prioritizes 'Full' files over others.
    """
    candidates = [f for f in all_files if chain_key in f.lower() and file_type in f.lower() and 'promo' not in f.lower()]
    
    if not candidates:
        return None
        
    # If multiple files exist, prefer one with 'full' in the name (usually the complete snapshot)
    full_files = [f for f in candidates if 'full' in f.lower()]
    if full_files:
        return max(full_files, key=os.path.getsize) # Return largest 'full' file
    
    # Otherwise return the largest candidate (likely the main data file)
    return max(candidates, key=os.path.getsize)

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    """Downloads and merges data from Kaggle with memory optimization."""
    
    # 1. Download
    try:
        path = kagglehub.dataset_download("erlichsefi/israeli-supermarkets-2024")
    except Exception as e:
        st.error(f"Kaggle Download Error: {e}")
        return pd.DataFrame()

    # 2. Map Files
    all_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            all_files.append(os.path.join(root, file))
            
    # Debug: Uncomment to see files if issues persist
    # st.write(f"Found {len(all_files)} files in {path}")

    chains = {
        "osher_ad": "Osher Ad",
        "rami_levy": "Rami Levy", 
        "victory": "Victory",
        "shufersal": "Shufersal",
        "yohananof": "Yohananof"
    }
    
    master_df = []
    
    # 3. Process Chains
    status_text = st.empty()
    
    for key, pretty_name in chains.items():
        status_text.text(f"Processing {pretty_name}...")
        
        try:
            # Find files
            p_file = get_file_path(all_files, key, "price")
            s_file = get_file_path(all_files, key, "store")
            
            if not p_file or not s_file:
                continue
                
            # --- MEMORY OPTIMIZATION ---
            # We ONLY read specific columns to avoid crashing memory with 1.7GB+ data
            # Column names vary, so we read header first to identify correct cols
            p_header = pd.read_csv(p_file, nrows=0).columns.tolist()
            s_header = pd.read_csv(s_file, nrows=0).columns.tolist()
            
            # Identify columns dynamically
            def find_col(cols, keywords):
                for c in cols:
                    if any(k in c.lower() for k in keywords): return c
                return None

            # Price File Cols
            p_item_name = find_col(p_header, ['itemname', 'item_name'])
            p_item_price = find_col(p_header, ['itemprice', 'item_price'])
            p_store_id = find_col(p_header, ['storeid', 'store_id'])
            p_item_code = find_col(p_header, ['itemcode', 'item_code'])
            
            # Store File Cols - Expanded to include Address, City, Zip
            s_store_id = find_col(s_header, ['storeid', 'store_id'])
            s_store_name = find_col(s_header, ['storename', 'store_name'])
            s_address = find_col(s_header, ['address'])
            s_city = find_col(s_header, ['city'])
            s_zip = find_col(s_header, ['zipcode', 'zip_code', 'zip'])
            
            # Coords might not exist in all files
            s_lat = find_col(s_header, ['latitude', 'lat'])
            s_lon = find_col(s_header, ['longitude', 'lon', 'lng'])

            if not (p_item_name and p_item_price and p_store_id and s_store_id):
                continue

            # Load Data (Usecols is crucial here)
            p_cols = [p_item_name, p_item_price, p_store_id]
            if p_item_code: p_cols.append(p_item_code)
            
            # Build store columns list dynamically
            s_cols = [s_store_id]
            if s_store_name: s_cols.append(s_store_name)
            if s_address: s_cols.append(s_address)
            if s_city: s_cols.append(s_city)
            if s_zip: s_cols.append(s_zip)
            if s_lat: s_cols.append(s_lat)
            if s_lon: s_cols.append(s_lon)
            
            pdf = pd.read_csv(p_file, usecols=p_cols, dtype=str, on_bad_lines='skip')
            sdf = pd.read_csv(s_file, usecols=s_cols, dtype=str, on_bad_lines='skip')
            
            # Standardize names
            pdf = pdf.rename(columns={
                p_item_name: 'ItemName', 
                p_item_price: 'ItemPrice', 
                p_store_id: 'StoreId',
                p_item_code: 'ItemCode'
            })
            
            # Rename store columns dynamically
            store_rename_map = {s_store_id: 'StoreId'}
            if s_store_name: store_rename_map[s_store_name] = 'StoreName'
            if s_address: store_rename_map[s_address] = 'Address'
            if s_city: store_rename_map[s_city] = 'City'
            if s_zip: store_rename_map[s_zip] = 'ZipCode'
            if s_lat: store_rename_map[s_lat] = 'Latitude'
            if s_lon: store_rename_map[s_lon] = 'Longitude'
            
            sdf = sdf.rename(columns=store_rename_map)
            
            # Convert Price to float
            pdf['ItemPrice'] = pd.to_numeric(pdf['ItemPrice'], errors='coerce')
            pdf = pdf.dropna(subset=['ItemPrice'])
            
            # Merge
            merged = pd.merge(pdf, sdf, on='StoreId', how='inner')
            merged['ChainName'] = pretty_name
            
            master_df.append(merged)
            
            # Force garbage collection
            del pdf, sdf
            gc.collect()
            
        except Exception as e:
            print(f"Error loading {pretty_name}: {e}")
            continue

    status_text.empty()
    
    if not master_df:
        return pd.DataFrame()
        
    final_df = pd.concat(master_df, ignore_index=True)
    
    # Final cleanup
    if 'Latitude' in final_df.columns:
        final_df['Latitude'] = pd.to_numeric(final_df['Latitude'], errors='coerce')
        final_df['Longitude'] = pd.to_numeric(final_df['Longitude'], errors='coerce')
        
    return final_df

@st.cache_data(show_spinner=False)
def get_osm_coordinates(address_str):
    """
    Uses OpenStreetMap (Nominatim) to find coordinates for an address string.
    Cached to prevent hitting rate limits on repeated lookups.
    """
    try:
        geolocator = Nominatim(user_agent="israel_grocery_analytics_app")
        # Enforce rate limiting to be polite to OSM (1 second delay)
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        location = geocode(address_str)
        if location:
            return location.latitude, location.longitude
    except Exception:
        return None, None
    return None, None

# --- UI LAYOUT ---

st.title("🇮🇱 Supermarket Analytics Suite")
st.markdown("Analyze prices across Israel's major chains: **Rami Levy, Victory, Osher Ad, Shufersal, and Yohananof**.")

with st.spinner("Processing large dataset (this may take 30-60s)..."):
    df = load_data()

if df.empty:
    st.error("No data loaded. Check the debug logs or file structure.")
    st.stop()

# Metric Cards
c1, c2, c3 = st.columns(3)
c1.metric("Total Products Indexed", f"{len(df):,}")
c2.metric("Chains Tracked", f"{df['ChainName'].nunique()}")
c3.metric("Stores Tracked", f"{df['StoreName'].nunique()}")

# --- RAW DATA VIEWER ---
with st.expander("📊 View Raw Data (Validation)"):
    st.write("First 100 rows of the processed dataset:")
    st.dataframe(df.head(100), use_container_width=True)
    st.caption(f"Total rows in memory: {len(df):,}")

# --- TABS FOR ANALYSIS ---
tab1, tab2, tab3 = st.tabs(["🔎 Product Search", "🧺 Basket Comparator", "🗺️ Geospatial Analysis"])

# --- TAB 1: SINGLE PRODUCT SEARCH ---
with tab1:
    st.subheader("Price Check & Distribution")
    
    col_search, col_filter = st.columns([3, 1])
    with col_search:
        search_term = st.text_input("Search Product (Hebrew):", "חלב 3%", help="Try terms like: חלב, קפה, פסטה, קולה")
    with col_filter:
        selected_chains = st.multiselect("Filter Chains", df['ChainName'].unique(), default=df['ChainName'].unique())

    if search_term:
        # Filter Data
        mask = (df['ItemName'].str.contains(search_term, na=False)) & (df['ChainName'].isin(selected_chains))
        results = df[mask].copy()
        
        if not results.empty:
            # 1. Stats Table
            avg_prices = results.groupby('ChainName')['ItemPrice'].agg(['mean', 'min', 'max', 'count']).sort_values('mean')
            st.dataframe(avg_prices.style.format("{:.2f} ₪"), use_container_width=True)
            
            # 2. Visuals
            row1_col1, row1_col2 = st.columns(2)
            
            with row1_col1:
                # Bar Chart: Average Price
                fig_bar = px.bar(
                    avg_prices.reset_index(), 
                    x='ChainName', y='mean', 
                    color='mean',
                    color_continuous_scale='RdYlGn_r',
                    title=f"Average Price for '{search_term}'",
                    labels={'mean': 'Price (₪)', 'ChainName': 'Chain'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with row1_col2:
                # Box/Violin Plot: Price Consistency
                fig_box = px.box(
                    results, 
                    x='ChainName', y='ItemPrice', 
                    color='ChainName',
                    title="Price Consistency (Spread)",
                    points="outliers"
                )
                st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.warning("No products found. Try a simpler search term.")

# --- TAB 2: BASKET COMPARATOR ---
with tab2:
    st.subheader("🛒 Basket Cost Analysis")
    st.markdown("Select generic items to simulate a shopping trip and see who wins.")
    
    common_items = {
        "Milk (חלב)": "חלב",
        "Eggs (ביצים)": "ביצים",
        "White Bread (לחם)": "לחם",
        "Rice (אורז)": "אורז",
        "Pasta (פסטה)": "פסטה",
        "Cottage Cheese (קוטג')": "קוטג",
        "Coffee (קפה)": "קפה נמס"
    }
    
    selected_basket = st.multiselect("Build your Basket:", list(common_items.keys()), default=["Milk (חלב)", "Eggs (ביצים)", "White Bread (לחם)"])
    
    if selected_basket:
        basket_data = []
        
        for item_label in selected_basket:
            search_key = common_items[item_label]
            subset = df[df['ItemName'].str.contains(search_key, na=False)]
            if not subset.empty:
                chain_avgs = subset.groupby('ChainName')['ItemPrice'].mean().reset_index()
                chain_avgs['Item'] = item_label
                basket_data.append(chain_avgs)
        
        if basket_data:
            basket_df = pd.concat(basket_data)
            
            # Calculate Total Basket Cost per Chain
            total_cost = basket_df.groupby('ChainName')['ItemPrice'].sum().reset_index().sort_values('ItemPrice')
            
            # Visualization
            col_res1, col_res2 = st.columns([2, 1])
            
            with col_res1:
                fig_basket = px.bar(
                    total_cost, 
                    x='ChainName', y='ItemPrice', 
                    text='ItemPrice',
                    color='ItemPrice',
                    color_continuous_scale='RdYlGn_r',
                    title="Total Basket Cost by Chain",
                    labels={'ItemPrice': 'Total Cost (₪)'}
                )
                fig_basket.update_traces(texttemplate='%{text:.2f} ₪', textposition='outside')
                st.plotly_chart(fig_basket, use_container_width=True)
                
            with col_res2:
                st.write("### 🏆 Ranking")
                for i, row in total_cost.iterrows():
                    st.write(f"**{row['ChainName']}**: {row['ItemPrice']:.2f} ₪")
                
                cheapest_chain = total_cost.iloc[0]['ChainName']
                diff = total_cost.iloc[-1]['ItemPrice'] - total_cost.iloc[0]['ItemPrice']
                st.success(f"**{cheapest_chain}** is the cheapest option! You save {diff:.2f} ₪.")

# --- TAB 3: GEOSPATIAL ---
with tab3:
    st.subheader("📍 Store Geospatial Analysis")
    
    view_mode = st.radio("Select View Mode:", ["Single Product (Price Comparison)", "All Stores (Network Overview)"], horizontal=True)

    if view_mode == "Single Product (Price Comparison)":
        # 1. Identify Common Items (Intersection of Chains)
        if 'ItemCode' not in df.columns:
            st.error("ItemCode column missing from dataset. Cannot perform exact code matching.")
        else:
            # We only want items that appear in at least 2 distinct chains for valid comparison
            code_stats = df.groupby('ItemCode').agg(
                ChainCount=('ChainName', 'nunique'),
                ExampleName=('ItemName', 'first')
            )
            
            # Filter: Must be in at least 2 chains
            common_codes_df = code_stats[code_stats['ChainCount'] >= 2].sort_values('ChainCount', ascending=False)
            
            if common_codes_df.empty:
                st.warning("No items found that share the exact same Item Code across different chains.")
            else:
                common_codes_df['Label'] = (
                    common_codes_df['ExampleName'].astype(str) + 
                    " (" + common_codes_df.index.astype(str) + ") - " + 
                    common_codes_df['ChainCount'].astype(str) + " Chains"
                )
                
                selected_label = st.selectbox(
                    f"Select a Common Product ({len(common_codes_df):,} items available):", 
                    common_codes_df['Label'].tolist()
                )
                
                selected_code = common_codes_df[common_codes_df['Label'] == selected_label].index[0]
                map_subset = df[df['ItemCode'] == selected_code].copy()
                
                if not map_subset.empty:
                    # Group by store identifiers
                    group_cols = ['StoreName', 'ChainName']
                    possible_ids = ['City', 'ZipCode', 'Address']
                    for col in possible_ids:
                        if col in map_subset.columns:
                            group_cols.append(col)
                    
                    agg_dict = {'ItemPrice': 'mean'}
                    if 'Latitude' in map_subset.columns: agg_dict['Latitude'] = 'first'
                    if 'Longitude' in map_subset.columns: agg_dict['Longitude'] = 'first'
                    
                    store_view = map_subset.groupby(group_cols, as_index=False, dropna=False).agg(agg_dict)
                    
                    # --- REPAIR & MAP LOGIC (Shared) ---
                    # We define this logic inline for this block
                    if 'Latitude' in store_view.columns:
                        missing_mask = store_view['Latitude'].isna() | store_view['Longitude'].isna()
                        missing_count = missing_mask.sum()
                        
                        st.info(f"Stores for {selected_label}: {len(store_view)} | Missing Coords: {missing_count}")
                        
                        force_all = st.checkbox("Force update ALL displayed stores", value=False, key="force_single")
                        
                        if force_all:
                            to_process = store_view
                            btn_label = f"📍 Force Update All {len(store_view)} Stores (Slow)"
                        else:
                            to_process = store_view[missing_mask]
                            btn_label = "📍 Repair missing coordinates via OSM (Slow)"

                        if not to_process.empty:
                            if st.button(btn_label, key="btn_single"):
                                progress_bar = st.progress(0)
                                log_container = st.expander("API Transaction Log", expanded=True)
                                total_rows = len(to_process)
                                
                                for i, (index, row) in enumerate(to_process.iterrows()):
                                    if pd.notna(row.get('StoreName')):
                                        parts = [str(row['StoreName'])]
                                        if pd.notna(row.get('City')): parts.append(str(row['City']))
                                        parts.append("Israel")
                                        query_str = ", ".join(parts)
                                        lat, lon = get_osm_coordinates(query_str)
                                        
                                        if lat and lon:
                                            store_view.at[index, 'Latitude'] = lat
                                            store_view.at[index, 'Longitude'] = lon
                                            log_container.success(f"✅ Found: '{query_str}' -> ({lat:.4f}, {lon:.4f})")
                                        else:
                                            log_container.error(f"❌ Failed: '{query_str}' - No match found")
                                    progress_bar.progress((i + 1) / total_rows)
                                st.success("Repair complete.")
                    
                    if 'Latitude' in store_view.columns:
                        # --- NEW: SHOW DATAFRAME BEFORE DROP ---
                        st.write("### 📋 Store Data Overview (Before Visualization)")
                        st.dataframe(store_view, use_container_width=True)
                        # ----------------------------------------
                        store_view = store_view.dropna(subset=['Latitude', 'Longitude'])
                    
                    if not store_view.empty:
                        fig_map = px.scatter_mapbox(
                            store_view, 
                            lat="Latitude", lon="Longitude",
                            color="ItemPrice",
                            size="ItemPrice",
                            hover_name="StoreName",
                            hover_data={"ChainName": True, "ItemPrice": ":.2f"},
                            color_continuous_scale="Jet",
                            zoom=7, 
                            mapbox_style="carto-positron",
                            title=f"Price Heatmap for '{selected_label}'"
                        )
                        st.plotly_chart(fig_map, use_container_width=True)
                    else:
                        st.warning("No valid coordinates available.")

    else:
        # --- ALL STORES MODE ---
        st.markdown("### 🏬 Full Network Overview")
        
        # Prepare unique store list from the full dataframe
        group_cols = ['StoreName', 'ChainName']
        possible_ids = ['City', 'ZipCode', 'Address']
        for col in possible_ids:
            if col in df.columns:
                group_cols.append(col)
        
        agg_dict = {}
        if 'Latitude' in df.columns: agg_dict['Latitude'] = 'first'
        if 'Longitude' in df.columns: agg_dict['Longitude'] = 'first'
        
        # We just want unique stores, price doesn't matter here
        all_stores = df.groupby(group_cols, as_index=False, dropna=False).agg(agg_dict)
        
        if 'Latitude' in all_stores.columns:
            missing_mask = all_stores['Latitude'].isna() | all_stores['Longitude'].isna()
            missing_count = missing_mask.sum()
            
            st.write(f"**Total Stores found:** {len(all_stores)}")
            st.write(f"**Stores missing coordinates:** {missing_count}")
            
            force_all = st.checkbox("Force update ALL displayed stores", value=False, key="force_all")
            
            if force_all:
                to_process = all_stores
                btn_label = f"📍 Force Update All {len(all_stores)} Stores (Very Slow)"
            else:
                to_process = all_stores[missing_mask]
                btn_label = "📍 Repair missing coordinates via OSM (Slow)"
            
            if not to_process.empty:
                if st.button(btn_label, key="btn_all"):
                    progress_bar = st.progress(0)
                    log_container = st.expander("API Transaction Log", expanded=True)
                    total_rows = len(to_process)
                    
                    for i, (index, row) in enumerate(to_process.iterrows()):
                        if pd.notna(row.get('StoreName')):
                            parts = [str(row['StoreName'])]
                            if pd.notna(row.get('City')): parts.append(str(row['City']))
                            parts.append("Israel")
                            query_str = ", ".join(parts)
                            lat, lon = get_osm_coordinates(query_str)
                            
                            if lat and lon:
                                all_stores.at[index, 'Latitude'] = lat
                                all_stores.at[index, 'Longitude'] = lon
                                log_container.success(f"✅ Found: '{query_str}' -> ({lat:.4f}, {lon:.4f})")
                            else:
                                log_container.error(f"❌ Failed: '{query_str}' - No match found")
                        progress_bar.progress((i + 1) / total_rows)
                    st.success("Repair complete.")

            # --- NEW: SHOW DATAFRAME BEFORE DROP ---
            st.write("### 📋 Store Data Overview (Before Visualization)")
            st.dataframe(all_stores, use_container_width=True)
            # ----------------------------------------

            # Map for All Stores
            # Filter valid
            valid_stores = all_stores.dropna(subset=['Latitude', 'Longitude'])
            
            if not valid_stores.empty:
                fig_map = px.scatter_mapbox(
                    valid_stores, 
                    lat="Latitude", lon="Longitude",
                    color="ChainName", # Color by Chain instead of Price
                    hover_name="StoreName",
                    hover_data=["City", "Address"] if "City" in valid_stores.columns else None,
                    zoom=7, 
                    mapbox_style="carto-positron",
                    title="All Store Locations by Chain"
                )
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.warning("No valid coordinates available yet. Use the Repair button above.")