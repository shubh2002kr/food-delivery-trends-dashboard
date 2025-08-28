# app.py
# 🍽️ Food Delivery Trends — Zomato/Swiggy
# Run: streamlit run app.py

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Food Delivery Trends — Zomato/Swiggy", page_icon="🍽️", layout="wide")

# ===========================
# Sample Data Generator
# ===========================

@st.cache_data
def make_sample_df(n_rows=1200, seed=42):
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1, 8, 0, 0)
    platforms = ["Zomato", "Swiggy"]
    cities = ["Bengaluru", "Mumbai", "Delhi", "Hyderabad", "Chennai", "Pune", "Kolkata"]
    cuisines = ["North Indian", "South Indian", "Chinese", "Biryani", "Pizza", "Burgers", "Desserts", "Healthy"]
    restaurants = [f"Restaurant {i}" for i in range(1, 51)]
    payment_methods = ["UPI", "Card", "COD", "Wallet"]

    city_coords = {
        "Bengaluru": (12.9716, 77.5946),
        "Mumbai": (19.0760, 72.8777),
        "Delhi": (28.6139, 77.2090),
        "Hyderabad": (17.3850, 78.4867),
        "Chennai": (13.0827, 80.2707),
        "Pune": (18.5204, 73.8567),
        "Kolkata": (22.5726, 88.3639),
    }

    rows = []
    for i in range(n_rows):
        order_datetime = start + timedelta(minutes=int(rng.integers(0, 60 * 24 * 180)))
        platform = rng.choice(platforms)
        city = rng.choice(cities)
        restaurant = rng.choice(restaurants)
        cuisine = rng.choice(cuisines)

        delivery_time = max(10, int(rng.normal(35, 12)))
        distance_km = round(max(0.5, rng.normal(3.5, 1.5)), 2)

        base_value = rng.uniform(150, 600)
        surge = 1.0 + (0.15 if order_datetime.hour in [12, 13, 20, 21] else 0)
        order_value = round(base_value * surge, 2)

        rating = float(np.clip(rng.normal(4.1, 0.6), 1.0, 5.0))
        is_delayed = int(delivery_time > 45)
        payment = rng.choice(payment_methods, p=[0.55, 0.2, 0.2, 0.05])

        customer_id = f"C{rng.integers(10000, 99999)}"
        order_id = f"O{i+1:06d}"

        lat0, lon0 = city_coords[city]
        lat = lat0 + rng.normal(0, 0.03)
        lon = lon0 + rng.normal(0, 0.03)

        rows.append({
            "order_id": order_id,
            "order_datetime": order_datetime,
            "platform": platform,
            "city": city,
            "restaurant_name": restaurant,
            "cuisine": cuisine,
            "delivery_time_min": delivery_time,
            "distance_km": distance_km,
            "order_value": order_value,
            "rating": round(rating, 2),
            "is_delayed": is_delayed,
            "payment_method": payment,
            "customer_id": customer_id,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
        })

    return pd.DataFrame(rows)

# ===========================
# Loading & Harmonization
# ===========================

REQUIRED_ORDER_COLS = [
    "order_id","order_datetime","platform","city","restaurant_name","cuisine",
    "delivery_time_min","distance_km","order_value","rating","is_delayed",
    "payment_method","customer_id","lat","lon"
]

def infer_platform_from_filename(name: str):
    if not name:
        return None
    n = name.lower()
    if "zomato" in n: return "Zomato"
    if "swiggy" in n: return "Swiggy"
    return None

@st.cache_data
def read_csv_flex(file, platform_hint=None):
    """Read CSV, normalize column names, add platform if missing, parse datetime if present."""
    if file is None:
        return None
    df = pd.read_csv(file)

    # normalize column names (strip / lower / spaces->underscores)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # common variants → standard names
    rename_map = {
        "datetime": "order_datetime",
        "timestamp": "order_datetime",
        "time": "order_datetime",
        "order_time": "order_datetime",
        "restaurant": "restaurant_name",
        "shop_name": "restaurant_name",
        "eta_min": "delivery_time_min",
        "delivery_time": "delivery_time_min",
        "distance": "distance_km",
        "amount": "order_value",
        "total_amount": "order_value",
        "price": "order_value",
        "review_score": "rating",
        "payment": "payment_method",
        "customer": "customer_id",
        "latitude": "lat",
        "longitude": "lon",
        "orderid": "order_id",
        "order_no": "order_id",
        "order_number": "order_id",
    }
    df = df.rename(columns=rename_map)

    # add platform if missing
    if "platform" not in df.columns and platform_hint:
        df["platform"] = platform_hint

    # coerce types
    if "order_datetime" in df.columns:
        df["order_datetime"] = pd.to_datetime(df["order_datetime"], errors="coerce")
    for col in ["delivery_time_min","distance_km","order_value","rating","lat","lon"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "is_delayed" in df.columns:
        df["is_delayed"] = (
            df["is_delayed"].astype(str).str.lower()
            .map({"true":1,"1":1,"yes":1,"y":1,"false":0,"0":0,"no":0,"n":0})
            .fillna(pd.to_numeric(df["is_delayed"], errors="coerce"))
        )
        df["is_delayed"] = df["is_delayed"].fillna(0).astype(int)
    return df

def ensure_minimum_columns(df: pd.DataFrame) -> pd.DataFrame:
    # create a synthetic order_id if missing
    if "order_id" not in df.columns:
        df["order_id"] = np.arange(1, len(df) + 1)
    # ensure platform exists (if filename inference & column both missing)
    if "platform" not in df.columns:
        df["platform"] = "Unknown"
    return df

def kpi(label, value, help_text=None):
    st.metric(label, value, help=help_text)

def to_hhmm(minutes: float):
    if pd.isna(minutes):
        return "—"
    m = int(minutes)
    h, mm = divmod(m, 60)
    return f"{h}h {mm}m" if h else f"{mm}m"

# ===========================
# Sidebar — Data & Single Dropzone
# ===========================

st.sidebar.title("Data Source")
mode = st.sidebar.radio(
    "Choose data",
    ["Use sample dataset", "Upload CSVs (one drag-and-drop)"],
    index=0
)

uploaded_files = None
if mode == "Upload CSVs (one drag-and-drop)":
    uploaded_files = st.sidebar.file_uploader(
        "Drag & drop one or many CSVs (Zomato/Swiggy/combined)",
        type=["csv"],
        accept_multiple_files=True,
        key="multi_csv"
    )

# Build dataframe
if mode == "Use sample dataset":
    df = make_sample_df()
else:
    parts = []
    if not uploaded_files:
        st.info("Drop one or more CSV files to continue (can be mixed from Zomato and Swiggy).")
        st.stop()
    for f in uploaded_files:
        plat = infer_platform_from_filename(f.name) or None
        d = read_csv_flex(f, platform_hint=plat)
        if d is not None and not d.empty:
            parts.append(d)
    if not parts:
        st.warning("No readable rows found in the uploaded files.")
        st.stop()
    df = pd.concat(parts, ignore_index=True, sort=False)

# Guarantee critical columns
df = ensure_minimum_columns(df)

# enrich time columns
if "order_datetime" in df.columns:
    df = df.sort_values("order_datetime")
    df["date"] = df["order_datetime"].dt.date
    df["hour"] = df["order_datetime"].dt.hour
    df["weekday"] = df["order_datetime"].dt.day_name()

# quick missing notice
missing = [c for c in REQUIRED_ORDER_COLS if c not in df.columns]
if missing:
    st.sidebar.warning(f"Missing columns detected: {missing}. The app will still work with available columns.")

# ===========================
# Header
# ===========================

st.title("🍽️ Food Delivery Trends — Zomato/Swiggy")
st.markdown(
    "<p style='margin-top:-10px; margin-bottom:10px; font-size:0.95rem;'>"
    "🔹 Built by <b>Shubh Kumar</b>"
    "</p>",
    unsafe_allow_html=True,
)
st.caption("Drop one or multiple CSVs (mixed allowed). The app merges them, infers platforms, and analyzes trends.")

# ===========================
# Filters
# ===========================

lcol, rcol = st.sidebar.columns(2)
with lcol:
    city_sel = st.multiselect("City", sorted(df["city"].dropna().unique()) if "city" in df else [])
with rcol:
    plat_sel = st.multiselect("Platform", sorted(df["platform"].dropna().unique()) if "platform" in df else [])

cuisine_sel = st.sidebar.multiselect("Cuisine", sorted(df["cuisine"].dropna().unique()) if "cuisine" in df else [])
date_range = None
if "order_datetime" in df:
    min_d, max_d = df["order_datetime"].min(), df["order_datetime"].max()
    if pd.notna(min_d) and pd.notna(max_d):
        date_range = st.sidebar.date_input(
            "Date range",
            value=(min_d.date(), max_d.date()),
            min_value=min_d.date(),
            max_value=max_d.date(),
        )

# Apply filters
mask = pd.Series(True, index=df.index)
if city_sel and "city" in df: mask &= df["city"].isin(city_sel)
if plat_sel and "platform" in df: mask &= df["platform"].isin(plat_sel)
if cuisine_sel and "cuisine" in df: mask &= df["cuisine"].isin(cuisine_sel)
if date_range and "order_datetime" in df:
    d1 = pd.to_datetime(date_range[0])
    d2 = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    mask &= (df["order_datetime"] >= d1) & (df["order_datetime"] <= d2)

fdf = df[mask].copy()

# Download filtered
st.sidebar.markdown("---")
st.sidebar.download_button("Download filtered CSV", data=fdf.to_csv(index=False), file_name="filtered_food_delivery.csv")

# ===========================
# KPIs
# ===========================

k1, k2, k3, k4, k5 = st.columns(5)
total_orders = int(fdf["order_id"].nunique()) if "order_id" in fdf else len(fdf)
avg_rating = round(fdf["rating"].mean(), 2) if "rating" in fdf else None
avg_delivery = fdf["delivery_time_min"].mean() if "delivery_time_min" in fdf else None
revenue = fdf["order_value"].sum() if "order_value" in fdf else None
delay_rate = (fdf["is_delayed"].mean() * 100) if "is_delayed" in fdf else None

with k1: kpi("Total Orders", f"{total_orders:,}")
with k2: kpi("Avg. Rating", f"{avg_rating:.2f}" if avg_rating is not None else "—")
with k3: kpi("Avg. Delivery", to_hhmm(avg_delivery) if avg_delivery is not None else "—", "Mean delivery time")
with k4: kpi("Revenue", f"₹{revenue:,.0f}" if revenue is not None else "—")
with k5: kpi("Delay Rate", f"{delay_rate:.1f}%" if delay_rate is not None else "—")

st.markdown("---")

# ===========================
# Time Series — Orders & Revenue
# ===========================

if "date" in fdf:
    ts1, ts2 = st.columns(2)
    by_date = fdf.groupby("date").agg(orders=("order_id","nunique"), revenue=("order_value","sum")).reset_index()

    with ts1:
        fig = px.line(by_date, x="date", y="orders", markers=True, title="Orders over Time")
        st.plotly_chart(fig, use_container_width=True)
    with ts2:
        if "order_value" in fdf:
            fig = px.line(by_date, x="date", y="revenue", markers=True, title="Revenue over Time")
            st.plotly_chart(fig, use_container_width=True)

# ===========================
# Distribution — Delivery & Ratings
# ===========================

dist1, dist2 = st.columns(2)
with dist1:
    if "delivery_time_min" in fdf:
        fig = px.histogram(fdf, x="delivery_time_min", nbins=30, title="Delivery Time Distribution (minutes)")
        st.plotly_chart(fig, use_container_width=True)
with dist2:
    if "rating" in fdf:
        fig = px.histogram(fdf, x="rating", nbins=20, title="Ratings Distribution (1–5)")
        st.plotly_chart(fig, use_container_width=True)

# ===========================
# Top Entities — Cuisines & Restaurants
# ===========================

top1, top2 = st.columns(2)
with top1:
    if "cuisine" in fdf:
        top_cui = fdf["cuisine"].value_counts().reset_index()
        top_cui.columns = ["cuisine", "orders"]
        fig = px.bar(top_cui, x="cuisine", y="orders", title="Top Cuisines by Orders")
        st.plotly_chart(fig, use_container_width=True)
with top2:
    if "restaurant_name" in fdf:
        top_rest = fdf["restaurant_name"].value_counts().head(15).reset_index()
        top_rest.columns = ["restaurant_name", "orders"]
        fig = px.bar(top_rest, x="orders", y="restaurant_name", orientation="h", title="Top Restaurants by Orders (Top 15)")
        st.plotly_chart(fig, use_container_width=True)

# ===========================
# Peak Hours & Weekday Patterns
# ===========================

if "hour" in fdf and "weekday" in fdf:
    st.subheader("Demand Patterns")
    ph1, ph2 = st.columns(2)
    with ph1:
        by_hour = fdf.groupby("hour").size().reset_index(name="orders")
        fig = px.bar(by_hour, x="hour", y="orders", title="Orders by Hour")
        st.plotly_chart(fig, use_container_width=True)
    with ph2:
        by_weekday_hour = fdf.groupby(["weekday","hour"]).size().reset_index(name="orders")
        fig = px.density_heatmap(by_weekday_hour, x="hour", y="weekday", z="orders", nbinsx=24, title="Heatmap: Hour vs Weekday")
        st.plotly_chart(fig, use_container_width=True)

# ===========================
# Platform & City Breakdown (safe)
# ===========================

br1, br2 = st.columns(2)
with br1:
    if "platform" in fdf:
        plat = fdf.groupby("platform").size().reset_index(name="orders")
        if "order_value" in fdf:
            plat_rev = fdf.groupby("platform")["order_value"].sum().reset_index(name="revenue")
            plat = plat.merge(plat_rev, on="platform", how="left")
        fig = px.bar(plat, x="platform", y="orders", title="Orders by Platform")
        st.plotly_chart(fig, use_container_width=True)

with br2:
    if "city" in fdf:
        city = fdf.groupby("city").size().reset_index(name="orders")
        fig = px.bar(city, x="city", y="orders", title="Orders by City")
        st.plotly_chart(fig, use_container_width=True)

# ===========================
# Geo Map (if coordinates available)
# ===========================

if "lat" in fdf and "lon" in fdf:
    st.subheader("Order Locations")
    sample_map = fdf.sample(min(1000, len(fdf)), random_state=7)
    fig = px.scatter_mapbox(
        sample_map, lat="lat", lon="lon", hover_name="restaurant_name",
        hover_data=["city","platform","order_value"], zoom=3, height=450
    )
    fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

# ===========================
# Data Table
# ===========================

st.subheader("Data Preview")
st.dataframe(fdf.head(1000))
st.caption("Tip: Use the camera icon in Plotly charts to export PNGs.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Plotly, and Pandas — by Shubh Kumar")
