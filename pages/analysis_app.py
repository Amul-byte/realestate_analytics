# app.py — Real Estate Analytics Dashboard (Refined)
# Author: Amul Poudel  •  GitHub: https://github.com/Amul-byte
# Notes:
# - Production-friendly structure with caching and filters
# - Uses only Streamlit Cloud / Render-friendly libraries

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# Page Config & Minimal Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="🏙️ Real Estate Analytics",
    page_icon="🏗️",
    layout="wide",
    menu_items={
        "Get help": "https://github.com/Amul-byte",
        "About": "Interactive Real Estate Analytics • Streamlit + Plotly • Portfolio-ready"
    },
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 1rem; }
        h1, h2, h3 { letter-spacing: .2px; }
        .metric-card { border:1px solid rgba(0,0,0,0.08); border-radius:14px; padding:1rem; }
        .small { opacity:.8; }
        .subtle { color: #64748B; }
        .tight { margin-top: .25rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 6px; }
        .stTabs [data-baseweb="tab"] { border-radius: 10px; padding: 8px 14px; }
        hr { border: none; border-top:1px solid rgba(0,0,0,.08); margin: .6rem 0 1rem 0; }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# Paths & Constants
# -----------------------------------------------------------------------------
DATA_DIR = Path("datasets")
CSV_PATH = DATA_DIR / "Data Visualization 1.csv"
FEATURES_PKL = DATA_DIR / "feature_list.pkl"

PLOTLY_TEMPLATE = "plotly_white"
MAP_ZOOM_DEFAULT = 10

# -----------------------------------------------------------------------------
# Cached Loaders
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    """Load CSV into DataFrame; return empty DataFrame if missing."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df

@st.cache_data(show_spinner=False)
def load_features_text(path: Path) -> str:
    """Load feature text string from pickle; return empty string if missing."""
    if not path.exists():
        return ""
    import pickle
    return pickle.load(open(path, "rb"))

@st.cache_data(show_spinner=False)
def compute_sector_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate numeric metrics by sector for map visualization."""
    if df.empty or "sector" not in df.columns:
        return pd.DataFrame()
    tmp = df.copy()
    # Compute price_per_sqft if not present
    if "price_per_sqft" not in tmp.columns and {"price", "built_up_area"}.issubset(tmp.columns):
        tmp["price_per_sqft"] = tmp["price"] / tmp["built_up_area"].replace(0, np.nan)
    cols = [c for c in ["price", "price_per_sqft", "built_up_area", "latitude", "longitude"] if c in tmp.columns]
    if not cols:
        return pd.DataFrame()
    g = tmp.groupby("sector", as_index=True)[cols].mean(numeric_only=True)
    g = g.dropna(subset=[c for c in ["latitude", "longitude"] if c in g.columns], how="any")
    return g

@st.cache_data(show_spinner=False)
def generate_wordcloud_array(text: str, width: int = 800, height: int = 500) -> np.ndarray:
    """Generate a wordcloud as an image array (cached)."""
    if not text:
        # Return a blank white image
        return np.ones((height, width, 3), dtype=np.uint8) * 255
    wc = WordCloud(
        width=width,
        height=height,
        background_color="white",
        stopwords=set(["s"]),
        min_font_size=8,
        prefer_horizontal=0.95,
        collocations=False,
        max_words=400,
    ).generate(text)
    return wc.to_array()

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def money(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"${x:,.0f}"

def sqft(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:,.0f} sqft"

def price_sqft(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"${x:,.0f}/sqft"

def apply_filters(
    df: pd.DataFrame,
    sectors: List[str] | None,
    prop_types: List[str] | None,
    bhk: List[int] | None,
    price_range: Tuple[int, int] | None,
    area_range: Tuple[int, int] | None,
) -> pd.DataFrame:
    """Return filtered DataFrame with defensive checks."""
    if df.empty:
        return df
    out = df.copy()

    # Derived
    if "price_per_sqft" not in out.columns and {"price", "built_up_area"}.issubset(out.columns):
        out["price_per_sqft"] = out["price"] / out["built_up_area"].replace(0, np.nan)

    if sectors and "sector" in out.columns and "overall" not in sectors:
        out = out[out["sector"].isin(sectors)]

    if prop_types and "property_type" in out.columns:
        out = out[out["property_type"].isin(prop_types)]

    if bhk and "bedRoom" in out.columns:
        out = out[out["bedRoom"].isin(bhk)]

    if price_range and "price" in out.columns:
        out = out[out["price"].between(price_range[0], price_range[1])]

    if area_range and "built_up_area" in out.columns:
        out = out[out["built_up_area"].between(area_range[0], area_range[1])]

    return out

# -----------------------------------------------------------------------------
# Sidebar — Filters
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🔎 Filters")
    raw_df = load_csv(CSV_PATH)

    if raw_df.empty:
        st.warning("Data file not found. Place CSV at: datasets/Data Visualization 1.csv")
        sectors = []
        prop_types = []
        bhk = []
        price_range = None
        area_range = None
    else:
        # basic introspection
        sectors_list = sorted(raw_df["sector"].dropna().unique().tolist()) if "sector" in raw_df.columns else []
        prop_types_list = sorted(raw_df["property_type"].dropna().unique().tolist()) if "property_type" in raw_df.columns else []
        bhk_list = sorted([int(x) for x in raw_df["bedRoom"].dropna().unique().tolist()]) if "bedRoom" in raw_df.columns else []

        min_price = int(np.nanmin(raw_df["price"])) if "price" in raw_df.columns else 0
        max_price = int(np.nanmax(raw_df["price"])) if "price" in raw_df.columns else 1_000_000

        min_area = int(np.nanmin(raw_df["built_up_area"])) if "built_up_area" in raw_df.columns else 0
        max_area = int(np.nanmax(raw_df["built_up_area"])) if "built_up_area" in raw_df.columns else 5000

        sectors = st.multiselect("Sector", options=["overall"] + sectors_list, default=["overall"])
        prop_types = st.multiselect("Property Type", options=prop_types_list, default=prop_types_list[:2] if prop_types_list else [])
        bhk = st.multiselect("BHK", options=bhk_list, default=bhk_list[:3] if bhk_list else [])

        price_range = st.slider("Price Range", min_price, max_price, (min_price, max_price), step=max(1, (max_price - min_price)//100))
        area_range = st.slider("Area (sqft)", min_area, max_area, (min_area, max_area), step=max(1, (max_area - min_area)//100))

        st.caption("Tip: Use multiple sectors/property types to compare cohorts.")
        st.divider()
        show_map = st.checkbox("Show Map", value=True)
        show_wordcloud = st.checkbox("Show WordCloud", value=True)

# -----------------------------------------------------------------------------
# Title & Intro
# -----------------------------------------------------------------------------
st.title("🏙️ Real Estate Analytics Dashboard")
st.caption("Explore market dynamics across sectors, property types, and BHK configurations.")

# -----------------------------------------------------------------------------
# Data Prep
# -----------------------------------------------------------------------------
df = raw_df.copy()
df = apply_filters(df, sectors, prop_types, bhk, price_range, area_range)
sector_agg = compute_sector_agg(df)

# -----------------------------------------------------------------------------
# KPI Cards
# -----------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Listings", f"{len(df):,}")
with c2:
    st.metric("Median Price", money(df["price"].median() if "price" in df.columns else None))
with c3:
    st.metric("Median Area", sqft(df["built_up_area"].median() if "built_up_area" in df.columns else None))
with c4:
    med_pps = df["price_per_sqft"].median() if "price_per_sqft" in df.columns else None
    st.metric("Median $/sqft", price_sqft(med_pps))

st.write("---")

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab_overview, tab_map, tab_distrib, tab_insights = st.tabs(
    ["Overview", "Geomap", "Distributions", "Insights"]
)

# --- Overview Tab ---
with tab_overview:
    st.subheader("Area vs Price")
    if {"built_up_area", "price"}.issubset(df.columns):
        color_col = "bedRoom" if "bedRoom" in df.columns else None
        fig_scatter = px.scatter(
            df, x="built_up_area", y="price",
            color=color_col,
            hover_data=[c for c in df.columns if c not in ("latitude","longitude")],
            template=PLOTLY_TEMPLATE,
            title="Price vs. Built-up Area",
            height=500
        )
        st.plotly_chart(fig_scatter, width='stretch')
    else:
        st.info("Need columns: built_up_area, price.")

    st.subheader("Features WordCloud")
    if show_wordcloud:
        features_text = load_features_text(FEATURES_PKL)
        wc_img = generate_wordcloud_array(features_text, width=1100, height=350)
        fig_wc, ax = plt.subplots(figsize=(11, 3.8), dpi=150)
        ax.imshow(wc_img)
        ax.axis("off")
        st.pyplot(fig_wc, width='stretch')
    else:
        st.caption("WordCloud is hidden (toggle in sidebar).")

# --- Geomap Tab ---
with tab_map:
    st.subheader("Sector Price per Sqft — Geomap")
    if show_map and not sector_agg.empty and {"latitude","longitude"}.issubset(sector_agg.columns):
        # Ensure color column exists
        if "price_per_sqft" not in sector_agg.columns and {"price","built_up_area"}.issubset(sector_agg.columns):
            sector_agg["price_per_sqft"] = sector_agg["price"] / sector_agg["built_up_area"].replace(0, np.nan)

        map_fig = px.scatter_map(
            sector_agg.reset_index(),
            lat="latitude", lon="longitude",
            color="price_per_sqft" if "price_per_sqft" in sector_agg.columns else None,
            size="built_up_area" if "built_up_area" in sector_agg.columns else None,
            hover_name="sector",
            hover_data=[c for c in sector_agg.columns if c not in ("latitude","longitude")],
            color_continuous_scale=px.colors.cyclical.IceFire,
            zoom=MAP_ZOOM_DEFAULT,
            height=620,
            template=PLOTLY_TEMPLATE,
        )
        map_fig.update_layout(map_style="open-street-map", margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(map_fig, width='stretch')
        st.caption("Map displays sector means; filtered by your selections.")
    else:
        st.info("Map unavailable. Ensure sector aggregation & lat/long columns are present.")

# --- Distributions Tab ---
with tab_distrib:
    left, right = st.columns((1.25, 1))
    with left:
        st.subheader("BHK Price Distribution (Box)")
        if {"bedRoom", "price"}.issubset(df.columns):
            fig_box = px.box(
                df[df["bedRoom"] <= (df["bedRoom"].max() if "bedRoom" in df.columns else 4)],
                x="bedRoom", y="price", template=PLOTLY_TEMPLATE, title="BHK Price Range"
            )
            st.plotly_chart(fig_box, width='stretch')
        else:
            st.info("Need columns: bedRoom, price.")
    with right:
        st.subheader("BHK Share (Pie)")
        if "bedRoom" in df.columns:
            fig_pie = px.pie(df, names="bedRoom", template=PLOTLY_TEMPLATE, title="BHK Composition")
            st.plotly_chart(fig_pie, width='stretch')
        else:
            st.info("Need column: bedRoom.")

    st.subheader("Price Density by Property Type")
    if {"price","property_type"}.issubset(df.columns):
        fig_hist, ax = plt.subplots(figsize=(10, 4), dpi=140)
        # Hist + KDE overlays
        for label, sub in df.groupby("property_type"):
            vals = sub["price"].dropna()
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=60, alpha=0.35, density=True, label=f"{label} hist")
            sns.kdeplot(vals, linewidth=1.8, label=f"{label} KDE", ax=ax)
        ax.set_xlabel("Price")
        ax.set_ylabel("Density")
        ax.legend()
        st.pyplot(fig_hist, width='stretch')
        st.caption("Overlaid histograms and KDEs for price, split by property type.")
    else:
        st.info("Need columns: price, property_type.")

# --- Insights Tab ---
with tab_insights:
    st.subheader("Top Sectors by Median $/sqft")
    if {"sector","price_per_sqft"}.issubset(df.columns):
        top = (
            df[["sector","price_per_sqft"]].dropna()
              .groupby("sector")["price_per_sqft"]
              .median()
              .sort_values(ascending=False)
              .head(12)
              .reset_index()
        )
        st.dataframe(top, width='stretch')
    else:
        st.info("Need columns: sector, price_per_sqft (or price & built_up_area to derive).")

    st.markdown("### Summary Statistics (numeric)")
    if not df.empty:
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols):
            st.dataframe(df[num_cols].describe().T, width='stretch')
        else:
            st.caption("No numeric columns found.")
    else:
        st.caption("No data in current selection.")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.write("---")
st.caption(
    "Built by **Amul Poudel** · "
    "[GitHub](https://github.com/Amul-byte) · "
    "Deployed on Streamlit Cloud / Render"
)
