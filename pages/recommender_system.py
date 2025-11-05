# recommendersystem.py
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from difflib import get_close_matches
import plotly.express as px

# ---------------------------
# Page config (first)
# ---------------------------
st.set_page_config(page_title="Recommend Apartments", page_icon="🏗️", layout="wide")

# ---------------------------
# Helper styling
# ---------------------------
st.markdown("""
<style>
.block-container { padding-top: 1.1rem; }
.card { border-radius: 14px; padding: .9rem 1rem; border: 1px solid rgba(255,255,255,.12); }
.badge { padding:.15rem .5rem; border-radius:999px; border:1px solid rgba(255,255,255,.25); font-size:.8rem; opacity:.9; }
.small { opacity:.8; font-size:.92rem; }
.tbl th, .tbl td { font-size: .95rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Cache loaders
# ---------------------------
@st.cache_resource(show_spinner=False)
def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

# @st.cache_data(show_spinner=False)
def _list_to_sorted_unique(seq):
    return sorted(pd.Series(seq).astype(str).unique().tolist())

# ---------------------------
# Load artifacts
# ---------------------------
try:
    location_df = _load_pickle("datasets/location_distance.pkl")
    cosine_sim1 = _load_pickle("datasets/cosine_sim1.pkl")
    cosine_sim2 = _load_pickle("datasets/cosine_sim2.pkl")
    cosine_sim3 = _load_pickle("datasets/cosine_sim3.pkl")
except Exception as e:
    st.error(f"Failed to load pickles: {e}")
    st.stop()

# Basic sanity checks
if not isinstance(location_df, pd.DataFrame):
    st.error("`location_distance.pkl` must be a pandas DataFrame.")
    st.stop()

# ---------------------------
# Utilities
# ---------------------------
def _safe_get_index(df: pd.DataFrame, name: str):
    """Return exact or nearest match for an index label."""
    if name in df.index:
        return name
    matches = get_close_matches(name, df.index.astype(str), n=1, cutoff=0.6)
    return matches[0] if matches else None

def _format_km(meters: float) -> str:
    try:
        return f"{meters/1000:.1f} km"
    except Exception:
        return "—"

def combine_similarity(w1: float, w2: float, w3: float):
    # All cosine_sim arrays should share the same order as location_df.index
    return (w1 * cosine_sim1) + (w2 * cosine_sim2) + (w3 * cosine_sim3)

def recommend_properties_with_scores(property_name: str, top_n: int = 5, w1=0.5, w2=0.8, w3=1.0):
    target = _safe_get_index(location_df, property_name)
    if target is None:
        raise ValueError(f"‘{property_name}’ not found. Try another name.")

    cos_mat = combine_similarity(w1, w2, w3)
    idx = location_df.index.get_loc(target)

    sim_scores = list(enumerate(cos_mat[idx]))
    # sort by similarity desc
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # remove itself (index 0 after sorting might be self; guard by id)
    filtered = [(i, s) for (i, s) in sorted_scores if i != idx][:top_n]
    top_indices = [i for (i, _) in filtered]
    top_scores = [float(s) for (_, s) in filtered]
    top_names = location_df.index[top_indices].tolist()

    out = pd.DataFrame({
        "Rank": range(1, len(top_names) + 1),
        "Property": top_names,
        "Similarity": top_scores
    })
    return target, out

# ---------------------------
# Header
# ---------------------------
st.markdown("""
<div class="card">
  <h2 style="margin:.1rem 0 .4rem 0;">Apartment Recommender</h2>
  <div class="small">Find similar apartments by multiple feature spaces and explore nearby places within a radius.</div>
</div>
""", unsafe_allow_html=True)
st.write("")

# ---------------------------
# Sidebar controls
# ---------------------------
with st.sidebar:
    st.header("Controls")
    # st.caption("Tune the weights for the three cosine similarity spaces and choose how many results to show.")
    w1 = 30
    w2 = 20
    w3 = 8
    top_n = st.slider("Top N similar", 3, 20, 7, 1)
    st.caption("Tip: Start with defaults. If results look noisy, adjust Top N.")

# ---------------------------
# Tabs
# ---------------------------
tab_similar, tab_radius = st.tabs(["Similar Apartments", "Nearby by Radius"])

# ========== TAB 1: SIMILAR APARTMENTS ==========
with tab_similar:
        st.subheader("Find Similar Apartments")
        all_apts = _list_to_sorted_unique(location_df.index)
        query = st.selectbox("Select an apartment", all_apts, index=0, key="sim_apartment")

        run_sim = st.button("Recommend", type="primary")
        if run_sim:
            try:
                target, df_sim = recommend_properties_with_scores(query, top_n=top_n, w1=w1, w2=w2, w3=w3)

                # KPI header
                k1, k2 = st.columns(2)
                with k1:
                    st.metric("Base Apartment", target)
                with k2:
                    st.metric("Results", len(df_sim))

                # Chart
                fig = px.bar(
                    df_sim,
                    x="Similarity",
                    y="Property",
                    orientation="h",
                    title="Top Similar Apartments",
                    text=[f"{v:.3f}" for v in df_sim["Similarity"]],
                )
                # fig.update_layout(height=420, xaxis_title="Similarity (cosine)", yaxis_title="")
                st.plotly_chart(fig, width='stretch')

                # Table (styled)
                styled = df_sim.copy()
                styled["Similarity"] = styled["Similarity"].map(lambda x: f"{x:.4f}")
                # st.dataframe(styled, width='stretch', hide_index=True, column_config={
                #     "Rank": st.column_config.NumberColumn(format="%d"),
                #     "Property": st.column_config.TextColumn(),
                #     "Similarity": st.column_config.TextColumn("Similarity (cosine)")
                # })

                # Download
                # csv = df_sim.to_csv(index=False).encode("utf-8")
                # st.download_button("⬇️ Download CSV", data=csv, file_name=f"similar_{target}.csv", mime="text/csv")

            except Exception as e:
                st.error(str(e))

# ========== TAB 2: NEARBY BY RADIUS ==========
with tab_radius:
    st.subheader("Find Nearby Apartments by Radius")
    # Assumption: location_df is a square matrix where rows are properties and columns are locations (distances in meters).
    colA, colB = st.columns([1, 1], vertical_alignment="center")

    with colA:
        location_options = _list_to_sorted_unique(location_df.columns)
        selected_location = st.selectbox("Pick a reference location", location_options, key="radius_location")
        radius_km = st.slider("Radius (km)", min_value=0.5, max_value=25.0, value=5.0, step=0.5)

        go = st.button("Search Nearby")
        if go:
            try:
                ser = location_df[location_df[selected_location] < radius_km * 1000][selected_location].sort_values()
                ser = ser.head(15)  # cap to 15 for display
                if ser.empty:
                    st.info("No apartments found within that radius. Try increasing it.")
                else:
                    out = pd.DataFrame({
                        "Property": ser.index.astype(str),
                        "Distance": ser.values.astype(float)
                    }).reset_index(drop=True)
                    out.insert(0, "Rank", range(1, len(out) + 1))
                    out["Distance (km)"] = out["Distance"].map(lambda x: f"{x/1000:.2f}")
                    st.dataframe(out[["Rank", "Property", "Distance (km)"]], width='stretch', hide_index=True)

                    # Chart of distances
                    fig2 = px.bar(
                        out,
                        x="Distance",
                        y="Property",
                        orientation="h",
                        title=f"Nearest to {selected_location} (≤ {radius_km:.1f} km)",
                        text=[_format_km(v) for v in out["Distance"]],
                    )
                    fig2.update_layout(height=500, xaxis_title="Distance (km)", yaxis_title="")
                    st.plotly_chart(fig2, width='stretch')

            except Exception as e:
                st.error(f"Search failed: {e}")

st.write("---")
st.caption(
    "Built by **Amul Poudel** · "
    "[GitHub](https://github.com/Amul-byte) · "
    "Deployed on Streamlit Cloud / Render"
)
