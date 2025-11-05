# app.py
import streamlit as st
import pandas as pd
from pathlib import Path

# ---------------------------
# Page config (must be first)
# ---------------------------
st.set_page_config(
    page_title="Real Estate Suite",
    page_icon="🏗️",
    layout="wide"
)

# ---------------------------
# Minimal styling
# ---------------------------
st.markdown("""
<style>
/* tighten top padding */
.block-container { padding-top: 1.2rem; }

/* gradient hero */
.hero {
  padding: 1.2rem 1.4rem;
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(59,130,246,0.15), rgba(236,72,153,0.12));
  border: 1px solid rgba(255,255,255,0.15);
}

/* glassy cards */
.card {
  border-radius: 16px;
  padding: 1rem 1.1rem;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.03);
  transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease;
}
.card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 22px rgba(0,0,0,0.20);
  border-color: rgba(255,255,255,0.25);
}
.small { opacity:.8; font-size:.92rem; }
.footer { opacity:.75; font-size:.9rem; }
.kpi { text-align:center; padding:.6rem .8rem; border-radius:14px; border:1px dashed rgba(255,255,255,0.18); }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Hero
# ---------------------------
st.markdown("""
<div class="hero">
  <h1 style="margin-bottom:.4rem;">Real Estate Analytics Suite</h1>
  <div class="small">
    Explore market patterns, price predictions, and nearby recommendations — all in one place.
  </div>
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------------------
# Quick stats (safe + optional)
# ---------------------------
rows = sectors = avg_price = med_pps = "—"
data_path_csv = Path("datasets") / "Data Visualization 1.csv"

try:
    @st.cache_data(show_spinner=False)
    def _load_home_data(path: Path):
        df = pd.read_csv(path)
        return df

    df = _load_home_data(data_path_csv)
    rows = f"{len(df):,}"
    if "sector" in df.columns:
        sectors = df["sector"].nunique()
    if "price" in df.columns:
        avg_price = f"{df['price'].mean():,.2f} Cr."
    if "price_per_sqft" in df.columns:
        med_pps = f"{df['price_per_sqft'].median():,.0f}"
except Exception:
    pass

k1, k2, k3, k4 = st.columns(4)
with k1: st.markdown(f'<div class="kpi"><h3>Rows</h3><h2>{rows}</h2></div>', unsafe_allow_html=True)
with k2: st.markdown(f'<div class="kpi"><h3>Sectors</h3><h2>{sectors}</h2></div>', unsafe_allow_html=True)
with k3: st.markdown(f'<div class="kpi"><h3>Avg Price</h3><h2>{avg_price}</h2></div>', unsafe_allow_html=True)
with k4: st.markdown(f'<div class="kpi"><h3>Median ₹/sqft</h3><h2>{med_pps}</h2></div>', unsafe_allow_html=True)

st.write("")

# ---------------------------
# Navigation cards
# ---------------------------
st.subheader("Features")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="card">
      <h3>Analysis</h3>
      <div class="small">Interactive maps, wordclouds, and BHK comparisons to spot pricing patterns.</div>
    </div>
    """, unsafe_allow_html=True)
    # Link to your page (adjust the page path/label to your filenames)
    # st.page_link("pages/Analysis_app.py", label="Open Analysis", icon="")
    

with c2:
    st.markdown("""
    <div class="card">
      <h3>Price Predictor</h3>
      <div class="small">Fill a short form and get an instant price estimate with a confidence band.</div>
    </div>
    """, unsafe_allow_html=True)
    # st.page_link("pages/Price_predictor.py", label="Open Price Predictor", icon="")

with c3:
    st.markdown("""
    <div class="card">
      <h3>Recommender</h3>
      <div class="small">Find similar apartments and explore nearby options by radius.</div>
    </div>
    """, unsafe_allow_html=True)
    # st.page_link("pages/Recommender_System.py", label="Open Recommender", icon="")

st.write("")

# ---------------------------
# Tips / Notes
# ---------------------------
with st.expander("Tips (read me once)"):
    st.markdown("""
- Use the **sidebar** page list or the buttons above to navigate.
- Analysis map tooltips work best when your dataset includes `sector`, `latitude`, `longitude`, `built_up_area`, `price`, and `price_per_sqft`.
- If you update models or pickles, just refresh — data loads are cached for speed.
""")

st.write("")

# ---------------------------
# Footer
# ---------------------------
st.write("---")
st.caption(
    "Built by **Amul Poudel** · "
    "[GitHub](https://github.com/Amul-byte) · "
    "Deployed on Streamlit Cloud / Render"
)