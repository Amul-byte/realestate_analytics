import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Property Input", page_icon="🏗️", layout="wide")
st.title(" ")
st.title("🏙️ Property Details")
st.caption("Fill the form and get an instant price prediction.")

# ---------------------------
# Helpers (cached loaders)
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data(show_spinner=False)
def get_sector_options(df):
    # Clean and sort unique sectors; pick mode as default
    opts = sorted(pd.Series(df["sector"].dropna().unique()).astype(str).tolist())
    default = None
    try:
        default = str(df["sector"].mode()[0])
    except Exception:
        default = opts[0] if opts else ""
    return opts, default

# ---------------------------
# Load artifacts safely
# ---------------------------
try:
    df = load_pickle("df.pkl")
except Exception as e:
    st.error(f"Couldn't load df.pkl: {e}")
    st.stop()

try:
    pipeline = load_pickle("pipeline.pkl")
except Exception as e:
    st.error(f"Couldn't load pipeline.pkl: {e}")
    st.stop()

sector_options, sector_default = get_sector_options(df)

# ---------------------------
# Small style touch
# ---------------------------
st.markdown("""
<style>
.small-hint { opacity: 0.7; font-size: 0.9rem; }
.block-container { padding-top: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Default values (you can tweak)
# ---------------------------
DEFAULTS = {
    "property_type": "flat",              # ["flat","house"]
    "furnishing_type": "semifurnished",   # ["unfurnished","semifurnished","furnished"]
    "luxury_category": "Medium",          # ["Low","Medium","High"]
    "bedRoom": 3,
    "bathroom": 2,
    "balcony": '1',
    "built_up_area": 1200,
    "servant_room": 0,
    "store_room": 0,
    "sector": sector_default,
    "agePossession": "Relatively New",    # ["Under Construction","Old Property ","New Property ","Moderately Old","Relatively New"]
    "floor_category": "Mid Floor",        # ["Mid Floor","Low Floor","High Floor"]
}

# ---------------------------
# Form
# ---------------------------
with st.form("property_form", clear_on_submit=False):
    st.subheader("Tell us about the property")

    # Row 1
    c1, c2, c3 = st.columns([1.1, 1, 1])
    with c1:
        property_type = st.selectbox(
            "🏠 Property Type",
            ["flat", "house"],
            index=["flat","house"].index(DEFAULTS["property_type"]),
            help="Choose the closest match."
        )
    with c2:
        furnishing_type = st.selectbox(
            "🪑 Furnishing Type",
            ["unfurnished", "semifurnished", "furnished"],
            index=["unfurnished", "semifurnished", "furnished"].index(DEFAULTS["furnishing_type"])
        )
    with c3:
        luxury_category = st.selectbox(
            "💎 Luxury Category",
            ["Low", "Medium", "High"],
            index=["Low","Medium","High"].index(DEFAULTS["luxury_category"])
        )

    # Row 2
    c4, c5, c6 = st.columns([1, 1, 1])
    with c4:
        bedRoom = st.number_input("🛏️ Bedrooms", min_value=0, max_value=12, value=DEFAULTS["bedRoom"], step=1)
    with c5:
        bathroom = st.number_input("🛁 Bathrooms", min_value=0, max_value=12, value=DEFAULTS["bathroom"], step=1)
    with c6:
        # balcony = st.number_input("🪟 Balconies", min_value=0, max_value=3, value=DEFAULTS["balcony"], step=1)
        balcony = st.selectbox(
            "🪟 Balconies",
            ["1", "2", "3","3+"],
            index=["1", "2", "3","3+"].index(DEFAULTS["balcony"])
        )

    # Row 3
    c7, c8, c9 = st.columns([1, 1, 1])
    with c7:
        built_up_area = st.number_input(
            "📐 Built-up Area (sq ft)",
            min_value=0, max_value=100000, value=DEFAULTS["built_up_area"], step=50,
            help="Enter total built-up area."
        )
    with c8:
        servant_room = st.number_input("👤 Servant Room (count)", min_value=0, max_value=2, value=DEFAULTS["servant_room"], step=1)
    with c9:
        store_room = st.number_input("📦 Store Room (count)", min_value=0, max_value=2, value=DEFAULTS["store_room"], step=1)

    # Row 4
    c10, c11, c12 = st.columns([1, 1, 1])
    with c10:
        # default index for sector
        if sector_options:
            sector_index = sector_options.index(DEFAULTS["sector"]) if DEFAULTS["sector"] in sector_options else 0
        else:
            sector_options = [""]
            sector_index = 0
        sector = st.selectbox("📍 Sector", sector_options, index=sector_index)
    with c11:
        agePossession = st.selectbox(
            "📅 Age / Possession",
            ["Under Construction", "Old Property ", "New Property ", "Moderately Old", "Relatively New"],
            index=["Under Construction", "Old Property ", "New Property ", "Moderately Old", "Relatively New"].index(DEFAULTS["agePossession"])
        )
    with c12:
        floor_category = st.selectbox(
            "⬆️ Floor Category",
            ["Mid Floor", "Low Floor", "High Floor"],
            index=["Mid Floor","Low Floor","High Floor"].index(DEFAULTS["floor_category"])
        )

    st.markdown("<div class='small-hint'>Tip: Keep option labels exactly as in training to avoid unknown-category errors.</div>", unsafe_allow_html=True)

    left, right = st.columns([1, 1])
    with left:
        submitted = st.form_submit_button("🚀 Submit")
    with right:
        reset = st.form_submit_button("↩️ Reset to Defaults")

# If reset pressed, just rerun the app (defaults will apply)
if reset:
    st.rerun()

# ---------------------------
# On submit
# ---------------------------
if submitted:
    # Column names EXACT (including spaces) to match your pipeline
    user_inputs = {
        "property_type": property_type,
        "sector": sector,
        "bedRoom": bedRoom,
        "bathroom": bathroom,
        "balcony": balcony,
        "agePossession": agePossession,
        "built_up_area": built_up_area,
        "servant room": servant_room,
        "store room": store_room,
        "furnishing_type": furnishing_type,
        "luxury_category": luxury_category,
        "floor_category": floor_category,
    }

    user_df = pd.DataFrame([user_inputs])
    st.success("Inputs captured! 🎉")
    # st.dataframe(user_df, use_container_width=True)
    base = np.expm1(pipeline.predict(user_df))[0]
    low = base - 0.22
    high = base + 0.22
    
    st.text(f'The price of flat is between {round(low,2)} Cr and {round(high,2)} Cr')

