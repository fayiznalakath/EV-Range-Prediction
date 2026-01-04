import streamlit as st
import numpy as np
import joblib
import base64

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="EV Range Prediction",
    page_icon="⚡",
    layout="wide"
)

# --------------------------------------------------
# Background + CSS
# --------------------------------------------------
def apply_bg_and_css(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <style>

    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
    }}

    .stApp, .stApp * {{
        color: white !important;
    }}

    h1, h2, h3 {{
        color: #00f2ff !important;
    }}

    /* Text input styling */
    input {{
        background-color: #0f172a !important;
        color: white !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }}

    /* Selectbox */
    div[data-baseweb="select"] > div {{
        background-color: #0f172a !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }}

    /* Dropdown */
    div[role="listbox"] {{
        background-color: #111 !important;
        color: white !important;
    }}

    /* Slider */
    div[data-baseweb="slider"] {{
        color: white !important;
    }}

    /* Radio */
    div[role="radiogroup"] label {{
        color: white !important;
    }}

    /* Main button */
    .stButton > button {{
        background-color: #00e676 !important;
        color: black !important;
        font-weight: bold;
        border-radius: 10px;
        width: 100%;
        height: 3em;
    }}

    /* ===== FIX STREAMLIT HEADER VISIBILITY ===== */
    header[data-testid="stHeader"] {{
        background-color: #0b1220 !important;
    }}

    header[data-testid="stHeader"] * {{
        color: white !important;
    }}

    </style>
    """, unsafe_allow_html=True)

apply_bg_and_css("ev_dark.png")

# --------------------------------------------------
# Load Model & Encoders
# --------------------------------------------------
model = joblib.load("gbr_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# --------------------------------------------------
# Helper: Safe numeric input with validation
# --------------------------------------------------
def get_number(label, default, min_val, max_val):
    value = st.text_input(label, str(default))
    try:
        value = float(value)
        if not (min_val <= value <= max_val):
            st.warning(f"{label} must be between {min_val} and {max_val}")
            return None
        return value
    except ValueError:
        st.warning(f"{label} must be a number")
        return None

# --------------------------------------------------
# Layout
# --------------------------------------------------
_, main_ui, _ = st.columns([0.2, 1.1, 1.7])

with main_ui:
    st.markdown(
    "<h1 style='font-size:37px;'>⚡ EV Range Prediction ⚡</h1>",
    unsafe_allow_html=True)
    
    st.write("Predict **real-world EV Cars driving range** using machine learning.")

    col1, col2 = st.columns(2)

    with col1:
        battery = get_number("Battery capacity (kWh)", 60, 20, 150)
        weight = get_number("Vehicle weight (kg)", 1700, 800, 3000)
        motor = get_number("Motor power (kW)", 110, 40, 400)
        speed = st.slider("Average speed (km/h)", 10, 150, 60)
        road = st.selectbox("Road type", ["City", "Highway", "Mixed"])
        elevation = get_number("Elevation gain (m)", 100, 0, 3000)

    with col2:
        temp = st.slider("Ambient temperature (°C)", -10, 50, 28)
        style = st.selectbox("Driving style", ["Eco", "Normal", "Aggressive"])
        ac = st.radio("AC usage", ["Off", "On"], horizontal=True)
        passengers = st.slider("Passenger load", 0, 6, 2)
        regen = st.selectbox("Regen braking level", ["Low", "Medium", "High"])

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Predict EV Range"):
        if None in [battery, weight, motor, elevation, passengers]:
            st.error("Please fix input errors before prediction.")
        else:
            X = np.array([[
                battery,
                weight,
                motor,
                speed,
                label_encoders["Road type"].transform([road])[0],
                elevation,
                temp,
                label_encoders["AC usage"].transform([ac])[0],
                label_encoders["Driving style"].transform([style])[0],
                passengers,
                label_encoders["Regen braking level"].transform([regen])[0]
            ]])

            pred = model.predict(X)[0]
            st.success(f"Estimated Real-World EV Range: **{pred:.2f} km**")
