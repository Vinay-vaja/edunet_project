"""
=============================================================
 AI Disease Prediction System - Streamlit Web App
 SDG 3 – Good Health and Well-being
=============================================================
 A clean, professional, and beginner-friendly UI for
 predicting diabetes using a trained ML model.
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt

# ── Page Configuration ────────────────────────────────────
st.set_page_config(
    page_title="AI Disease Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS — Dark + Green Theme ───────────────────────
st.markdown("""
<style>
    /* ─── Google Font ─── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ─── Global Dark Background ─── */
    .stApp {
        background: #0b0e11 !important;
        color: #e6e6e6 !important;
    }

    /* ─── Sidebar Dark Styling ─── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1318 0%, #121920 50%, #0f1318 100%) !important;
        border-right: 1px solid rgba(0, 210, 106, 0.15);
    }
    section[data-testid="stSidebar"] * {
        color: #c4cdd6 !important;
    }
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h1 {
        color: #00d26a !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(0, 210, 106, 0.2) !important;
    }
    section[data-testid="stSidebar"] table {
        border-collapse: collapse;
    }
    section[data-testid="stSidebar"] th {
        background: rgba(0, 210, 106, 0.12) !important;
        color: #00d26a !important;
        font-weight: 600;
        border-bottom: 1px solid rgba(0, 210, 106, 0.25) !important;
    }
    section[data-testid="stSidebar"] td {
        border-bottom: 1px solid rgba(255,255,255,0.05) !important;
    }
    section[data-testid="stSidebar"] strong {
        color: #e0e0e0 !important;
    }

    /* ─── Main Header ─── */
    .main-header {
        text-align: center;
        padding: 2.5rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d26a 0%, #00ff85 40%, #a0ffcf 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
        letter-spacing: -0.5px;
        text-shadow: 0 0 40px rgba(0, 210, 106, 0.3);
    }
    .main-header p {
        font-size: 1.1rem;
        color: #7a8a9a;
        margin-top: 0;
        letter-spacing: 0.3px;
    }

    /* ─── SDG Badge ─── */
    .sdg-badge {
        display: inline-block;
        background: linear-gradient(135deg, #00d26a 0%, #00a854 100%);
        color: #0b0e11 !important;
        padding: 0.45rem 1.4rem;
        border-radius: 25px;
        font-size: 0.82rem;
        font-weight: 700;
        margin: 0.6rem auto;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        box-shadow: 0 0 20px rgba(0, 210, 106, 0.25), 0 4px 15px rgba(0, 210, 106, 0.15);
    }

    /* ─── Custom Divider ─── */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #00d26a, #00ff85, #00d26a, transparent);
        border: none;
        border-radius: 2px;
        margin: 2rem 0;
        opacity: 0.5;
    }

    /* ─── Dark Card Styling ─── */
    .input-card {
        background: linear-gradient(145deg, #131920 0%, #0f1419 100%);
        border: 1px solid rgba(0, 210, 106, 0.12);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
    }
    .input-card h3 {
        color: #00d26a !important;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    /* ─── Number Input Dark Styling ─── */
    .stNumberInput label {
        color: #a0aec0 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    .stNumberInput input {
        background: #1a2028 !important;
        border: 1px solid rgba(0, 210, 106, 0.2) !important;
        color: #e6e6e6 !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
    }
    .stNumberInput input:focus {
        border-color: #00d26a !important;
        box-shadow: 0 0 0 2px rgba(0, 210, 106, 0.15) !important;
    }
    /* Step buttons in number input */
    .stNumberInput button {
        background: rgba(0, 210, 106, 0.1) !important;
        border: 1px solid rgba(0, 210, 106, 0.2) !important;
        color: #00d26a !important;
    }
    .stNumberInput button:hover {
        background: rgba(0, 210, 106, 0.2) !important;
    }

    /* ─── Result Boxes ─── */
    .result-box {
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 1rem 0;
        animation: fadeIn 0.7s ease-out;
        backdrop-filter: blur(10px);
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .result-positive {
        background: linear-gradient(145deg, rgba(220, 38, 38, 0.12) 0%, rgba(185, 28, 28, 0.08) 100%);
        border: 1px solid rgba(239, 68, 68, 0.35);
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.1), inset 0 1px 0 rgba(239, 68, 68, 0.1);
    }
    .result-negative {
        background: linear-gradient(145deg, rgba(0, 210, 106, 0.1) 0%, rgba(0, 168, 84, 0.06) 100%);
        border: 1px solid rgba(0, 210, 106, 0.3);
        box-shadow: 0 0 30px rgba(0, 210, 106, 0.1), inset 0 1px 0 rgba(0, 210, 106, 0.1);
    }
    .result-box h2 {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .result-box p {
        font-size: 0.95rem;
        color: #9ca3af;
    }
    .result-positive h2 { color: #ef4444; }
    .result-negative h2 { color: #00d26a; }

    /* ─── Probability Bar ─── */
    .prob-container {
        background: #1a2028;
        border-radius: 12px;
        height: 32px;
        margin: 0.6rem 0;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .prob-fill {
        height: 100%;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #0b0e11;
        font-weight: 700;
        font-size: 0.82rem;
        transition: width 1s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    }

    /* ─── Feature Info Cards ─── */
    .feature-info {
        background: #131920;
        border: 1px solid rgba(0, 210, 106, 0.1);
        border-radius: 14px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .feature-info h4 {
        color: #7a8a9a;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }

    /* ─── About / SDG Section ─── */
    .about-section {
        background: linear-gradient(145deg, rgba(0, 210, 106, 0.06) 0%, rgba(0, 168, 84, 0.03) 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(0, 210, 106, 0.15);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
    }
    .about-section h3 {
        color: #00d26a !important;
        font-weight: 700;
    }
    .about-section p, .about-section li {
        color: #a0aec0;
        line-height: 1.7;
    }
    .about-section strong {
        color: #e0e0e0;
    }

    /* ─── Heading Overrides ─── */
    h1, h2, h3, h4, h5, h6 {
        color: #e6e6e6 !important;
    }
    .stMarkdown h3 {
        color: #00d26a !important;
        font-weight: 700;
    }

    /* ─── Paragraph & Text ─── */
    p, span, li, td, th, label, .stMarkdown {
        color: #c4cdd6;
    }

    /* ─── Metrics Dark Styling ─── */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #131920 0%, #171e27 100%);
        border: 1px solid rgba(0, 210, 106, 0.12);
        border-radius: 16px;
        padding: 1rem 1.2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    [data-testid="stMetricLabel"] {
        color: #7a8a9a !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }
    [data-testid="stMetricValue"] {
        color: #00d26a !important;
        font-weight: 800 !important;
        font-size: 1.8rem !important;
    }

    /* ─── Button Styling ─── */
    .stButton > button {
        background: linear-gradient(135deg, #00d26a 0%, #00a854 100%) !important;
        color: #0b0e11 !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 0 25px rgba(0, 210, 106, 0.3), 0 4px 15px rgba(0, 210, 106, 0.2) !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #00ff85 0%, #00d26a 100%) !important;
        box-shadow: 0 0 35px rgba(0, 210, 106, 0.45), 0 6px 20px rgba(0, 210, 106, 0.3) !important;
        transform: translateY(-2px) !important;
    }
    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* ─── Expander Dark Styling ─── */
    .streamlit-expanderHeader {
        background: #131920 !important;
        border: 1px solid rgba(0, 210, 106, 0.12) !important;
        border-radius: 12px !important;
        color: #c4cdd6 !important;
        font-weight: 600 !important;
    }
    .streamlit-expanderContent {
        background: #0f1419 !important;
        border: 1px solid rgba(0, 210, 106, 0.08) !important;
        border-top: none !important;
    }
    details {
        background: #131920 !important;
        border: 1px solid rgba(0, 210, 106, 0.12) !important;
        border-radius: 12px !important;
    }
    details summary {
        color: #c4cdd6 !important;
        font-weight: 600 !important;
    }
    details > div {
        background: #0f1419 !important;
    }

    /* ─── Dataframe / Table Dark ─── */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
    [data-testid="stDataFrame"] > div {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ─── Stat Labels ─── */
    .stat-card {
        background: linear-gradient(145deg, #131920 0%, #171e27 100%);
        border: 1px solid rgba(0, 210, 106, 0.12);
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        transition: all 0.3s ease;
    }
    .stat-card:hover {
        border-color: rgba(0, 210, 106, 0.3);
        box-shadow: 0 0 25px rgba(0, 210, 106, 0.1), 0 6px 25px rgba(0, 0, 0, 0.3);
        transform: translateY(-3px);
    }
    .stat-card .stat-value {
        font-size: 2rem;
        font-weight: 800;
        color: #00d26a;
        margin-bottom: 0.2rem;
    }
    .stat-card .stat-label {
        font-size: 0.82rem;
        color: #7a8a9a;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ─── Glow Text Utility ─── */
    .glow-green {
        color: #00d26a;
        text-shadow: 0 0 10px rgba(0, 210, 106, 0.3);
    }

    /* ─── Live Badge ─── */
    .live-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(0, 210, 106, 0.1);
        border: 1px solid rgba(0, 210, 106, 0.25);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #00d26a;
        margin-left: 0.5rem;
    }
    .live-badge::before {
        content: '';
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #00d26a;
        animation: pulse-dot 1.5s infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(0, 210, 106, 0.5); }
        50% { opacity: 0.7; box-shadow: 0 0 0 6px rgba(0, 210, 106, 0); }
    }

    /* ─── Section Title ─── */
    .section-title {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 1.5rem 0 0.8rem 0;
    }
    .section-title h3 {
        margin: 0;
        color: #e6e6e6 !important;
        font-weight: 700;
    }
    .section-title .accent-line {
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(0, 210, 106, 0.3), transparent);
    }

    /* ─── Footer ─── */
    .footer-styled {
        text-align: center;
        padding: 2.5rem 0 1.5rem 0;
        color: #4a5568;
        font-size: 0.82rem;
        border-top: 1px solid rgba(0, 210, 106, 0.08);
        margin-top: 2rem;
    }
    .footer-styled a {
        color: #00d26a;
        text-decoration: none;
    }

    /* ─── Hide Streamlit Branding ─── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        background: rgba(11, 14, 17, 0.8) !important;
        backdrop-filter: blur(10px);
    }

    /* ─── Scrollbar ─── */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #0b0e11;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(0, 210, 106, 0.25);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(0, 210, 106, 0.4);
    }

    /* ─── Top Nav Bar ─── */
    .top-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: linear-gradient(135deg, #0f1318 0%, #151c24 100%);
        border: 1px solid rgba(0, 210, 106, 0.1);
        border-radius: 16px;
        padding: 0.8rem 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .top-nav .nav-brand {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .top-nav .nav-brand .logo-icon {
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, #00d26a, #00a854);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        box-shadow: 0 0 15px rgba(0, 210, 106, 0.25);
    }
    .top-nav .nav-brand span {
        font-weight: 700;
        color: #e6e6e6;
        font-size: 1.1rem;
        letter-spacing: -0.3px;
    }
    .top-nav .nav-links {
        display: flex;
        gap: 1.5rem;
        align-items: center;
    }
    .top-nav .nav-links a {
        color: #7a8a9a;
        text-decoration: none;
        font-size: 0.88rem;
        font-weight: 500;
        transition: color 0.2s;
        padding: 0.3rem 0;
        border-bottom: 2px solid transparent;
    }
    .top-nav .nav-links a:hover,
    .top-nav .nav-links a.active {
        color: #00d26a;
        border-bottom-color: #00d26a;
    }
    .top-nav .nav-status {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .top-nav .nav-status .status-pill {
        background: rgba(0, 210, 106, 0.1);
        border: 1px solid rgba(0, 210, 106, 0.2);
        color: #00d26a;
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* ─── Glass Card ─── */
    .glass-card {
        background: linear-gradient(145deg, rgba(19, 25, 32, 0.9) 0%, rgba(15, 20, 25, 0.95) 100%);
        border: 1px solid rgba(0, 210, 106, 0.1);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 0.8rem 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(0, 210, 106, 0.25);
        box-shadow: 0 8px 40px rgba(0, 0, 0, 0.3), 0 0 20px rgba(0, 210, 106, 0.05);
    }

    /* ─── Input Section Header ─── */
    .input-section-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 0.5rem;
    }
    .input-section-header .icon-box {
        width: 36px;
        height: 36px;
        background: rgba(0, 210, 106, 0.1);
        border: 1px solid rgba(0, 210, 106, 0.2);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
    }
    .input-section-header span {
        color: #e6e6e6;
        font-weight: 600;
        font-size: 1rem;
    }

    /* ─── Animated Background Gradient ─── */
    .bg-glow {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 400px;
        background: radial-gradient(ellipse at 50% 0%, rgba(0, 210, 106, 0.06) 0%, transparent 70%);
        pointer-events: none;
        z-index: 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Background Glow ───────────────────────────────────────
st.markdown('<div class="bg-glow"></div>', unsafe_allow_html=True)

# ── Load Model and Scaler ────────────────────────────────
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model.pkl")
    scaler_path = os.path.join(base_dir, "scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler


@st.cache_data
def load_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "dataset", "diabetes.csv")
    return pd.read_csv(data_path)


model, scaler = load_model()
df = load_dataset()

# ── Header ────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 1.5rem 0 0.5rem 0;">
    <h1 style="font-size:2rem; font-weight:800; color:#00d26a; margin-bottom:0.2rem;">🩺 HealthPredict AI</h1>
    <p style="color:#7a8a9a; font-size:0.9rem; margin:0;">Diabetes Risk Prediction · SDG 3 — Good Health & Well-being</p>
</div>
""", unsafe_allow_html=True)

# ── Check Model ───────────────────────────────────────────
if model is None or scaler is None:
    st.error("⚠️ Model not found! Run `python train.py` first.")
    st.stop()

# ══════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Dashboard", "🔍 Predict", "📊 Analytics", "📂 Dataset"])


# ─────────────────────────────────────────────────────────
# TAB 1 — DASHBOARD
# ─────────────────────────────────────────────────────────
with tab1:
    st.markdown("#### Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples", df.shape[0])
    c2.metric("Features", df.shape[1] - 1)
    c3.metric("Diabetic", int((df['Outcome'] == 1).sum()))
    c4.metric("Non-Diabetic", int((df['Outcome'] == 0).sum()))

    st.markdown("---")
    st.markdown("**About this project**")
    st.markdown("""
    This AI system uses **Machine Learning** to predict diabetes risk from clinical parameters.

    | | |
    |---|---|
    | 🎯 Goal | Early detection of diabetes |
    | 📊 Dataset | Pima Indians Diabetes (768 records) |
    | 🤖 Model | Decision Tree / Logistic Regression |
    | 🌍 SDG | SDG 3 — Good Health and Well-being |

    **How to use:** Go to the **Predict** tab, enter patient values, and click Predict.  
    Use **Analytics** for charts and **Dataset** to explore the raw data.
    """)


# ─────────────────────────────────────────────────────────
# TAB 2 — PREDICT
# ─────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### Enter Patient Data")

    col1, col2 = st.columns(2)
    with col1:
        pregnancies    = st.number_input("🤰 Pregnancies",          min_value=0,   max_value=20,  value=1,    step=1)
        glucose        = st.number_input("🍬 Glucose (mg/dL)",      min_value=0,   max_value=300, value=120,  step=1)
        blood_pressure = st.number_input("💉 Blood Pressure (mmHg)",min_value=0,   max_value=200, value=70,   step=1)
        skin_thickness = st.number_input("📏 Skin Thickness (mm)",  min_value=0,   max_value=100, value=20,   step=1)
    with col2:
        insulin        = st.number_input("💊 Insulin (mu U/ml)",    min_value=0,   max_value=900, value=80,   step=1)
        bmi            = st.number_input("⚖️ BMI (kg/m²)",          min_value=0.0, max_value=70.0,value=25.0, step=0.1, format="%.1f")
        dpf            = st.number_input("🧬 Diabetes Pedigree",    min_value=0.0, max_value=3.0, value=0.5,  step=0.01,format="%.3f")
        age            = st.number_input("🎂 Age (years)",           min_value=1,   max_value=120, value=30,   step=1)

    predict_btn = st.button("🔍 Predict Now", type="primary", width="stretch")

    if predict_btn:
        feature_names = ['Pregnancies','Glucose','BloodPressure',
                         'SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
        input_df = pd.DataFrame([[pregnancies, glucose, blood_pressure,
                                   skin_thickness, insulin, bmi, dpf, age]],
                                 columns=feature_names)
        # pass as numpy array — avoids feature name mismatch warning
        input_scaled = scaler.transform(input_df.values)
        prediction   = model.predict(input_scaled)[0]
        probability  = model.predict_proba(input_scaled)[0]

        prob_d  = probability[1] * 100
        prob_nd = probability[0] * 100

        st.markdown("---")
        if prediction == 1:
            st.error(f"⚠️ **Diabetes Detected** — Risk probability: **{prob_d:.1f}%**\n\nPlease consult a healthcare professional.")
        else:
            st.success(f"✅ **No Diabetes Detected** — Healthy probability: **{prob_nd:.1f}%**\n\nMaintain a healthy lifestyle!")

        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"**🔴 Diabetes Risk: {prob_d:.1f}%**")
            st.progress(float(probability[1]))
            st.markdown(f"**🟢 No Diabetes: {prob_nd:.1f}%**")
            st.progress(float(probability[0]))
        with r2:
            summary = pd.DataFrame({
                "Parameter": ['Pregnancies','Glucose','BloodPressure',
                               'SkinThickness','Insulin','BMI',
                               'DiabetesPedigreeFunction','Age'],
                "Value": [pregnancies, glucose, blood_pressure,
                          skin_thickness, insulin, bmi, dpf, age]
            })
            st.dataframe(summary, hide_index=True, width="stretch")


# ─────────────────────────────────────────────────────────
# TAB 3 — ANALYTICS
# ─────────────────────────────────────────────────────────
with tab3:
    st.markdown("#### Data Analytics")

    chart_type = st.selectbox("Select chart", [
        "Correlation Heatmap",
        "Outcome Distribution",
        "Feature Distributions",
        "Glucose vs BMI"
    ])

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#0b0e11')
    ax.set_facecolor('#131920')
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a3540')
    ax.tick_params(colors='#7a8a9a')
    ax.title.set_color('#e6e6e6')
    ax.xaxis.label.set_color('#7a8a9a')
    ax.yaxis.label.set_color('#7a8a9a')

    if chart_type == "Correlation Heatmap":
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="Greens",
                    linewidths=0.5, linecolor='#1a2028', ax=ax,
                    annot_kws={"size": 8, "color": "#e6e6e6"})
        ax.set_title("Feature Correlation Matrix")
        ax.tick_params(colors='#7a8a9a', labelsize=8)

    elif chart_type == "Outcome Distribution":
        counts = df['Outcome'].value_counts()
        bars = ax.bar(['No Diabetes', 'Diabetes'], counts.values,
                      color=['#00d26a', '#ef4444'], width=0.4)
        ax.set_title("Outcome Distribution")
        ax.set_ylabel("Count")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(int(bar.get_height())), ha='center', color='#e6e6e6', fontsize=11)

    elif chart_type == "Feature Distributions":
        plt.close(fig)
        features = ['Glucose', 'BMI', 'Age', 'Insulin']
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        fig.patch.set_facecolor('#0b0e11')
        for i, feat in enumerate(features):
            axes[i].set_facecolor('#131920')
            df[df['Outcome']==0][feat].plot(kind='hist', ax=axes[i], alpha=0.6,
                                             color='#00d26a', bins=20, label='No Diabetes')
            df[df['Outcome']==1][feat].plot(kind='hist', ax=axes[i], alpha=0.6,
                                             color='#ef4444', bins=20, label='Diabetes')
            axes[i].set_title(feat, color='#e6e6e6', fontsize=10)
            axes[i].tick_params(colors='#7a8a9a', labelsize=8)
            for spine in axes[i].spines.values():
                spine.set_edgecolor('#2a3540')
        axes[0].legend(fontsize=8, facecolor='#131920', labelcolor='#c4cdd6')
        plt.tight_layout()

    elif chart_type == "Glucose vs BMI":
        diabetic     = df[df['Outcome'] == 1]
        non_diabetic = df[df['Outcome'] == 0]
        ax.scatter(non_diabetic['Glucose'], non_diabetic['BMI'],
                   alpha=0.5, color='#00d26a', s=20, label='No Diabetes')
        ax.scatter(diabetic['Glucose'], diabetic['BMI'],
                   alpha=0.5, color='#ef4444', s=20, label='Diabetes')
        ax.set_xlabel("Glucose")
        ax.set_ylabel("BMI")
        ax.set_title("Glucose vs BMI")
        ax.legend(facecolor='#131920', labelcolor='#c4cdd6', fontsize=9)

    st.pyplot(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────
# TAB 4 — DATASET
# ─────────────────────────────────────────────────────────
with tab4:
    st.markdown("#### Dataset Explorer")

    d1, d2, d3 = st.columns(3)
    d1.metric("Rows", df.shape[0])
    d2.metric("Columns", df.shape[1])
    d3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.markdown("**Raw Data (first 20 rows)**")
    st.dataframe(df.head(20), hide_index=True, width="stretch")

    st.markdown("**Statistical Summary**")
    st.dataframe(df.describe().round(2), width="stretch")

    st.markdown("**Feature Descriptions**")
    st.markdown("""
    | Feature | Description | Unit |
    |---|---|---|
    | Pregnancies | Number of pregnancies | count |
    | Glucose | Plasma glucose (2h oral test) | mg/dL |
    | BloodPressure | Diastolic blood pressure | mm Hg |
    | SkinThickness | Triceps skinfold thickness | mm |
    | Insulin | 2-Hour serum insulin | mu U/ml |
    | BMI | Body mass index | kg/m² |
    | DiabetesPedigreeFunction | Genetic diabetes risk score | — |
    | Age | Age of patient | years |
    | Outcome | 1 = Diabetic, 0 = Not Diabetic | — |
    """)

# ── Footer ────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#4a5568; font-size:0.78rem;'>"
    "🩺 HealthPredict AI · SDG 3 · Built with Streamlit & scikit-learn · © 2026</p>",
    unsafe_allow_html=True
)
