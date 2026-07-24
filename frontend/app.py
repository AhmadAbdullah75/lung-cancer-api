"""
Lung Cancer Risk Predictor - Frontend
Ahmad Abdullah

Custom dark glassmorphism UI over the FastAPI /predict endpoint.
Run with: streamlit run app.py
"""

import os
import requests
import streamlit as st
import plotly.graph_objects as go

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Pulmo AI — Risk Intelligence",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# GLOBAL STYLE
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(56, 189, 248, 0.14), transparent 40%),
        radial-gradient(circle at 85% 0%, rgba(168, 85, 247, 0.14), transparent 45%),
        radial-gradient(circle at 50% 100%, rgba(16, 185, 129, 0.10), transparent 50%),
        #05070d;
    color: #e6edf3;
}

#MainMenu, footer, header {visibility: hidden;}

/* ---- Hero ---- */
.hero {
    text-align: center;
    padding: 2.2rem 1rem 1.4rem 1rem;
}
.hero-badge {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    background: rgba(56, 189, 248, 0.08);
    border: 1px solid rgba(56, 189, 248, 0.35);
    color: #7dd3fc;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    line-height: 1.1;
    margin: 0;
    background: linear-gradient(120deg, #f8fafc 20%, #7dd3fc 55%, #c4b5fd 90%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    color: #8b98a9;
    font-size: 1rem;
    max-width: 560px;
    margin: 0.9rem auto 0 auto;
    line-height: 1.55;
}

/* ---- Glass card ---- */
.glass-card {
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.3rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #7dd3fc;
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::before {
    content: '';
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #7dd3fc;
    box-shadow: 0 0 8px #7dd3fc;
}

/* ---- Sliders ---- */
div[data-testid="stSlider"] label p {
    font-size: 0.86rem !important;
    color: #c9d3de !important;
    font-weight: 500;
}
div[data-baseweb="slider"] > div > div {
    background: linear-gradient(90deg, #38bdf8, #a855f7) !important;
}
div[data-testid="stSlider"] [data-testid="stTickBar"] { display: none; }

/* ---- Selectbox / radio ---- */
div[data-testid="stSelectbox"] label p, div[role="radiogroup"] label p {
    color: #c9d3de !important;
    font-size: 0.86rem !important;
}

/* ---- Submit button ---- */
div[data-testid="stFormSubmitButton"] button {
    width: 100%;
    background: linear-gradient(120deg, #0ea5e9, #8b5cf6);
    border: none;
    border-radius: 14px;
    padding: 0.85rem 0;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    letter-spacing: 0.02em;
    color: white;
    box-shadow: 0 8px 24px rgba(139, 92, 246, 0.35);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
div[data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(139, 92, 246, 0.5);
}

/* ---- Result banner ---- */
.result-banner {
    text-align: center;
    padding: 1.6rem 1rem;
    border-radius: 20px;
    margin: 1rem 0 1.4rem 0;
    font-weight: 700;
    font-size: 1.5rem;
    letter-spacing: 0.02em;
}
.result-low {
    background: rgba(16, 185, 129, 0.10);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #6ee7b7;
}
.result-medium {
    background: rgba(245, 158, 11, 0.10);
    border: 1px solid rgba(245, 158, 11, 0.4);
    color: #fcd34d;
}
.result-high {
    background: rgba(239, 68, 68, 0.10);
    border: 1px solid rgba(239, 68, 68, 0.4);
    color: #fca5a5;
}

/* ---- Prob rows ---- */
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.65rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
}
.prob-label { width: 78px; color: #8b98a9; }
.prob-track {
    flex: 1;
    height: 8px;
    background: rgba(255,255,255,0.06);
    border-radius: 999px;
    overflow: hidden;
}
.prob-fill { height: 100%; border-radius: 999px; }
.prob-value { width: 52px; text-align: right; color: #c9d3de; }

/* ---- Footer ---- */
.footer {
    text-align: center;
    margin-top: 2.4rem;
    padding-top: 1.2rem;
    border-top: 1px solid rgba(255,255,255,0.06);
    color: #4b5563;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.04em;
}
.footer span { color: #6b7280; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# HERO
# ----------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-badge">Voting Ensemble · RF + SVM + LR</div>
    <h1 class="hero-title">Pulmo AI</h1>
    <p class="hero-sub">
        A clinical exposure &amp; symptom analyzer built on a replicated
        machine learning framework. Adjust the factors below to see how
        risk level shifts in real time.
    </p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# FORM
# ----------------------------------------------------------------------------
GROUPS = {
    "Environmental Exposure": [
        ("Air Pollution", "Air Pollution"),
        ("Dust Allergy", "Dust Allergy"),
        ("OccuPational Hazards", "Occupational Hazards"),
        ("Passive Smoker", "Passive Smoking"),
    ],
    "Lifestyle Factors": [
        ("Alcohol use", "Alcohol Use"),
        ("Smoking", "Smoking"),
        ("Balanced Diet", "Balanced Diet"),
        ("Obesity", "Obesity"),
    ],
    "Genetic & Chronic Background": [
        ("Genetic Risk", "Genetic Risk"),
        ("chronic Lung Disease", "Chronic Lung Disease"),
        ("Frequent Cold", "Frequent Cold"),
    ],
    "Physical Symptoms": [
        ("Chest Pain", "Chest Pain"),
        ("Coughing of Blood", "Coughing of Blood"),
        ("Dry Cough", "Dry Cough"),
        ("Fatigue", "Fatigue"),
        ("Weight Loss", "Weight Loss"),
        ("Shortness of Breath", "Shortness of Breath"),
        ("Wheezing", "Wheezing"),
        ("Swallowing Difficulty", "Swallowing Difficulty"),
        ("Clubbing of Finger Nails", "Clubbing of Finger Nails"),
        ("Snoring", "Snoring"),
    ],
}

with st.form("risk_form"):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Patient Profile</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 10, 100, 40)
    with c2:
        gender_label = st.selectbox("Gender", ["Male", "Female"])
        gender = 1 if gender_label == "Male" else 2
    st.markdown('</div>', unsafe_allow_html=True)

    values = {}
    for group_name, fields in GROUPS.items():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="section-label">{group_name}</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, (key, label) in enumerate(fields):
            with cols[i % 2]:
                values[key] = st.slider(label, 1, 9, 3, key=key)
        st.markdown('</div>', unsafe_allow_html=True)

    submitted = st.form_submit_button("Analyze Risk")

# ----------------------------------------------------------------------------
# RESULT
# ----------------------------------------------------------------------------
if submitted:
    payload = {"Age": age, "Gender": gender, **values}

    try:
        with st.spinner("Running inference through the voting ensemble..."):
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=15)

        if resp.status_code == 200:
            result = resp.json()
            pred = result["prediction"]
            probs = {
                "Low": result["probability_low"],
                "Medium": result["probability_medium"],
                "High": result["probability_high"],
            }

            css_class = {"Low": "result-low", "Medium": "result-medium", "High": "result-high"}[pred]
            icon = {"Low": "✅", "Medium": "⚠️", "High": "🔴"}[pred]
            st.markdown(
                f'<div class="result-banner {css_class}">{icon}&nbsp;&nbsp;{pred} Risk</div>',
                unsafe_allow_html=True,
            )

            # Gauge chart
            gauge_color = {"Low": "#34d399", "Medium": "#fbbf24", "High": "#f87171"}[pred]
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probs[pred] * 100,
                number={"suffix": "%", "font": {"color": "#e6edf3", "family": "Space Grotesk"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#4b5563", "tickfont": {"color": "#6b7280"}},
                    "bar": {"color": gauge_color},
                    "bgcolor": "rgba(255,255,255,0.03)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 40], "color": "rgba(255,255,255,0.03)"},
                        {"range": [40, 70], "color": "rgba(255,255,255,0.05)"},
                        {"range": [70, 100], "color": "rgba(255,255,255,0.07)"},
                    ],
                },
            ))
            fig.update_layout(
                height=240,
                margin=dict(l=20, r=20, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#e6edf3"},
            )
            st.plotly_chart(fig, use_container_width=True)

            # Probability breakdown
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Class Probabilities</div>', unsafe_allow_html=True)
            colors = {"Low": "#34d399", "Medium": "#fbbf24", "High": "#f87171"}
            for label, val in probs.items():
                st.markdown(f"""
                <div class="prob-row">
                    <div class="prob-label">{label}</div>
                    <div class="prob-track">
                        <div class="prob-fill" style="width:{val*100:.1f}%; background:{colors[label]};"></div>
                    </div>
                    <div class="prob-value">{val*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.caption(
                "This is a statistical model output based on symptom and exposure scores, "
                "not a clinical diagnosis. Consult a physician for medical concerns."
            )
        else:
            st.error(f"API error {resp.status_code}: {resp.text}")

    except requests.exceptions.ConnectionError:
        st.error(f"Could not reach the API at {API_URL}. Is it running?")

# ----------------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------------
st.markdown("""
<div class="footer">
    Voting Ensemble (RF · SVM · LR) &middot; SMOTE-balanced &middot; RF feature selection
    <br><span>Ahmad Abdullah</span>
</div>
""", unsafe_allow_html=True)