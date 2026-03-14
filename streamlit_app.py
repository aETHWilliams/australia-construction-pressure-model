import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Australia Construction Pressure Index",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Sora:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #f0f4f8; color: #1a2332; }
    .main { background-color: #f0f4f8; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }
    .header-wrap { background: linear-gradient(135deg, #1e3a5f 0%, #2563a8 60%, #3b82c4 100%); border-radius: 16px; padding: 2.5rem 3rem; margin-bottom: 2rem; box-shadow: 0 4px 24px rgba(37,99,168,0.18); }
    .header-title { font-family: 'Sora', sans-serif; font-size: 2.4rem; font-weight: 700; color: #ffffff; letter-spacing: -0.5px; line-height: 1.15; margin-bottom: 0.4rem; }
    .header-sub { font-size: 0.9rem; color: #a8c8e8; font-weight: 300; letter-spacing: 0.8px; text-transform: uppercase; }
    .metric-card { background: #ffffff; border: 1px solid #dbe8f5; border-radius: 12px; padding: 1.2rem 1rem; text-align: center; box-shadow: 0 2px 8px rgba(37,99,168,0.07); }
    .metric-value { font-family: 'Sora', sans-serif; font-size: 1.8rem; font-weight: 700; color: #2563a8; display: block; }
    .metric-label { font-size: 0.7rem; color: #6b8cae; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.2rem; }
    .section-title { font-family: 'Sora', sans-serif; font-size: 1.2rem; font-weight: 600; color: #1e3a5f; border-bottom: 2px solid #dbe8f5; padding-bottom: 0.5rem; margin-bottom: 1.2rem; margin-top: 2rem; }
    .suburb-row { background: #ffffff; border: 1px solid #dbe8f5; border-radius: 10px; padding: 0.9rem 1.2rem; margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 1px 4px rgba(37,99,168,0.05); }
    .score-high { color: #dc2626; font-weight: 600; }
    .score-med  { color: #d97706; font-weight: 600; }
    .score-low  { color: #2563a8; font-weight: 600; }
    .stat-card { background: #ffffff; border: 1px solid #dbe8f5; border-radius: 10px; padding: 1.2rem 1.5rem; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(37,99,168,0.07); }
    div[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #dbe8f5; }
    .about-box { background: #ffffff; border: 1px solid #dbe8f5; border-left: 4px solid #2563a8; border-radius: 10px; padding: 1.5rem; font-size: 0.88rem; color: #4a6080; line-height: 1.9; box-shadow: 0 2px 8px rgba(37,99,168,0.05); }
    .stTextInput input { background: #ffffff !important; border: 1px solid #dbe8f5 !important; color: #1a2332 !important; border-radius: 8px !important; }
    .tag { display: inline-block; background: #e8f0fb; color: #2563a8; font-size: 0.72rem; font-weight: 500; padding: 0.2rem 0.6rem; border-radius: 20px; margin-right: 0.3rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    return pd.read_csv("aus_pressure_scores_v4.csv")


results = load_data()

# Header
st.markdown("""
<div class="header-wrap">
    <div class="header-title">Australia Construction Pressure Index</div>
    <div class="header-sub">Predictive ML Model &nbsp;·&nbsp; 7.3M Records &nbsp;·&nbsp; 2,442 Suburbs Nationally</div>
</div>
""", unsafe_allow_html=True)

# Metrics
c1, c2, c3, c4, c5 = st.columns(5)
metrics = [
    ("0.938", "Model AUC Score"),
    ("100%", "Backtest Hit Rate"),
    ("2,442", "Suburbs Analysed"),
    ("7.3M+", "Records Trained On"),
    ("5", "Data Sources"),
]
for col, (val, label) in zip([c1, c2, c3, c4, c5], metrics):
    col.markdown(f"""
    <div class="metric-card">
        <span class="metric-value">{val}</span>
        <span class="metric-label">{label}</span>
    </div>""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("""
<div style='font-family:Sora,sans-serif;font-size:1.1rem;font-weight:600;color:#1e3a5f;padding:0.5rem 0 1rem'>Filters</div>
""", unsafe_allow_html=True)

states = ["All"] + sorted(results["state"].unique().tolist())
selected = st.sidebar.selectbox("State", states)
min_score = st.sidebar.slider("Minimum Pressure Score", 0, 100, 75)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size:0.78rem; color:#6b8cae; line-height:2'>
<b style='color:#1e3a5f'>Data Sources</b><br>
ABS Building Approvals 2022-26<br>
ABS Regional Population 2023-24<br>
ABS Population History 2001-2024<br>
SEIFA Socioeconomic Index 2021<br>
ABS Approvals 2025-26 FYTD
</div>""", unsafe_allow_html=True)

# Filter
filtered = results.copy()
if selected != "All":
    filtered = filtered[filtered["state"] == selected]
filtered = filtered[filtered["pressure_score"] >= min_score]
filtered = filtered.sort_values("pressure_score", ascending=False)

# Search
st.markdown('<div class="section-title">Suburb Search</div>', unsafe_allow_html=True)
search = st.text_input("Suburb Search", label_visibility="collapsed", placeholder="Search any suburb — e.g. Ripley, Byford, Sunbury...")
if search:
    found = results[results["sa2_name"].str.contains(search, case=False, na=False)]
    if len(found) > 0:
        for _, row in found.iterrows():
            score = row["pressure_score"]
            cls = "score-high" if score >= 99 else "score-med" if score >= 90 else "score-low"
            st.markdown(f"""
            <div class="suburb-row">
                <div>
                    <b style='color:#1e3a5f;font-size:1rem'>{row['sa2_name']}</b>
                    <span style='color:#6b8cae; font-size:0.82rem'> &nbsp;{row['state']}</span>
                </div>
                <div style='text-align:right'>
                    <span class='{cls}' style='font-size:1.1rem'>{score}/100</span>
                    <span style='color:#6b8cae; font-size:0.8rem; margin-left:1rem'>
                        Pop growth {row['erp_change_pct']}%
                        &nbsp;|&nbsp;
                        {int(row['years_of_growth'])}/22 consecutive growth years
                    </span>
                </div>
            </div>""", unsafe_allow_html=True)

            # Show detailed stat card and zoom map to suburb
            if pd.notna(row.get('lat')) and pd.notna(row.get('lon')):
                score_color = '#dc2626' if score >= 99 else '#d97706' if score >= 90 else '#2563a8'
                st.markdown(f"""
                <div class="stat-card">
                    <div style='font-family:Sora,sans-serif;font-size:1rem;font-weight:600;color:#1e3a5f;margin-bottom:0.8rem'>
                        {row['sa2_name']} — Full Stats
                    </div>
                    <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;'>
                        <div><span style='font-size:0.72rem;color:#6b8cae;text-transform:uppercase;letter-spacing:1px'>Pressure Score</span><br>
                            <span style='font-size:1.4rem;font-weight:700;color:{score_color}'>{score}/100</span></div>
                        <div><span style='font-size:0.72rem;color:#6b8cae;text-transform:uppercase;letter-spacing:1px'>Pop Growth</span><br>
                            <span style='font-size:1.4rem;font-weight:700;color:#1e3a5f'>{row['erp_change_pct']}%</span></div>
                        <div><span style='font-size:0.72rem;color:#6b8cae;text-transform:uppercase;letter-spacing:1px'>20yr Growth</span><br>
                            <span style='font-size:1.4rem;font-weight:700;color:#1e3a5f'>{round(row['growth_20yr']*100,1)}%</span></div>
                        <div><span style='font-size:0.72rem;color:#6b8cae;text-transform:uppercase;letter-spacing:1px'>Consecutive Growth Yrs</span><br>
                            <span style='font-size:1.4rem;font-weight:700;color:#1e3a5f'>{int(row['years_of_growth'])}/22</span></div>
                        <div><span style='font-size:0.72rem;color:#6b8cae;text-transform:uppercase;letter-spacing:1px'>2025-26 Approvals FYTD</span><br>
                            <span style='font-size:1.4rem;font-weight:700;color:#1e3a5f'>{int(row['dwellings_2526_fytd']) if pd.notna(row['dwellings_2526_fytd']) else "N/A"}</span></div>
                        <div><span style='font-size:0.72rem;color:#6b8cae;text-transform:uppercase;letter-spacing:1px'>State</span><br>
                            <span style='font-size:1.4rem;font-weight:700;color:#1e3a5f'>{row['state']}</span></div>
                    </div>
                </div>""", unsafe_allow_html=True)

                # Zoom map to this suburb
                suburb_map = pd.DataFrame({'lat': [row['lat']], 'lon': [row['lon']]})
                st.map(suburb_map, latitude=row['lat'], longitude=row['lon'], zoom=11)

    else:
        st.markdown("<span style='color:#6b8cae'>No suburb found. Try a different name.</span>", unsafe_allow_html=True)

# Map
st.markdown('<div class="section-title">Pressure Map</div>', unsafe_allow_html=True)

k1, k2, k3 = st.columns(3)
k1.markdown("""
<div style='background:#fff1f1;border:1px solid #fca5a5;border-left:4px solid #dc2626;border-radius:8px;padding:0.8rem 1rem'>
<span style='font-size:0.72rem;color:#6b8cae;text-transform:uppercase;letter-spacing:1px'>Score 99–100</span><br>
<span style='font-size:1rem;font-weight:700;color:#dc2626'>🔴 Critical Pressure</span><br>
<span style='font-size:0.78rem;color:#4a6080'>Top 25% of construction activity highly likely. Act now.</span>
</div>""", unsafe_allow_html=True)

k2.markdown("""
<div style='background:#fffbeb;border:1px solid #fcd34d;border-left:4px solid #d97706;border-radius:8px;padding:0.8rem 1rem'>
<span style='font-size:0.72rem;color:#6b8cae;text-transform:uppercase;letter-spacing:1px'>Score 90–98</span><br>
<span style='font-size:1rem;font-weight:700;color:#d97706'>🟡 High Pressure</span><br>
<span style='font-size:0.78rem;color:#4a6080'>Strong signals of upcoming surge. Worth monitoring closely.</span>
</div>""", unsafe_allow_html=True)

k3.markdown("""
<div style='background:#eff6ff;border:1px solid #93c5fd;border-left:4px solid #2563a8;border-radius:8px;padding:0.8rem 1rem'>
<span style='font-size:0.72rem;color:#6b8cae;text-transform:uppercase;letter-spacing:1px'>Score below 90</span><br>
<span style='font-size:1rem;font-weight:700;color:#2563a8'>🔵 Moderate / Low</span><br>
<span style='font-size:0.78rem;color:#4a6080'>Some growth signals present but surge less likely near-term.</span>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    "<span style='font-size:0.82rem;color:#6b8cae'>Top 500 highest-pressure suburbs shown.</span>",
    unsafe_allow_html=True
)

map_data = results.dropna(subset=['lat', 'lon']).sort_values('pressure_score', ascending=False).head(500)
st.map(map_data[['lat', 'lon']], latitude=-27.0, longitude=134.0, zoom=3)

# Table
st.markdown(
    f'<div class="section-title">Ranked Suburbs'
    f'<span style="font-family:Inter;font-size:0.88rem;color:#6b8cae;font-weight:400">'
    f' &nbsp;·&nbsp; {selected} &nbsp;·&nbsp; Score >= {min_score}'
    f' &nbsp;·&nbsp; {len(filtered):,} results</span></div>',
    unsafe_allow_html=True
)

display = filtered[[
    "sa2_name", "state", "pressure_score",
    "erp_change_pct", "growth_20yr",
    "years_of_growth", "dwellings_2526_fytd"
]].copy()
display.columns = [
    "Suburb", "State", "Pressure Score",
    "Pop Growth %", "20yr Growth",
    "Consecutive Growth Years", "2025-26 Approvals FYTD"
]
display["20yr Growth"] = (display["20yr Growth"] * 100).round(1).astype(str) + "%"
display["Pressure Score"] = display["Pressure Score"].round(1)
st.dataframe(display, use_container_width=True, height=480, hide_index=True)

# State chart
st.markdown('<div class="section-title">High Pressure Suburbs by State</div>', unsafe_allow_html=True)
state_counts = (
    results[results["pressure_score"] >= 75]
    .groupby("state").size()
    .reset_index(name="High Pressure Suburbs")
    .sort_values("High Pressure Suburbs", ascending=False)
)
st.bar_chart(state_counts.set_index("state"))

# About
st.markdown('<div class="section-title">About This Model</div>', unsafe_allow_html=True)
st.markdown("""
<div class="about-box">
<b style='color:#1e3a5f'>Methodology</b><br>
Trained on 2022-23 and 2023-24 historical data to predict 2024-25 construction surges.
Features include population growth rates, 20-year momentum, building approval history,
socioeconomic indices (SEIFA), and 2025-26 forward approval signals.<br><br>
<b style='color:#1e3a5f'>Validation</b><br>
Backtested against actual 2024-25 ABS data. Top 20 predictions all confirmed correct —
100% hit rate compared to 25% from random selection.<br><br>
<span class="tag">XGBoost</span>
<span class="tag">Random Forest</span>
<span class="tag">AUC 0.938</span>
<span class="tag">20 Features</span>
<span class="tag">scikit-learn</span>
<span class="tag">Streamlit</span>
<span class="tag">Python</span>
</div>
""", unsafe_allow_html=True)
