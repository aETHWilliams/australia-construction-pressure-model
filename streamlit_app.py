import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Australia Construction Pressure Index",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #0f1117;
        color: #e8e8e8;
    }
    .main { background-color: #0f1117; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .header-title {
        font-family: 'DM Serif Display', serif;
        font-size: 3rem;
        font-weight: 400;
        color: #ffffff;
        letter-spacing: -0.5px;
        line-height: 1.1;
        margin-bottom: 0.25rem;
    }
    .header-sub {
        font-size: 0.95rem;
        color: #8a8a9a;
        font-weight: 300;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #1a1d27;
        border: 1px solid #2a2d3a;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-value {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        color: #c8a96e;
        display: block;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #8a8a9a;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .section-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.4rem;
        color: #ffffff;
        border-bottom: 1px solid #2a2d3a;
        padding-bottom: 0.5rem;
        margin-bottom: 1.2rem;
        margin-top: 2rem;
    }
    .score-high { color: #e05c5c; font-weight: 500; }
    .score-med  { color: #e08c3a; font-weight: 500; }
    .score-low  { color: #c8a96e; font-weight: 500; }
    .suburb-row {
        background: #1a1d27;
        border: 1px solid #2a2d3a;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.4rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    div[data-testid="stSidebar"] {
        background-color: #13161f;
        border-right: 1px solid #2a2d3a;
    }
    .stDataFrame { border: 1px solid #2a2d3a; border-radius: 8px; }
    hr { border-color: #2a2d3a; }
    .about-box {
        background: #1a1d27;
        border: 1px solid #2a2d3a;
        border-left: 3px solid #c8a96e;
        border-radius: 8px;
        padding: 1.5rem;
        font-size: 0.85rem;
        color: #8a8a9a;
        line-height: 1.8;
    }
    .stTextInput input {
        background: #1a1d27 !important;
        border: 1px solid #2a2d3a !important;
        color: #e8e8e8 !important;
        border-radius: 6px !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    return pd.read_csv("aus_pressure_scores_v4.csv")


results = load_data()

COORDS = {
    'Ripley': (-27.67, 152.82),
    'Yarrabilba': (-27.79, 153.08),
    'Chambers Flat - Logan Reserve': (-27.73, 153.09),
    'Greenbank - North Maclean': (-27.75, 153.02),
    'Flagstone (West) - New Beith': (-27.79, 152.96),
    'Beaudesert': (-27.99, 152.86),
    'Redbank Plains': (-27.62, 152.86),
    'Palm Beach': (-28.10, 153.46),
    'Jimboomba - Glenlogan': (-27.83, 153.02),
    'Boronia Heights - Park Ridge': (-27.69, 153.02),
    'Caboolture - South': (-27.09, 152.95),
    'Beechboro': (-31.85, 115.93),
    'Byford': (-32.22, 116.02),
    'Baldivis - North': (-32.32, 115.83),
    'Baldivis - South': (-32.38, 115.83),
    'The Vines': (-31.72, 115.97),
    'Mandurah - North': (-32.51, 115.73),
    'Casuarina - Wandi': (-32.19, 115.88),
    'Armadale - Wungong - Brookdale': (-32.15, 116.01),
    'Piara Waters - Forrestdale': (-32.12, 115.92),
    'Wellard (West) - Bertram': (-32.26, 115.84),
    'Beaconsfield - Officer': (-38.07, 145.46),
    'Sunbury - South': (-37.59, 144.71),
    'Sunbury': (-37.58, 144.73),
    'Greenvale - Bulla': (-37.62, 144.89),
    'Lara': (-38.02, 144.40),
    'Mickleham - Yuroke': (-37.60, 144.90),
    'Pakenham - North West': (-38.06, 145.45),
    'Grovedale - Mount Duneed': (-38.22, 144.33),
    'Koo Wee Rup': (-38.20, 145.49),
    'Heidelberg West': (-37.76, 145.04),
    'Mount Barker': (-35.07, 138.86),
    'Victor Harbor': (-35.55, 138.62),
    'Gawler - South': (-34.62, 138.74),
    'Virginia - Waterloo Corner': (-34.63, 138.57),
    'Albion Park - Macquarie Pass': (-34.57, 150.80),
    'Picton - Tahmoor - Buxton': (-34.18, 150.61),
    'Thornton - Millers Forest': (-32.79, 151.64),
    'Kingswood - Werrington': (-33.75, 150.73),
    'Kurri Kurri - Abermain': (-32.82, 151.48),
    'Morisset - Cooranbong': (-33.10, 151.49),
    'Port Macquarie - West': (-31.44, 152.88),
    'Albury - East': (-36.07, 146.94),
}

st.markdown('<div class="header-title">Australia Construction<br>Pressure Index</div>', unsafe_allow_html=True)
st.markdown('<div class="header-sub">Predictive ML Model — 7.3M Records — 2,442 Suburbs Nationally</div>', unsafe_allow_html=True)

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

st.sidebar.markdown("### Filters")
states = ["All"] + sorted(results["state"].unique().tolist())
selected = st.sidebar.selectbox("State", states)
min_score = st.sidebar.slider("Minimum Pressure Score", 0, 100, 75)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size:0.75rem; color:#8a8a9a; line-height:1.8'>
<b style='color:#c8a96e'>Data Sources</b><br>
ABS Building Approvals 2022-26<br>
ABS Regional Population 2023-24<br>
ABS Population History 2001-2024<br>
SEIFA Socioeconomic Index 2021<br>
ABS Approvals 2025-26 FYTD
</div>""", unsafe_allow_html=True)

filtered = results.copy()
if selected != "All":
    filtered = filtered[filtered["state"] == selected]
filtered = filtered[filtered["pressure_score"] >= min_score]
filtered = filtered.sort_values("pressure_score", ascending=False)

st.markdown('<div class="section-title">Suburb Search</div>', unsafe_allow_html=True)
search = st.text_input("", placeholder="Type any suburb name...")
if search:
    found = results[results["sa2_name"].str.contains(search, case=False, na=False)]
    if len(found) > 0:
        for _, row in found.iterrows():
            score = row["pressure_score"]
            cls = "score-high" if score >= 99 else "score-med" if score >= 90 else "score-low"
            st.markdown(f"""
            <div class="suburb-row">
                <div>
                    <b style='color:#ffffff'>{row['sa2_name']}</b>
                    <span style='color:#8a8a9a; font-size:0.8rem'> — {row['state']}</span>
                </div>
                <div style='text-align:right'>
                    <span class='{cls}'>{score}/100</span>
                    <span style='color:#8a8a9a; font-size:0.8rem; margin-left:1rem'>
                        Pop growth {row['erp_change_pct']}% &nbsp;|&nbsp;
                        {int(row['years_of_growth'])}/22 growth years
                    </span>
                </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:#8a8a9a'>No suburb found.</span>", unsafe_allow_html=True)

st.markdown('<div class="section-title">Pressure Map</div>', unsafe_allow_html=True)

map_data = results.merge(
    pd.DataFrame([(k, v[0], v[1]) for k, v in COORDS.items()],
                 columns=['sa2_name', 'lat', 'lon']),
    on='sa2_name', how='inner'
)

m = folium.Map(location=[-27.0, 134.0], zoom_start=4, tiles='CartoDB dark_matter')

for _, row in map_data.iterrows():
    score = row['pressure_score']
    colour = '#e05c5c' if score >= 99 else '#e08c3a' if score >= 90 else '#c8a96e'
    radius = 6 + (score / 100) * 10
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=radius,
        color=colour,
        fill=True,
        fill_opacity=0.85,
        popup=folium.Popup(
            f"<b>{row['sa2_name']}</b> ({row['state']})<br>"
            f"Pressure Score: {score}/100<br>"
            f"Pop Growth: {row['erp_change_pct']}%<br>"
            f"Growth Years: {int(row['years_of_growth'])}/22<br>"
            f"20yr Growth: {round(row['growth_20yr']*100,1)}%",
            max_width=220
        ),
        tooltip=f"{row['sa2_name']} - {score}/100"
    ).add_to(m)

st_folium(m, width=None, height=480, returned_objects=[])

st.markdown(
    f'<div class="section-title">Ranked Suburbs — {selected} '
    f'<span style="font-family:DM Sans;font-size:0.9rem;color:#8a8a9a">'
    f'({len(filtered):,} results)</span></div>',
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

st.markdown('<div class="section-title">Distribution by State</div>', unsafe_allow_html=True)
state_counts = (
    results[results["pressure_score"] >= 75]
    .groupby("state").size()
    .reset_index(name="High Pressure Suburbs")
    .sort_values("High Pressure Suburbs", ascending=False)
)
st.bar_chart(state_counts.set_index("state"))

st.markdown('<div class="section-title">About This Model</div>', unsafe_allow_html=True)
st.markdown("""
<div class="about-box">
<b style='color:#c8a96e'>Methodology</b><br>
Trained on 2022-23 and 2023-24 data to predict 2024-25 construction surges.
Features include population growth rates, 20-year momentum, building approval
history, socioeconomic indices, and 2025-26 forward signals.<br><br>
<b style='color:#c8a96e'>Validation</b><br>
Backtested against actual 2024-25 ABS data. Top 20 predictions confirmed
correct — 100% hit rate vs 25% from random selection.<br><br>
<b style='color:#c8a96e'>Models</b>&nbsp; XGBoost + Random Forest &nbsp;|&nbsp;
<b style='color:#c8a96e'>AUC</b>&nbsp; 0.938 &nbsp;|&nbsp;
<b style='color:#c8a96e'>Features</b>&nbsp; 20 &nbsp;|&nbsp;
<b style='color:#c8a96e'>Built with</b>&nbsp; Python, scikit-learn, XGBoost, Streamlit
</div>
""", unsafe_allow_html=True)
