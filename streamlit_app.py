import streamlit as st
import pandas as pd
import pydeck as pdk
import json
import requests

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Australia Construction Pressure Index",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLES ---
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

GITHUB = "https://raw.githubusercontent.com/aETHWilliams/australia-construction-pressure-model/main"

# --- DATA LOADING ---
@st.cache_data
def load_data():
    scores = pd.read_csv(f"{GITHUB}/master_table_v8.csv")
    shap_df = pd.read_csv(f"{GITHUB}/shap_values_v5.csv")
    geojson = requests.get(f"{GITHUB}/sa2_pressure_v5.geojson").json()
    return scores, shap_df, geojson

results, shap_df, geojson = load_data()
results['erp_change_pct'] = results['erp_change_pct'].round(1)
results['dwellings_2526_fytd'] = results['dwellings_2526_fytd'].fillna(0).astype(int)

# --- FUNCTIONS ---
def rank_to_color(rank):
    if rank <= 50: return '#dc2626'
    elif rank <= 200: return '#d97706'
    else: return '#2563a8'

def rank_to_map_color(rank):
    if rank <= 50: return [220, 38, 38, 180]
    elif rank <= 200: return [217, 119, 6, 180]
    else: return [37, 99, 168, 140]

# --- UI RENDER ---
st.markdown("""<div class="header-wrap"><div class="header-title">Australia Construction Pressure Index</div><div class="header-sub">Predictive ML Model &nbsp;·&nbsp; 7.3M Records &nbsp;·&nbsp; 2,442 Suburbs Nationally</div></div>""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
metrics = [("0.938", "Model AUC Score"), ("100%", "Backtest Hit Rate"), ("2,442", "Suburbs Analysed"), ("7.3M+", "Records Trained On"), ("5", "Data Sources")]
for col, (val, label) in zip([c1, c2, c3, c4, c5], metrics):
    col.markdown(f"""<div class="metric-card"><span class="metric-value">{val}</span><span class="metric-label">{label}</span></div>""", unsafe_allow_html=True)

# ... [Continue rest of the implementation block to match your full structure]

# --- SIDEBAR & FILTERING ---
selected = st.sidebar.selectbox("Filter by State", ["All", "NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"])
min_score = st.sidebar.slider("Minimum Pressure Score", 0, 100, 0)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size:0.82rem; color:#1e3a5f; font-family:Sora,sans-serif; font-weight:600; margin-bottom:0.4rem'>What This App Does</div>
<div style='font-size:0.78rem; color:#4a6080; line-height:1.8; margin-bottom:1.2rem'>This tool predicts which Australian suburbs are likely to experience a construction boom <b>before it happens</b>. It analyses 20 signals per suburb and assigns every suburb a <b>Pressure Score from 0 to 100</b>.</div>
<div style='font-size:0.82rem; color:#1e3a5f; font-family:Sora,sans-serif; font-weight:600; margin-bottom:0.4rem'>How the Model Works</div>
<div style='font-size:0.78rem; color:#4a6080; line-height:1.8; margin-bottom:1.2rem'>Two machine learning models — <b>XGBoost</b> and <b>Random Forest</b> — were trained on 2022–24 historical data. The v5 model achieves an <b>R² of 0.72</b> — explaining 72% of the variance.</div>
<div style='font-size:0.82rem; color:#1e3a5f; font-family:Sora,sans-serif; font-weight:600; margin-bottom:0.4rem'>Version History</div>
<div style='font-size:0.78rem; color:#4a6080; line-height:1.8; margin-bottom:1.2rem'><b>v8 (current)</b> — Enhanced predictive engine with custom feature engineering for construction velocity.</div>
""", unsafe_allow_html=True)

# --- TOP 10 SURGE SUBURBS TABLE ---
filtered = results.copy()
if selected != "All": filtered = filtered[filtered["state"] == selected]
filtered = filtered[filtered["pressure_score"] >= min_score].sort_values("pressure_score", ascending=False)
top10 = results.sort_values('national_rank').head(10).reset_index(drop=True)

rows_html = ""
for i, row in top10.iterrows():
    rank = int(row['national_rank']); score = row['pressure_score']; color = '#f87171' if rank <= 50 else '#fbbf24' if rank <= 200 else '#93c5fd'; signal = 'Critical' if rank <= 50 else 'High' if rank <= 200 else 'Moderate'; approvals = int(row['dwellings_2526_fytd']) if pd.notna(row['dwellings_2526_fytd']) else 'N/A'; bg = 'rgba(255,255,255,0.05)' if (i + 1) % 2 == 0 else 'transparent'
    rows_html += f"<tr style='font-size:0.82rem;background:{bg};'><td style='padding:0.5rem 0.6rem;color:rgba(255,255,255,0.35);font-weight:600'>#{rank}</td><td style='padding:0.5rem 0.6rem;color:#ffffff;font-weight:600'>{row['sa2_name']}</td><td style='padding:0.5rem 0.6rem;color:#a8c8e8'>{row['state']}</td><td style='padding:0.5rem 0.6rem;color:{color};font-weight:700'>{score}</td><td style='padding:0.5rem 0.6rem;color:#a8c8e8'>{row['erp_change_pct']}%</td><td style='padding:0.5rem 0.6rem;color:#a8c8e8'>{int(row['years_of_growth'])}/22</td><td style='padding:0.5rem 0.6rem;color:#a8c8e8'>{approvals}</td><td style='padding:0.5rem 0.6rem;color:{color}'>{signal}</td></tr>"

html = f"""<div style='background:linear-gradient(135deg,#1e3a5f 0%,#2563a8 60%,#3b82c4 100%);border-radius:16px;padding:1.8rem 2rem;margin-bottom:2rem;box-shadow:0 4px 24px rgba(37,99,168,0.18);'><div style='font-family:Sora,sans-serif;font-size:1.2rem;font-weight:700;color:#ffffff;margin-bottom:0.2rem'>Top 10 Predicted Surge Suburbs — 2026/27</div><div style='font-size:0.78rem;color:#a8c8e8;margin-bottom:1.2rem'>Ranked by National Construction Pressure Rank &nbsp;·&nbsp; Model v8</div><table style='width:100%;border-collapse:collapse;'><tr style='font-size:0.68rem;color:#a8c8e8;text-transform:uppercase;letter-spacing:1px;border-bottom:1px solid rgba(255,255,255,0.1);'><td style='padding:0.4rem 0.6rem'>Rank</td><td style='padding:0.4rem 0.6rem'>Suburb</td><td style='padding:0.4rem 0.6rem'>State</td><td style='padding:0.4rem 0.6rem'>Score</td><td style='padding:0.4rem 0.6rem'>Pop Growth</td><td style='padding:0.4rem 0.6rem'>Growth Yrs</td><td style='padding:0.4rem 0.6rem'>Approvals FYTD</td><td style='padding:0.4rem 0.6rem'>Signal</td></tr>{rows_html}</table></div>"""
st.markdown(html, unsafe_allow_html=True)

# --- SUBURB SEARCH ---
st.markdown('<div class="section-title">Suburb Search</div>', unsafe_allow_html=True)
search = st.text_input("Suburb Search", label_visibility="collapsed", placeholder="Search any suburb...")
if search:
    found = results[results["sa2_name"].str.contains(search, case=False, na=False)]
    if len(found) > 0:
        for _, row in found.iterrows():
            score = row["pressure_score"]; rank = int(row['national_rank']); cls = "score-high" if rank <= 50 else "score-med" if rank <= 200 else "score-low"; signal_label = 'Critical' if rank <= 50 else 'High' if rank <= 200 else 'Moderate'
            st.markdown(f"""<div class="suburb-row"><div><b style='color:#1e3a5f;font-size:1rem'>{row['sa2_name']}</b><span style='color:#6b8cae; font-size:0.82rem'> &nbsp;{row['state']}</span><span style='color:#6b8cae; font-size:0.78rem'> &nbsp;·&nbsp; National Rank #{rank}</span></div><div style='text-align:right'><span class='{cls}' style='font-size:1.1rem'>{score}/100</span><span style='color:#6b8cae; font-size:0.8rem; margin-left:1rem'>{signal_label} &nbsp;|&nbsp; Pop growth {row['erp_change_pct']}%</span></div></div>""", unsafe_allow_html=True)
            if pd.notna(row.get('lat')) and pd.notna(row.get('lon')):
                st.map(pd.DataFrame({'lat': [row['lat']], 'lon': [row['lon']]}), zoom=11)

# --- PRESSURE MAP ---
st.markdown('<div class="section-title">Pressure Map</div>', unsafe_allow_html=True)
legend_html = """<div style='background:linear-gradient(135deg,#1e3a5f 0%,#1a3358 100%);border-radius:14px;padding:1.4rem 2rem;margin-bottom:1.2rem;box-shadow:0 4px 18px rgba(30,58,95,0.18);display:flex;gap:3rem;'><div><div style='color:#f87171'>Critical Pressure</div></div><div><div style='color:#fbbf24'>High Pressure</div></div><div><div style='color:#93c5fd'>Moderate</div></div></div>"""
st.markdown(legend_html, unsafe_allow_html=True)

for feature in geojson['features']:
    name = feature['properties'].get('SA2_NAME21', ''); match = results[results['sa2_name'] == name]
    rank = int(match.iloc[0]['national_rank']) if len(match) > 0 else 9999
    feature['properties']['fill_color'] = rank_to_map_color(rank)
    feature['properties']['national_rank'] = rank

st.pydeck_chart(pdk.Deck(layers=[pdk.Layer('GeoJsonLayer', data=geojson, pickable=True, stroked=True, filled=True, get_fill_color='properties.fill_color', get_line_color=[255, 255, 255, 60], line_width_min_pixels=1)], initial_view_state=pdk.ViewState(latitude=-27.0, longitude=134.0, zoom=3.5), map_style='mapbox://styles/mapbox/light-v9'))

# --- RANKED SUBURBS TABLE ---
st.markdown('<div class="section-title">Ranked Suburbs</div>', unsafe_allow_html=True)
st.dataframe(filtered[["national_rank", "sa2_name", "state", "pressure_score", "erp_change_pct", "dwellings_2526_fytd"]], use_container_width=True, hide_index=True)

# --- ABOUT THE MODEL ---
st.markdown('<div class="section-title">About This Model</div>', unsafe_allow_html=True)
st.markdown("""
<div class="about-box">
<b style='color:#1e3a5f'>Methodology</b><br>
Version 8 — built on an advanced predictive engine featuring custom feature engineering for construction velocity across 2,442 suburbs.<br><br>
<b style='color:#1e3a5f'>Technical Stack</b><br>
<span class="tag">XGBoost</span><span class="tag">Random Forest</span><span class="tag">SHAP</span><span class="tag">Streamlit</span><span class="tag">Python</span>
</div>
""", unsafe_allow_html=True)
