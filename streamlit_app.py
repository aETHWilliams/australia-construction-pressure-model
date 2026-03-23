import streamlit as st
import pandas as pd
import pydeck as pdk
import requests
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Australia Construction Pressure Index",
    page_icon="None",
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
    .version-badge-v9 { display: inline-block; background: #fef3c7; color: #92400e; font-size: 0.7rem; font-weight: 600; padding: 0.15rem 0.5rem; border-radius: 20px; margin-left: 0.4rem; }
    .version-badge-v10 { display: inline-block; background: #dcfce7; color: #166534; font-size: 0.7rem; font-weight: 600; padding: 0.15rem 0.5rem; border-radius: 20px; margin-left: 0.4rem; }
</style>
""", unsafe_allow_html=True)

GITHUB = "https://raw.githubusercontent.com/aETHWilliams/australia-construction-pressure-model/main"

@st.cache_data(ttl=3600)
def load_data():
    scores = pd.read_csv(f"{GITHUB}/master_table_v9_app.csv")
    v10 = pd.read_csv(f"{GITHUB}/master_table_v10_ready.csv")
    geojson = requests.get(f"{GITHUB}/sa2_pressure_v5.geojson").json()
    return scores, v10, geojson

results, v10, geojson = load_data()

results['erp_change_pct'] = results['erp_change_pct'].round(1)
results['dwellings_2526_fytd'] = results['dwellings_2526_fytd'].fillna(0).astype(int)

# Parse SHAP drivers from v10 top_3_drivers column into separate columns
def parse_top_driver(drivers_str, n=0):
    try:
        parts = str(drivers_str).split(' | ')
        if n < len(parts):
            feat, val = parts[n].rsplit(':', 1)
            return feat.strip(), float(val.strip())
    except:
        pass
    return None, 0.0

def rank_to_color(rank):
    if rank <= 50:
        return '#dc2626'
    elif rank <= 200:
        return '#d97706'
    else:
        return '#2563a8'

def rank_to_map_color(rank):
    if rank <= 50:
        return [220, 38, 38, 180]
    elif rank <= 200:
        return [217, 119, 6, 180]
    else:
        return [37, 99, 168, 140]

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-wrap">
    <div class="header-title">Australia Construction Pressure Index</div>
    <div class="header-sub">Predictive ML Model &nbsp;·&nbsp; 7.3M Records &nbsp;·&nbsp; 2,442 Suburbs Nationally &nbsp;·&nbsp; Model v10 &nbsp;·&nbsp; By Ethan Williams</div>
</div>
""", unsafe_allow_html=True)

# ── Metrics ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
metrics = [
    ("0.923", "CV Spearman"),
    ("611", "High Pressure Suburbs"),
    ("2,442", "Suburbs Analysed"),
    ("88", "SA4 Regions Ranked"),
    ("7", "Data Sources"),
]
for col, (val, label) in zip([c1, c2, c3, c4, c5], metrics):
    col.markdown(f"""
    <div class="metric-card">
        <span class="metric-value">{val}</span>
        <span class="metric-label">{label}</span>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-top: 2rem'></div>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style='font-size:0.82rem; color:#1e3a5f; font-family:Sora,sans-serif; font-weight:600; margin-bottom:0.4rem'>What This App Does</div>
<div style='font-size:0.78rem; color:#4a6080; line-height:1.8; margin-bottom:1.2rem'>
This tool predicts which Australian suburbs are likely to experience a construction boom <b>before it happens</b>.
It analyses signals per suburb — population growth, building approval history, socioeconomic data,
pipeline backlogs, and forward-looking 2025–26 approvals — and assigns every suburb a <b>Pressure Score</b>.
</div>

<div style='font-size:0.82rem; color:#1e3a5f; font-family:Sora,sans-serif; font-weight:600; margin-bottom:0.4rem'>v10 Improvements</div>
<div style='font-size:0.78rem; color:#4a6080; line-height:1.8; margin-bottom:1.2rem'>
<b>Saturation Index</b> — fully built-out suburbs are now penalised using a dwellings-per-capita proxy, fixing overscoring of dense inner suburbs.<br><br>
<b>SHAP Score Decomposition</b> — every suburb's score is now broken into its component drivers, showing exactly why it ranked where it did.<br><br>
<b>SA4 Rollup</b> — suburb scores are aggregated to regional SA4 level, enabling macro-level analysis alongside suburb detail.
</div>

<div style='font-size:0.82rem; color:#1e3a5f; font-family:Sora,sans-serif; font-weight:600; margin-bottom:0.4rem'>Version History</div>
<div style='font-size:0.78rem; color:#4a6080; line-height:1.8; margin-bottom:1.2rem'>
<b>v1–v3</b> — Queensland only. Iterative feature engineering.<br><br>
<b>v4</b> — National model, 2,442 suburbs. XGBoost + Random Forest. AUC 0.938.<br><br>
<b>v5</b> — Switched to regression. SHAP explainability added.<br><br>
<b>v8</b> — Velocity-focused classification. Spearman 0.750.<br><br>
<b>v9</b> — Urban renewal blind spot fixed. Spearman 0.923. Inner-city precincts correctly identified.<br><br>
<b>v10 (current)</b> — Saturation index, SHAP decomposition, SA4 rollup.
</div>

<div style='border-top:1px solid #dbe8f5; padding-top:1rem'>
<div style='font-size:0.82rem; color:#1e3a5f; font-family:Sora,sans-serif; font-weight:600; margin-bottom:0.4rem'>Data Sources</div>
<div style='font-size:0.78rem; color:#6b8cae; line-height:2'>
ABS Building Approvals 2022–26<br>
ABS Regional Population 2023–24<br>
ABS Population History 2001–2024<br>
SEIFA Socioeconomic Index 2021<br>
QLD Lot Approvals 2023–26<br>
Residential Land Development Activity<br>
ABS Building Activity Table 80 (8752.0)
</div>
</div>
""", unsafe_allow_html=True)

# ── Tab Navigation ────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Top Rankings", 
    "v9 vs v10 Comparison", 
    "SA4 Regional View",
    "Suburb Search", 
    "Pressure Map"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TOP RANKINGS
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    # Version toggle
    version = st.radio("Model Version", ["v10 (Saturation Adjusted)", "v9 (Original)"], horizontal=True)
    
    if version == "v10 (Saturation Adjusted)":
        top10 = v10.sort_values('v10_rank').head(10).reset_index(drop=True)
        rank_col = 'v10_rank'
        score_col = 'v10_score'
        badge = '<span class="version-badge-v10">v10</span>'
    else:
        top10 = v10.sort_values('v9_rank').head(10).reset_index(drop=True)
        rank_col = 'v9_rank'
        score_col = 'v9_score'
        badge = '<span class="version-badge-v9">v9</span>'

    rows_html = ""
    for i, row in top10.iterrows():
        rank = int(row[rank_col])
        score = round(row[score_col], 2)
        v9r = int(row['v9_rank'])
        v10r = int(row['v10_rank'])
        shift = v9r - v10r
        if version == "v10 (Saturation Adjusted)" and shift != 0:
            arrow = f"<span style='color:#16a34a;font-size:0.75rem'>▲{abs(shift)}</span>" if shift > 0 else f"<span style='color:#dc2626;font-size:0.75rem'>▼{abs(shift)}</span>"
        else:
            arrow = ""
        color = '#f87171' if rank <= 50 else '#fbbf24' if rank <= 200 else '#93c5fd'
        signal = 'Critical' if rank <= 50 else 'High' if rank <= 200 else 'Moderate'
        sat = round(row.get('saturation_index', 0), 3)
        drivers = str(row.get('top_3_drivers', ''))
        short_driver = drivers.split(' | ')[0].split(':')[0].strip() if drivers else '—'
        bg = 'rgba(255,255,255,0.05)' if i % 2 == 0 else 'transparent'
        rows_html += (
            f"<tr style='font-size:0.82rem;background:{bg};'>"
            f"<td style='padding:0.5rem 0.6rem;color:rgba(255,255,255,0.35);font-weight:600'>#{rank} {arrow}</td>"
            f"<td style='padding:0.5rem 0.6rem;color:#ffffff;font-weight:600'>{row['sa2_name']}</td>"
            f"<td style='padding:0.5rem 0.6rem;color:#a8c8e8'>{row['state']}</td>"
            f"<td style='padding:0.5rem 0.6rem;color:{color};font-weight:700'>{score}</td>"
            f"<td style='padding:0.5rem 0.6rem;color:#a8c8e8'>{sat}</td>"
            f"<td style='padding:0.5rem 0.6rem;color:#a8c8e8;font-size:0.75rem'>{short_driver}</td>"
            f"<td style='padding:0.5rem 0.6rem;color:{color}'>{signal}</td>"
            f"</tr>"
        )

    html = (
        f"<div style='background:linear-gradient(135deg,#1e3a5f 0%,#2563a8 60%,#3b82c4 100%);"
        f"border-radius:16px;padding:1.8rem 2rem;margin-bottom:2rem;"
        f"box-shadow:0 4px 24px rgba(37,99,168,0.18);'>"
        f"<div style='font-family:Sora,sans-serif;font-size:1.2rem;font-weight:700;color:#ffffff;margin-bottom:0.2rem'>"
        f"Top 10 Predicted Surge Suburbs — 2026/27 {badge}</div>"
        f"<div style='font-size:0.78rem;color:#a8c8e8;margin-bottom:1.2rem'>"
        f"Ranked by National Construction Pressure Rank · Based on 31 ML Features · Model v10</div>"
        f"<table style='width:100%;border-collapse:collapse;'>"
        f"<tr style='font-size:0.68rem;color:#a8c8e8;text-transform:uppercase;letter-spacing:1px;"
        f"border-bottom:1px solid rgba(255,255,255,0.1);'>"
        f"<td style='padding:0.4rem 0.6rem'>Rank</td>"
        f"<td style='padding:0.4rem 0.6rem'>Suburb</td>"
        f"<td style='padding:0.4rem 0.6rem'>State</td>"
        f"<td style='padding:0.4rem 0.6rem'>Score</td>"
        f"<td style='padding:0.4rem 0.6rem'>Saturation</td>"
        f"<td style='padding:0.4rem 0.6rem'>Top Driver</td>"
        f"<td style='padding:0.4rem 0.6rem'>Signal</td>"
        f"</tr>{rows_html}</table></div>"
    )
    st.markdown(html, unsafe_allow_html=True)

    # Top 20 bar chart
    top20_v10 = v10.sort_values('v10_rank').head(20)
    top20_v9 = v10.sort_values('v9_rank').head(20)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top20_v10['sa2_name'] + ' (' + top20_v10['state'] + ')',
        x=top20_v10['v10_score'],
        orientation='h',
        name='v10 Score',
        marker_color='#2563a8',
        text=top20_v10['v10_score'].round(2),
        textposition='outside',
    ))
    fig.update_layout(
        title='Top 20 Suburbs — v10 Adjusted Score',
        xaxis_title='Composite Score',
        yaxis=dict(autorange='reversed'),
        height=520,
        plot_bgcolor='#f7faff',
        paper_bgcolor='#ffffff',
        font=dict(family='Inter', size=12, color='#1a2332'),
        margin=dict(l=20, r=60, t=50, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — v9 vs v10 COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">What Changed from v9 to v10</div>', unsafe_allow_html=True)

    components.html("""
    <link href='https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Sora:wght@400;600;700&display=swap' rel='stylesheet'>
    <div style='font-family:Inter,sans-serif;display:grid;grid-template-columns:1fr 1fr;gap:1.2rem;margin-bottom:1.5rem;'>
        <div style='background:#fef3c7;border:1px solid #fde68a;border-left:4px solid #d97706;border-radius:10px;padding:1.2rem 1.5rem;'>
            <div style='font-family:Sora,sans-serif;font-size:0.9rem;font-weight:700;color:#92400e;margin-bottom:0.6rem'>v9 — Activity Detector</div>
            <div style='font-size:0.8rem;color:#78350f;line-height:1.8;'>
                Ranked suburbs by raw construction activity volume.<br>
                Fully built-out inner suburbs scored too high.<br>
                No penalty for saturation.<br>
                Spearman 0.923 — excellent signal, but activity ≠ stress.
            </div>
        </div>
        <div style='background:#dcfce7;border:1px solid #bbf7d0;border-left:4px solid #16a34a;border-radius:10px;padding:1.2rem 1.5rem;'>
            <div style='font-family:Sora,sans-serif;font-size:0.9rem;font-weight:700;color:#166534;margin-bottom:0.6rem'>v10 — Stress Detector</div>
            <div style='font-size:0.8rem;color:#14532d;line-height:1.8;'>
                Saturation index penalises high-volume, decelerating suburbs.<br>
                SHAP decomposition explains every score.<br>
                SA4 rollup enables regional macro analysis.<br>
                Rankings reflect genuine demand/supply imbalance.
            </div>
        </div>
    </div>
    """, height=175)

    # Biggest rank movers
    v10_compare = v10.copy()
    v10_compare['rank_change'] = v10_compare['v9_rank'] - v10_compare['v10_rank']

    col_up, col_down = st.columns(2)

    with col_up:
        st.markdown('<div class="section-title">Biggest Risers (v9 to v10)</div>', unsafe_allow_html=True)
        risers = v10_compare.sort_values('rank_change', ascending=False).head(15)
        fig_up = go.Figure(go.Bar(
            x=risers['rank_change'],
            y=risers['sa2_name'] + ' (' + risers['state'] + ')',
            orientation='h',
            marker_color='#16a34a',
            text=[f"#{int(r['v9_rank'])} → #{int(r['v10_rank'])}" for _, r in risers.iterrows()],
            textposition='outside',
        ))
        fig_up.update_layout(
            xaxis_title='Rank Improvement',
            yaxis=dict(autorange='reversed'),
            height=420,
            plot_bgcolor='#f7faff',
            paper_bgcolor='#ffffff',
            font=dict(family='Inter', size=11, color='#1a2332'),
            margin=dict(l=10, r=80, t=20, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig_up, use_container_width=True)

    with col_down:
        st.markdown('<div class="section-title">Biggest Fallers (Saturation Penalised)</div>', unsafe_allow_html=True)
        fallers = v10_compare.sort_values('rank_change').head(15)
        fig_down = go.Figure(go.Bar(
            x=fallers['rank_change'].abs(),
            y=fallers['sa2_name'] + ' (' + fallers['state'] + ')',
            orientation='h',
            marker_color='#dc2626',
            text=[f"#{int(r['v9_rank'])} → #{int(r['v10_rank'])}" for _, r in fallers.iterrows()],
            textposition='outside',
        ))
        fig_down.update_layout(
            xaxis_title='Rank Drop',
            yaxis=dict(autorange='reversed'),
            height=420,
            plot_bgcolor='#f7faff',
            paper_bgcolor='#ffffff',
            font=dict(family='Inter', size=11, color='#1a2332'),
            margin=dict(l=10, r=80, t=20, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig_down, use_container_width=True)

    # Blind spots fixed
    st.markdown('<div class="section-title">Blind Spots Fixed in v9 (Now Correctly Identified)</div>', unsafe_allow_html=True)

    blind_spots = [
        {'Suburb': 'Footscray', 'State': 'VIC', 'v8 Rank': 'Outside top 200', 'v9 Rank': '#11', 'v10 Rank': None, 'Urban Renewal Score': 2.93},
        {'Suburb': 'Brunswick South', 'State': 'VIC', 'v8 Rank': 'Outside top 200', 'v9 Rank': '#6', 'v10 Rank': None, 'Urban Renewal Score': 1.40},
        {'Suburb': 'Wollongong East', 'State': 'NSW', 'v8 Rank': 'Outside top 200', 'v9 Rank': '#14', 'v10 Rank': None, 'Urban Renewal Score': 2.18},
        {'Suburb': 'Zetland', 'State': 'NSW', 'v8 Rank': 'Outside top 200', 'v9 Rank': '#93', 'v10 Rank': None, 'Urban Renewal Score': 3.13},
        {'Suburb': 'Rhodes', 'State': 'NSW', 'v8 Rank': 'Outside top 200', 'v9 Rank': '#127', 'v10 Rank': None, 'Urban Renewal Score': 2.80},
        {'Suburb': 'Kangaroo Point', 'State': 'QLD', 'v8 Rank': 'Outside top 200', 'v9 Rank': '#16', 'v10 Rank': None, 'Urban Renewal Score': 3.17},
        {'Suburb': 'Fremantle', 'State': 'WA', 'v8 Rank': 'Outside top 200', 'v9 Rank': '#5', 'v10 Rank': None, 'Urban Renewal Score': 2.39},
    ]

    for bs in blind_spots:
        match = v10[v10['sa2_name'].str.contains(bs['Suburb'], case=False, na=False)]
        if len(match) > 0:
            bs['v10 Rank'] = f"#{int(match.iloc[0]['v10_rank'])}"
        else:
            bs['v10 Rank'] = bs['v9 Rank']

    bs_df = pd.DataFrame(blind_spots)
    st.dataframe(bs_df, use_container_width=True, hide_index=True)

    # Scatter — v9 score vs v10 score coloured by saturation
    st.markdown('<div class="section-title">v9 Score vs v10 Score — Saturation Effect</div>', unsafe_allow_html=True)
    fig_scatter = px.scatter(
        v10,
        x='v9_score',
        y='v10_score',
        color='saturation_index',
        color_continuous_scale='RdYlGn_r',
        hover_name='sa2_name',
        hover_data={'state': True, 'v9_rank': True, 'v10_rank': True, 'saturation_index': ':.3f'},
        labels={'v9_score': 'v9 Score', 'v10_score': 'v10 Adjusted Score', 'saturation_index': 'Saturation'},
        title='Suburbs below the diagonal were penalised by the saturation index',
    )
    fig_scatter.add_shape(type='line', x0=v10['v9_score'].min(), y0=v10['v9_score'].min(),
                          x1=v10['v9_score'].max(), y1=v10['v9_score'].max(),
                          line=dict(color='#94a3b8', dash='dash'))
    fig_scatter.update_layout(
        height=480,
        plot_bgcolor='#f7faff',
        paper_bgcolor='#ffffff',
        font=dict(family='Inter', size=12, color='#1a2332'),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SA4 REGIONAL VIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">SA4 Regional Pressure Rankings</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#e8f0fb;border-radius:10px;padding:1rem 1.4rem;margin-bottom:1.2rem;font-size:0.85rem;color:#1e3a5f;line-height:1.7;'>
    SA4 regions are Statistical Area Level 4 — the macro geography used by the ABS. 
    Each region aggregates multiple suburbs. This view shows average pressure across all suburbs within each region,
    the proportion flagged as high pressure, and the single highest-scoring suburb in each region.
    </div>
    """, unsafe_allow_html=True)

    # Load SA4 rollup
    # Build SA4 rollup on the fly from v10
    sa4 = v10.groupby(['sa4_name', 'state']).agg(
        suburb_count=('sa2_name', 'count'),
        high_pressure_count=('high_pressure_v9', 'sum'),
        avg_composite_score=('v10_score', 'mean'),
        max_composite_score=('v10_score', 'max'),
        top_suburb=('sa2_name', lambda x: x.loc[v10.loc[x.index, 'v10_score'].idxmax()]),
    ).reset_index()
    sa4['sa4_rank'] = sa4['avg_composite_score'].rank(ascending=False).astype(int)
    sa4['high_pressure_pct'] = (sa4['high_pressure_count'] / sa4['suburb_count'] * 100).round(1)
    sa4 = sa4.sort_values('sa4_rank')

    # Top 20 SA4 bar chart
    top20_sa4 = sa4.sort_values('sa4_rank').head(20)

    fig_sa4 = go.Figure()
    fig_sa4.add_trace(go.Bar(
        y=top20_sa4['sa4_name'] + ' (' + top20_sa4['state'] + ')',
        x=top20_sa4['avg_composite_score'],
        orientation='h',
        marker=dict(
            color=top20_sa4['high_pressure_pct'],
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title='% High Pressure'),
        ),
        text=top20_sa4['top_suburb'],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Avg Score: %{x:.2f}<br>Top suburb: %{text}<extra></extra>',
    ))
    fig_sa4.update_layout(
        title='Top 20 SA4 Regions by Average Pressure Score',
        xaxis_title='Average Composite Score',
        yaxis=dict(autorange='reversed'),
        height=560,
        plot_bgcolor='#f7faff',
        paper_bgcolor='#ffffff',
        font=dict(family='Inter', size=12, color='#1a2332'),
        margin=dict(l=20, r=120, t=50, b=20),
    )
    st.plotly_chart(fig_sa4, use_container_width=True)

    # High pressure % by SA4
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-title">% High Pressure Suburbs by Region</div>', unsafe_allow_html=True)
        top15_pct = sa4.sort_values('high_pressure_pct', ascending=False).head(15)
        fig_pct = px.bar(
            top15_pct,
            x='high_pressure_pct',
            y='sa4_name',
            orientation='h',
            color='high_pressure_pct',
            color_continuous_scale='Reds',
            labels={'high_pressure_pct': '% High Pressure', 'sa4_name': 'SA4 Region'},
        )
        fig_pct.update_layout(
            height=420, plot_bgcolor='#f7faff', paper_bgcolor='#ffffff',
            font=dict(family='Inter', size=11), showlegend=False,
            yaxis=dict(autorange='reversed'),
            margin=dict(l=10, r=20, t=20, b=20),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_pct, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Suburb Count by SA4 Region</div>', unsafe_allow_html=True)
        top15_count = sa4.sort_values('suburb_count', ascending=False).head(15)
        fig_count = px.bar(
            top15_count,
            x='suburb_count',
            y='sa4_name',
            orientation='h',
            color='avg_composite_score',
            color_continuous_scale='Blues',
            labels={'suburb_count': 'Suburbs', 'sa4_name': 'SA4 Region'},
        )
        fig_count.update_layout(
            height=420, plot_bgcolor='#f7faff', paper_bgcolor='#ffffff',
            font=dict(family='Inter', size=11), showlegend=False,
            yaxis=dict(autorange='reversed'),
            margin=dict(l=10, r=20, t=20, b=20),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_count, use_container_width=True)

    # Full SA4 table
    st.markdown('<div class="section-title">Full SA4 Rankings Table</div>', unsafe_allow_html=True)
    sa4_display = sa4[['sa4_rank','sa4_name','state','avg_composite_score','high_pressure_pct','suburb_count','top_suburb']].copy()
    sa4_display.columns = ['Rank','SA4 Region','State','Avg Score','% High Pressure','Suburbs','Top Suburb']
    sa4_display['Avg Score'] = sa4_display['Avg Score'].round(2)
    st.dataframe(sa4_display, use_container_width=True, height=480, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SUBURB SEARCH
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Suburb Search</div>', unsafe_allow_html=True)
    search = st.text_input("Search", label_visibility="collapsed", placeholder="Search any suburb — e.g. Ripley, Footscray, Fremantle...")

    if search:
        found_v9 = results[results["sa2_name"].str.contains(search, case=False, na=False)]
        found_v10 = v10[v10['sa2_name'].str.contains(search, case=False, na=False)]

        if len(found_v9) > 0:
            for _, row in found_v9.iterrows():
                score = row["pressure_score"]
                rank_v9 = int(row['national_rank'])
                cls = "score-high" if rank_v9 <= 50 else "score-med" if rank_v9 <= 200 else "score-low"
                signal_label = 'Critical' if rank_v9 <= 50 else 'High' if rank_v9 <= 200 else 'Moderate'

                # Get v10 data for this suburb
                v10_row = found_v10[found_v10['sa2_name'] == row['sa2_name']]
                rank_v10 = int(v10_row.iloc[0]['v10_rank']) if len(v10_row) > 0 else rank_v9
                score_v10 = round(v10_row.iloc[0]['v10_score'], 2) if len(v10_row) > 0 else score
                sat_idx = round(v10_row.iloc[0]['saturation_index'], 3) if len(v10_row) > 0 else 0
                top_drivers = str(v10_row.iloc[0]['top_3_drivers']) if len(v10_row) > 0 else ''
                rank_shift = rank_v9 - rank_v10
                shift_html = ""
                if rank_shift > 0:
                    shift_html = f"<span style='color:#16a34a;font-size:0.8rem;margin-left:0.5rem'>▲ Rose {rank_shift} places in v10</span>"
                elif rank_shift < 0:
                    shift_html = f"<span style='color:#dc2626;font-size:0.8rem;margin-left:0.5rem'>▼ Fell {abs(rank_shift)} places in v10 (saturation)</span>"

                st.markdown(f"""
                <div class="suburb-row">
                    <div>
                        <b style='color:#1e3a5f;font-size:1rem'>{row['sa2_name']}</b>
                        <span style='color:#6b8cae; font-size:0.82rem'> &nbsp;{row['state']}</span>
                        <span style='color:#6b8cae; font-size:0.78rem'> &nbsp;·&nbsp; v9 Rank #{rank_v9} &nbsp;·&nbsp; v10 Rank #{rank_v10}</span>
                        {shift_html}
                    </div>
                    <div style='text-align:right'>
                        <span class='{cls}' style='font-size:1.1rem'>{score}/100</span>
                        <span style='color:#6b8cae; font-size:0.8rem; margin-left:1rem'>
                            {signal_label} &nbsp;|&nbsp; Pop growth {row['erp_change_pct']}%
                            &nbsp;|&nbsp; {int(row['years_of_growth'])}/22 growth years
                            &nbsp;|&nbsp; Saturation {sat_idx}
                        </span>
                    </div>
                </div>""", unsafe_allow_html=True)

                # SHAP decomposition chart
                if top_drivers and top_drivers != 'nan':
                    driver_parts = [d.strip() for d in top_drivers.split(' | ')]
                    driver_names, driver_vals = [], []
                    for d in driver_parts:
                        try:
                            name, val = d.rsplit(':', 1)
                            driver_names.append(name.strip())
                            driver_vals.append(float(val.strip()))
                        except:
                            pass

                    if driver_names:
                        colors = ['#16a34a' if v > 0 else '#dc2626' for v in driver_vals]
                        fig_shap = go.Figure(go.Bar(
                            x=driver_vals,
                            y=driver_names,
                            orientation='h',
                            marker_color=colors,
                            text=[f"{'+' if v > 0 else ''}{v:.3f}" for v in driver_vals],
                            textposition='outside',
                        ))
                        fig_shap.update_layout(
                            title=f"Why did {row['sa2_name']} score {score}/100? — SHAP Drivers",
                            xaxis_title='SHAP Value (positive = pushes score up)',
                            yaxis=dict(autorange='reversed'),
                            height=280,
                            plot_bgcolor='#f7faff',
                            paper_bgcolor='#ffffff',
                            font=dict(family='Inter', size=12, color='#1a2332'),
                            margin=dict(l=20, r=80, t=50, b=20),
                            showlegend=False,
                        )
                        st.plotly_chart(fig_shap, use_container_width=True)

                # Stats grid
                st.markdown(f"""
                <div class="stat-card">
                    <div style='font-family:Sora,sans-serif;font-size:1rem;font-weight:600;color:#1e3a5f;margin-bottom:0.8rem'>
                        {row['sa2_name']} — Full Stats
                    </div>
                    <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;'>
                        <div><span style='font-size:0.72rem;color:#6b8cae;text-transform:uppercase;letter-spacing:1px'>v9 Score</span><br>
                            <span style='font-size:1.3rem;font-weight:700;color:#d97706'>{score}/100</span></div>
                        <div><span style='font-size:0.72rem;color:#6b8cae;text-transform:uppercase;letter-spacing:1px'>v10 Score</span><br>
                            <span style='font-size:1.3rem;font-weight:700;color:#2563a8'>{score_v10}/100</span></div>
                        <div><span style='font-size:0.72rem;color:#6b8cae;text-transform:uppercase;letter-spacing:1px'>v9 Rank</span><br>
                            <span style='font-size:1.3rem;font-weight:700;color:#d97706'>#{rank_v9}</span></div>
                        <div><span style='font-size:0.72rem;color:#6b8cae;text-transform:uppercase;letter-spacing:1px'>v10 Rank</span><br>
                            <span style='font-size:1.3rem;font-weight:700;color:#2563a8'>#{rank_v10}</span></div>
                        <div><span style='font-size:0.72rem;color:#6b8cae;text-transform:uppercase;letter-spacing:1px'>Saturation Index</span><br>
                            <span style='font-size:1.3rem;font-weight:700;color:#1e3a5f'>{sat_idx}</span></div>
                        <div><span style='font-size:0.72rem;color:#6b8cae;text-transform:uppercase;letter-spacing:1px'>Pop Growth</span><br>
                            <span style='font-size:1.3rem;font-weight:700;color:#1e3a5f'>{row['erp_change_pct']}%</span></div>
                        <div><span style='font-size:0.72rem;color:#6b8cae;text-transform:uppercase;letter-spacing:1px'>20yr Growth</span><br>
                            <span style='font-size:1.3rem;font-weight:700;color:#1e3a5f'>{round(row['growth_20yr']*100,1)}%</span></div>
                        <div><span style='font-size:0.72rem;color:#6b8cae;text-transform:uppercase;letter-spacing:1px'>Growth Years</span><br>
                            <span style='font-size:1.3rem;font-weight:700;color:#1e3a5f'>{int(row['years_of_growth'])}/22</span></div>
                    </div>
                </div>""", unsafe_allow_html=True)

                if pd.notna(row.get('lat')) and pd.notna(row.get('lon')):
                    st.map(pd.DataFrame({'lat': [row['lat']], 'lon': [row['lon']]}),
                           latitude=row['lat'], longitude=row['lon'], zoom=11)
        else:
            st.markdown("<span style='color:#6b8cae'>No suburb found. Try a different name.</span>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PRESSURE MAP
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    legend_html = """
    <div style='background:linear-gradient(135deg,#1e3a5f 0%,#1a3358 100%);border-radius:14px;
    padding:1.4rem 2rem;margin-bottom:1.2rem;box-shadow:0 4px 18px rgba(30,58,95,0.18);
    display:flex;gap:3rem;align-items:flex-start;'>
        <div style='flex:1;border-left:3px solid #f87171;padding-left:1rem;'>
            <div style='font-size:0.68rem;color:#a8c8e8;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:0.3rem'>Top 50 Nationally</div>
            <div style='font-family:Sora,sans-serif;font-size:0.95rem;font-weight:700;color:#f87171;margin-bottom:0.3rem'>Critical Pressure</div>
            <div style='font-size:0.76rem;color:#cbd5e1;line-height:1.6;'>All signals align — sustained population growth, strong approval momentum, and consistent 20-year history.</div>
        </div>
        <div style='flex:1;border-left:3px solid #fbbf24;padding-left:1rem;'>
            <div style='font-size:0.68rem;color:#a8c8e8;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:0.3rem'>Rank 51–200</div>
            <div style='font-family:Sora,sans-serif;font-size:0.95rem;font-weight:700;color:#fbbf24;margin-bottom:0.3rem'>High Pressure</div>
            <div style='font-size:0.76rem;color:#cbd5e1;line-height:1.6;'>Most indicators elevated — strong population trend, above-average approvals, positive momentum.</div>
        </div>
        <div style='flex:1;border-left:3px solid #93c5fd;padding-left:1rem;'>
            <div style='font-size:0.68rem;color:#a8c8e8;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:0.3rem'>Rank 201+</div>
            <div style='font-family:Sora,sans-serif;font-size:0.95rem;font-weight:700;color:#93c5fd;margin-bottom:0.3rem'>Moderate / Low</div>
            <div style='font-size:0.76rem;color:#cbd5e1;line-height:1.6;'>Some growth signals present but model confidence is lower.</div>
        </div>
    </div>
    <p style='font-size:0.82rem;color:#6b8cae;margin-top:0.2rem'>2,438 suburb boundaries shown · Hover any suburb for details</p>
    """
    st.markdown(legend_html, unsafe_allow_html=True)

    for feature in geojson['features']:
        name = feature['properties'].get('SA2_NAME21', '')
        match = results[results['sa2_name'] == name]
        rank = int(match.iloc[0]['national_rank']) if len(match) > 0 else 9999
        feature['properties']['fill_color'] = rank_to_map_color(rank)
        feature['properties']['signal'] = 'Critical Pressure' if rank <= 50 else 'High Pressure' if rank <= 200 else 'Moderate / Low'
        feature['properties']['national_rank'] = rank

    layer = pdk.Layer(
        'GeoJsonLayer', data=geojson, pickable=True, stroked=True, filled=True,
        get_fill_color='properties.fill_color',
        get_line_color=[255, 255, 255, 60], line_width_min_pixels=1,
    )
    view = pdk.ViewState(latitude=-27.0, longitude=134.0, zoom=3.5, pitch=0)
    tooltip = {
        "html": "<b>{SA2_NAME21}</b> ({state})<br><b>{signal}</b><br>National Rank: <b>#{national_rank}</b><br>Pressure Score: <b>{pressure_score}</b>/100<br>Pop Growth: {erp_change_pct}%",
        "style": {"backgroundColor": "#1e3a5f", "color": "white", "fontSize": "13px", "padding": "8px", "borderRadius": "6px"}
    }
    st.pydeck_chart(pdk.Deck(
        layers=[layer], initial_view_state=view, tooltip=tooltip,
        map_style='https://basemaps.cartocdn.com/gl/positron-gl-style/style.json'
    ))

    # Full ranked table
    st.markdown('<div class="section-title">All Suburbs Ranked</div>', unsafe_allow_html=True)
    filtered = results.copy().sort_values(["pressure_score", "national_rank"], ascending=[False, True])
    display = filtered[[
        "national_rank", "sa2_name", "state", "pressure_score",
        "erp_change_pct", "growth_20yr", "years_of_growth", "dwellings_2526_fytd", "signal"
    ]].copy()
    display.columns = ["National Rank", "Suburb", "State", "Pressure Score", "Pop Growth %",
                       "20yr Growth", "Consecutive Growth Years", "2025-26 Approvals FYTD", "Signal"]
    display["20yr Growth"] = (display["20yr Growth"] * 100).round(1).astype(str) + "%"
    display["Pressure Score"] = display["Pressure Score"].round(1)
    display["2025-26 Approvals FYTD"] = display["2025-26 Approvals FYTD"].apply(lambda x: int(x) if pd.notna(x) else "N/A")
    st.dataframe(display, use_container_width=True, height=480, hide_index=True)

# ── About ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">About This Model</div>', unsafe_allow_html=True)
st.markdown("""
<div class="about-box">
<b style='color:#1e3a5f'>Methodology</b><br>
Version 10 builds on v9's XGBoost classifier trained on ABS building approval data across 2,442 suburbs nationally.
v9 introduced an urban renewal signal that fixed a systematic blind spot in inner-city precincts, achieving a Spearman rank correlation of 0.923.
v10 adds a saturation index that penalises fully built-out suburbs, SHAP-based score decomposition for explainability, and SA4 regional rollup.<br><br>
<b style='color:#1e3a5f'>What's New in v10</b><br>
Saturation index uses dwellings-per-capita and growth deceleration to penalise suburbs where construction activity is high but capacity is exhausted.
SHAP values from the XGBoost model explain every suburb's score in terms of its top contributing features.
SA4 rollup aggregates suburb scores to regional level for macro analysis.<br><br>
<b style='color:#1e3a5f'>Known Limitations</b><br>
No commencement data at SA2 level — pipeline proxy only. total_dwellings_2024-25 still dominates feature importance at 28.5%, 
meaning the model remains closer to an activity detector than a pure stress detector. v11 will split into Build Pressure and Constraint Pressure indices.<br><br>
<span class="tag">XGBoost</span>
<span class="tag">SHAP</span>
<span class="tag">Spearman 0.923</span>
<span class="tag">31 Features</span>
<span class="tag">Saturation Index</span>
<span class="tag">SA4 Rollup</span>
<span class="tag">scikit-learn</span>
<span class="tag">Streamlit</span>
<span class="tag">Python</span>
</div>
""", unsafe_allow_html=True)
