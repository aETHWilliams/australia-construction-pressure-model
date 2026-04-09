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

results["erp_change_pct"] = results["erp_change_pct"].round(1)
results["dwellings_2526_fytd"] = results["dwellings_2526_fytd"].fillna(0).astype(int)

def rank_to_map_color(rank):
    if rank <= 50:
        return [220, 38, 38, 180]
    elif rank <= 200:
        return [217, 119, 6, 180]
    else:
        return [37, 99, 168, 140]

def get_urban_score(row):
    for col in ["urban_renewal_score", "urban_renewal_importance"]:
        if col in row and pd.notna(row[col]):
            return float(row[col])
    return 0.0

def classify_market_archetype(row):
    name = str(row.get("sa2_name", "")).lower()
    urban = get_urban_score(row)
    growth_years = float(row.get("years_of_growth", 0)) if pd.notna(row.get("years_of_growth", 0)) else 0
    pop_growth = float(row.get("erp_change_pct", 0)) if pd.notna(row.get("erp_change_pct", 0)) else 0

    urban_names = [
        "docklands", "rhodes", "zetland", "footscray", "southbank", "parramatta",
        "brunswick", "northbridge", "macquarie park", "kangaroo point", "fremantle"
    ]
    if any(x in name for x in urban_names):
        return "Urban Renewal"
    if urban >= 2.0:
        return "Inner Infill"
    if growth_years >= 12 and pop_growth >= 2.0:
        return "Greenfield Growth"
    return "Established / Mixed"

def classify_interpretation_risk(row):
    archetype = row["market_archetype"]
    rank = float(row.get("v10_rank", 9999))
    if archetype in ["Urban Renewal", "Inner Infill"] and rank <= 200:
        return "High"
    if archetype == "Greenfield Growth":
        return "Low"
    return "Medium"

def classify_pressure_frame(row):
    archetype = row["market_archetype"]
    sat = float(row.get("saturation_index", 0)) if pd.notna(row.get("saturation_index", 0)) else 0
    rank = float(row.get("v10_rank", 9999))

    if archetype == "Greenfield Growth" and rank <= 200:
        return "Likely Build Pressure"
    if archetype in ["Urban Renewal", "Inner Infill"] and rank <= 150:
        return "Needs Industry Validation"
    if sat >= 0.07:
        return "Maturing / Delivered"
    return "Likely Constraint Pressure"

def validation_signal(row):
    archetype = row["market_archetype"]
    if archetype in ["Urban Renewal", "Inner Infill"]:
        return "Demolition, enabling works, tender flow, contractor mobilisation"
    if archetype == "Greenfield Growth":
        return "Subdivision, servicing, trunk infrastructure, estate staging"
    return "Tender flow and site preparation"

v10["market_archetype"] = v10.apply(classify_market_archetype, axis=1)
v10["interpretation_risk"] = v10.apply(classify_interpretation_risk, axis=1)
v10["pressure_frame"] = v10.apply(classify_pressure_frame, axis=1)
v10["validation_signal"] = v10.apply(validation_signal, axis=1)

st.markdown("""
<div class="header-wrap">
    <div class="header-title">Australia Construction Pressure Index</div>
    <div class="header-sub">Suburb-level decision support model &nbsp;·&nbsp; 2,442 suburbs nationally &nbsp;·&nbsp; Model v10 &nbsp;·&nbsp; By Ethan Williams</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='background:#ffffff;border:1px solid #dbe8f5;border-radius:12px;padding:1rem 1.2rem;margin-bottom:1.6rem;
box-shadow:0 2px 8px rgba(37,99,168,0.05);font-size:0.92rem;color:#4a6080;line-height:1.8;'>
This model ranks Australian suburbs by likely construction pressure using approvals, population and pipeline signals.
Its main purpose is to distinguish between <b>near-term build pressure</b> and <b>constrained pipeline pressure</b>,
especially in apartment and urban renewal precincts where approvals and commencements may diverge.
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
metrics = [
    ("0.923", "CV Spearman"),
    ("611", "High Pressure Suburbs"),
    ("2,442", "Suburbs Analysed"),
    ("88", "SA4 Regions Ranked"),
    ("7.3M+", "Records Processed"),
]
for col, (val, label) in zip([c1, c2, c3, c4, c5], metrics):
    col.markdown(f"""
    <div class="metric-card">
        <span class="metric-value">{val}</span>
        <span class="metric-label">{label}</span>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-top: 2rem'></div>", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style='font-size:0.82rem; color:#1e3a5f; font-family:Sora,sans-serif; font-weight:600; margin-bottom:0.4rem'>What This App Does</div>
<div style='font-size:0.78rem; color:#4a6080; line-height:1.8; margin-bottom:1.2rem'>
This tool ranks Australian suburbs by likely construction pressure using approvals, population growth,
historical delivery, and pipeline signals. It is designed as a <b>decision-support model</b>, not a direct measure of commencements.
</div>

<div style='font-size:0.82rem; color:#1e3a5f; font-family:Sora,sans-serif; font-weight:600; margin-bottom:0.4rem'>Why This Matters</div>
<div style='font-size:0.78rem; color:#4a6080; line-height:1.8; margin-bottom:1.2rem'>
In some apartment and urban renewal precincts, approved dwelling volumes can remain high even when delivery is delayed by
feasibility, finance, staging, cost, or market conditions. That means approvals and commencements can tell different stories.
</div>

<div style='font-size:0.82rem; color:#1e3a5f; font-family:Sora,sans-serif; font-weight:600; margin-bottom:0.4rem'>v10 Positioning</div>
<div style='font-size:0.78rem; color:#4a6080; line-height:1.8; margin-bottom:1.2rem'>
<b>Saturation Adjustment</b> — reduces scores for suburbs that already appear heavily built out or decelerating.<br><br>
<b>SHAP Decomposition</b> — shows which signals are driving each suburb's result.<br><br>
<b>SA4 Rollup</b> — supports macro regional interpretation alongside suburb-level detail.
</div>

<div style='font-size:0.82rem; color:#1e3a5f; font-family:Sora,sans-serif; font-weight:600; margin-bottom:0.4rem'>Current Interpretation Gap</div>
<div style='font-size:0.78rem; color:#4a6080; line-height:1.8; margin-bottom:1.2rem'>
The main unresolved question is <b>approvals-to-commencement conversion</b>. The model currently sees pipeline pressure more directly
than actual on-site conversion, which is why this app now flags where <b>industry validation</b> is still needed.
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Executive Summary",
    "Industry Validation",
    "Approvals vs Commencement Risk",
    "v9 vs v10 Comparison",
    "SA4 Regional View",
    "Suburb Search",
    "Pressure Map"
])

with tab1:
    st.markdown('<div class="section-title">Executive Summary</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#e8f0fb;border-radius:10px;padding:1rem 1.4rem;margin-bottom:1.2rem;font-size:0.86rem;color:#1e3a5f;line-height:1.8;'>
    This model is strongest at identifying <b>where pressure signals are accumulating</b>.
    It is less direct at observing whether approved pipeline is converting into real site activity,
    especially in apartment and urban renewal markets. That distinction is the main focus of the next version.
    </div>
    """, unsafe_allow_html=True)

    version = st.radio("Model Version", ["v10 (Saturation Adjusted)", "v9 (Original)"], horizontal=True)

    if version == "v10 (Saturation Adjusted)":
        top10 = v10.sort_values("v10_rank").head(10).reset_index(drop=True)
        rank_col = "v10_rank"
        score_col = "v10_score"
        badge = '<span class="version-badge-v10">v10</span>'
    else:
        top10 = v10.sort_values("v9_rank").head(10).reset_index(drop=True)
        rank_col = "v9_rank"
        score_col = "v9_score"
        badge = '<span class="version-badge-v9">v9</span>'

    rows_html = ""
    for i, row in top10.iterrows():
        rank = int(row[rank_col])
        score = round(row[score_col], 2)
        v9r = int(row["v9_rank"])
        v10r = int(row["v10_rank"])
        shift = v9r - v10r
        arrow = ""
        if version == "v10 (Saturation Adjusted)" and shift != 0:
            arrow = f"<span style='color:#16a34a;font-size:0.75rem'>▲{abs(shift)}</span>" if shift > 0 else f"<span style='color:#dc2626;font-size:0.75rem'>▼{abs(shift)}</span>"
        color = "#f87171" if rank <= 50 else "#fbbf24" if rank <= 200 else "#93c5fd"
        signal = "Critical" if rank <= 50 else "High" if rank <= 200 else "Moderate"
        sat = round(row.get("saturation_index", 0), 3)
        bg = "rgba(255,255,255,0.05)" if i % 2 == 0 else "transparent"
        rows_html += (
            f"<tr style='font-size:0.82rem;background:{bg};'>"
            f"<td style='padding:0.5rem 0.6rem;color:rgba(255,255,255,0.35);font-weight:600'>#{rank} {arrow}</td>"
            f"<td style='padding:0.5rem 0.6rem;color:#ffffff;font-weight:600'>{row['sa2_name']}</td>"
            f"<td style='padding:0.5rem 0.6rem;color:#a8c8e8'>{row['state']}</td>"
            f"<td style='padding:0.5rem 0.6rem;color:{color};font-weight:700'>{score}</td>"
            f"<td style='padding:0.5rem 0.6rem;color:#a8c8e8'>{sat}</td>"
            f"<td style='padding:0.5rem 0.6rem;color:#a8c8e8'>{row['pressure_frame']}</td>"
            f"<td style='padding:0.5rem 0.6rem;color:{color}'>{signal}</td>"
            f"</tr>"
        )

    html = (
        f"<div style='background:linear-gradient(135deg,#1e3a5f 0%,#2563a8 60%,#3b82c4 100%);"
        f"border-radius:16px;padding:1.8rem 2rem;margin-bottom:2rem;"
        f"box-shadow:0 4px 24px rgba(37,99,168,0.18);'>"
        f"<div style='font-family:Sora,sans-serif;font-size:1.2rem;font-weight:700;color:#ffffff;margin-bottom:0.2rem'>"
        f"Top 10 Construction Pressure Suburbs — 2026/27 {badge}</div>"
        f"<div style='font-size:0.78rem;color:#a8c8e8;margin-bottom:1.2rem'>"
        f"Ranked by national construction pressure · Model output plus interpretation layer</div>"
        f"<table style='width:100%;border-collapse:collapse;'>"
        f"<tr style='font-size:0.68rem;color:#a8c8e8;text-transform:uppercase;letter-spacing:1px;border-bottom:1px solid rgba(255,255,255,0.1);'>"
        f"<td style='padding:0.4rem 0.6rem'>Rank</td>"
        f"<td style='padding:0.4rem 0.6rem'>Suburb</td>"
        f"<td style='padding:0.4rem 0.6rem'>State</td>"
        f"<td style='padding:0.4rem 0.6rem'>Score</td>"
        f"<td style='padding:0.4rem 0.6rem'>Saturation</td>"
        f"<td style='padding:0.4rem 0.6rem'>Frame</td>"
        f"<td style='padding:0.4rem 0.6rem'>Signal</td>"
        f"</tr>{rows_html}</table></div>"
    )
    st.markdown(html, unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-title">Industry Validation</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#ffffff;border:1px solid #dbe8f5;border-left:4px solid #2563a8;border-radius:10px;
    padding:1.2rem 1.4rem;margin-bottom:1.2rem;box-shadow:0 2px 8px rgba(37,99,168,0.05);font-size:0.85rem;
    color:#4a6080;line-height:1.8;'>
    This layer is designed to answer the main limitation in the model: <b>high-ranking suburbs are not all equal in certainty</b>.
    Some appear more likely to convert into near-term build activity. Others may represent constrained pipeline that still needs contractor judgement.
    </div>
    """, unsafe_allow_html=True)

    shortlist = v10[v10["v10_rank"] <= 200][[
        "sa2_name", "state", "v10_rank", "v10_score", "market_archetype",
        "pressure_frame", "interpretation_risk", "validation_signal"
    ]].copy().sort_values(["interpretation_risk", "v10_rank"], ascending=[True, True])

    shortlist.columns = [
        "Suburb", "State", "v10 Rank", "v10 Score", "Market Archetype",
        "Pressure Frame", "Interpretation Risk", "What Would Confirm It"
    ]

    st.dataframe(shortlist, use_container_width=True, height=500, hide_index=True)

    risk_counts = v10[v10["v10_rank"] <= 200]["interpretation_risk"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
    fig_risk = go.Figure(go.Bar(
        x=risk_counts.index,
        y=risk_counts.values,
        marker_color=["#16a34a", "#d97706", "#dc2626"]
    ))
    fig_risk.update_layout(
        title="Interpretation Risk Across Top 200 Suburbs",
        xaxis_title="Interpretation Risk",
        yaxis_title="Number of Suburbs",
        plot_bgcolor="#f7faff",
        paper_bgcolor="#ffffff",
        font=dict(family="Inter", size=12, color="#1a2332"),
        height=350,
        showlegend=False,
    )
    st.plotly_chart(fig_risk, use_container_width=True)

with tab3:
    st.markdown('<div class="section-title">Approvals vs Commencement Risk</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#ffffff;border:1px solid #dbe8f5;border-left:4px solid #2563a8;border-radius:10px;
    padding:1.2rem 1.4rem;margin-bottom:1.2rem;box-shadow:0 2px 8px rgba(37,99,168,0.05);font-size:0.85rem;
    color:#4a6080;line-height:1.8;'>
    In greenfield corridors, approvals often translate into visible construction activity more consistently.
    In apartment-led and urban renewal precincts, there can be a longer and less reliable lag between approval,
    feasibility, staging, and actual commencement. This view highlights suburbs where that distinction may matter most.
    </div>
    """, unsafe_allow_html=True)

    case_names = ["Rhodes", "Zetland", "Footscray", "Docklands", "Ripley"]
    case_rows = []
    for name in case_names:
        match = v10[v10["sa2_name"].str.contains(name, case=False, na=False)]
        if len(match) > 0:
            row = match.iloc[0]
            case_rows.append({
                "Suburb": row["sa2_name"],
                "State": row["state"],
                "v10 Rank": int(row["v10_rank"]),
                "v10 Score": round(row["v10_score"], 2),
                "Market Archetype": row["market_archetype"],
                "Pressure Frame": row["pressure_frame"],
                "Interpretation Risk": row["interpretation_risk"],
                "What Would Confirm It": row["validation_signal"]
            })

    st.dataframe(pd.DataFrame(case_rows), use_container_width=True, hide_index=True)

with tab4:
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
                Strong signal, but activity is not the same as build certainty.
            </div>
        </div>
        <div style='background:#dcfce7;border:1px solid #bbf7d0;border-left:4px solid #16a34a;border-radius:10px;padding:1.2rem 1.5rem;'>
            <div style='font-family:Sora,sans-serif;font-size:0.9rem;font-weight:700;color:#166534;margin-bottom:0.6rem'>v10 — Pressure Ranking with Interpretation Layer</div>
            <div style='font-size:0.8rem;color:#14532d;line-height:1.8;'>
                Saturation index penalises high-volume, decelerating suburbs.<br>
                SHAP decomposition explains every score.<br>
                SA4 rollup enables regional macro analysis.<br>
                New framing highlights where industry validation is still needed.
            </div>
        </div>
    </div>
    """, height=175)

    v10_compare = v10.copy()
    v10_compare["rank_change"] = v10_compare["v9_rank"] - v10_compare["v10_rank"]

    col_up, col_down = st.columns(2)

    with col_up:
        st.markdown('<div class="section-title">Biggest Risers (v9 to v10)</div>', unsafe_allow_html=True)
        risers = v10_compare.sort_values("rank_change", ascending=False).head(15)
        fig_up = go.Figure(go.Bar(
            x=risers["rank_change"],
            y=risers["sa2_name"] + " (" + risers["state"] + ")",
            orientation="h",
            marker_color="#16a34a",
            text=[f"#{int(r['v9_rank'])} → #{int(r['v10_rank'])}" for _, r in risers.iterrows()],
            textposition="outside",
        ))
        fig_up.update_layout(
            xaxis_title="Rank Improvement",
            yaxis=dict(autorange="reversed"),
            height=420,
            plot_bgcolor="#f7faff",
            paper_bgcolor="#ffffff",
            font=dict(family="Inter", size=11, color="#1a2332"),
            margin=dict(l=10, r=80, t=20, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig_up, use_container_width=True)

    with col_down:
        st.markdown('<div class="section-title">Biggest Fallers (Saturation Penalised)</div>', unsafe_allow_html=True)
        fallers = v10_compare.sort_values("rank_change").head(15)
        fig_down = go.Figure(go.Bar(
            x=fallers["rank_change"].abs(),
            y=fallers["sa2_name"] + " (" + fallers["state"] + ")",
            orientation="h",
            marker_color="#dc2626",
            text=[f"#{int(r['v9_rank'])} → #{int(r['v10_rank'])}" for _, r in fallers.iterrows()],
            textposition="outside",
        ))
        fig_down.update_layout(
            xaxis_title="Rank Drop",
            yaxis=dict(autorange="reversed"),
            height=420,
            plot_bgcolor="#f7faff",
            paper_bgcolor="#ffffff",
            font=dict(family="Inter", size=11, color="#1a2332"),
            margin=dict(l=10, r=80, t=20, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig_down, use_container_width=True)

with tab5:
    st.markdown('<div class="section-title">SA4 Regional Pressure Rankings</div>', unsafe_allow_html=True)

    sa4 = v10.groupby(["sa4_name", "state"]).agg(
        suburb_count=("sa2_name", "count"),
        avg_composite_score=("v10_score", "mean"),
        top_suburb=("sa2_name", lambda x: x.loc[v10.loc[x.index, "v10_score"].idxmax()]),
    ).reset_index()

    sa4["sa4_rank"] = sa4["avg_composite_score"].rank(ascending=False).astype(int)
    sa4 = sa4.sort_values("sa4_rank")

    st.dataframe(sa4, use_container_width=True, height=500, hide_index=True)

with tab6:
    st.markdown('<div class="section-title">Suburb Search</div>', unsafe_allow_html=True)
    search = st.text_input("Search", label_visibility="collapsed", placeholder="Search any suburb")

    if search:
        found = v10[v10["sa2_name"].str.contains(search, case=False, na=False)]
        if len(found) > 0:
            display = found[[
                "sa2_name", "state", "v10_rank", "v10_score", "market_archetype",
                "pressure_frame", "interpretation_risk", "validation_signal"
            ]].copy()
            display.columns = [
                "Suburb", "State", "v10 Rank", "v10 Score", "Market Archetype",
                "Pressure Frame", "Interpretation Risk", "What Would Confirm It"
            ]
            st.dataframe(display, use_container_width=True, hide_index=True)
        else:
            st.info("No suburb found.")

with tab7:
    st.markdown('<div class="section-title">Pressure Map</div>', unsafe_allow_html=True)

    for feature in geojson["features"]:
        name = feature["properties"].get("SA2_NAME21", "")
        match = results[results["sa2_name"] == name]
        rank = int(match.iloc[0]["national_rank"]) if len(match) > 0 else 9999
        feature["properties"]["fill_color"] = rank_to_map_color(rank)
        feature["properties"]["signal"] = "Critical Pressure" if rank <= 50 else "High Pressure" if rank <= 200 else "Moderate / Low"
        feature["properties"]["national_rank"] = rank

    layer = pdk.Layer(
        "GeoJsonLayer",
        data=geojson,
        pickable=True,
        stroked=True,
        filled=True,
        get_fill_color="properties.fill_color",
        get_line_color=[255, 255, 255, 60],
        line_width_min_pixels=1,
    )

    view = pdk.ViewState(latitude=-27.0, longitude=134.0, zoom=3.5, pitch=0)
    tooltip = {
        "html": "<b>{SA2_NAME21}</b><br><b>{signal}</b><br>National Rank: <b>#{national_rank}</b>",
        "style": {"backgroundColor": "#1e3a5f", "color": "white", "fontSize": "13px", "padding": "8px", "borderRadius": "6px"}
    }

    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        tooltip=tooltip,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
    ))

st.markdown('<div class="section-title">About This Model</div>', unsafe_allow_html=True)
st.markdown("""
<div class="about-box">
<b style='color:#1e3a5f'>Methodology</b><br>
Version 10 builds on v9's XGBoost framework trained on suburb-level approvals, demographic and pipeline signals across 2,442 suburbs nationally.
The model is designed to rank <b>construction pressure</b>, not to directly observe commencements at suburb level.<br><br>

<b style='color:#1e3a5f'>What This New Layer Adds</b><br>
The app now adds a practical interpretation layer that flags where high rankings are more likely to reflect near-term build pressure,
and where they may instead reflect constrained pipeline that still needs industry validation.<br><br>

<b style='color:#1e3a5f'>Key Interpretation Gap</b><br>
The main unresolved issue is <b>approvals-to-commencement conversion</b>. The model currently sees approvals and pipeline pressure more directly
than actual on-site conversion, particularly in apartment and urban renewal precincts where feasibility, staging, and finance can delay delivery.<br><br>

<span class="tag">XGBoost</span>
<span class="tag">Interpretation Layer</span>
<span class="tag">Build Pressure</span>
<span class="tag">Constraint Pressure</span>
<span class="tag">SA4 Rollup</span>
<span class="tag">Streamlit</span>
</div>
""", unsafe_allow_html=True)
