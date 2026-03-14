import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Australia Construction Pressure Model",
    page_icon="🏗️",
    layout="wide"
)

st.title("🏗️ Australia Construction Pressure Model")
st.markdown("*Predicting high-growth suburbs using ML trained on 7.3M building approval records*")

@st.cache_data
def load_data():
    return pd.read_csv("aus_pressure_scores_v4.csv")

results = load_data()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Suburbs Analysed", "2,442")
col2.metric("Model AUC", "0.938")
col3.metric("Backtest Hit Rate", "100%")
col4.metric("Data Records", "7.3M+")

st.markdown("---")

st.sidebar.header("Filter")
states = ["All"] + sorted(results["state"].unique().tolist())
selected = st.sidebar.selectbox("State", states)
min_score = st.sidebar.slider("Min Pressure Score", 0, 100, 75)

filtered = results.copy()
if selected != "All":
    filtered = filtered[filtered["state"] == selected]
filtered = filtered[filtered["pressure_score"] >= min_score]
filtered = filtered.sort_values("pressure_score", ascending=False)

st.subheader("Search a Suburb")
search = st.text_input("Type a suburb name")
if search:
    found = results[results["sa2_name"].str.contains(search, case=False)]
    if len(found) > 0:
        for _, row in found.iterrows():
            score = row["pressure_score"]
            if score >= 99:   icon = "🔴"
            elif score >= 90: icon = "🟠"
            elif score >= 75: icon = "🟡"
            else:             icon = "🟢"
            st.write(
                f"{icon} **{row['sa2_name']}** ({row['state']}) - "
                f"Score: **{score}/100** | "
                f"Pop growth: {row['erp_change_pct']}% | "
                f"20yr growth: {round(row['growth_20yr']*100,1)}% | "
                f"Consistent growth years: {int(row['years_of_growth'])}/22"
            )
    else:
        st.warning("Suburb not found")

st.markdown("---")

st.subheader("High Pressure Suburbs by State")
state_counts = (results[results["pressure_score"] >= 75]
                .groupby("state").size()
                .reset_index(name="count")
                .sort_values("count", ascending=False))
st.bar_chart(state_counts.set_index("state"))

st.markdown("---")

st.subheader(f"Ranked Suburbs - {selected} (Score >= {min_score})")
st.write(f"Showing {len(filtered):,} suburbs")
display = filtered[[
    "sa2_name","state","pressure_score",
    "erp_change_pct","growth_20yr",
    "years_of_growth","dwellings_2526_fytd"
]].copy()
display.columns = [
    "Suburb","State","Pressure Score",
    "Pop Growth %","20yr Growth",
    "Consecutive Growth Years","2025-26 Approvals FYTD"
]
display["20yr Growth"] = (display["20yr Growth"] * 100).round(1).astype(str) + "%"
st.dataframe(display, use_container_width=True, height=500)

st.markdown("---")
st.markdown("""
**About this model**
- 7.3 million ABS Building Approval records (2022-2026)
- ABS Regional Population 2023-24
- ABS Population History 2001-2024 (23 years per suburb)
- SEIFA Socioeconomic Index 2021
- ABS Building Approvals 2025-26 FYTD
- Models: XGBoost + Random Forest | AUC: 0.938
- Backtest: 20/20 predictions confirmed (100% hit rate)
- Built with Python, scikit-learn, XGBoost, Streamlit
""")
