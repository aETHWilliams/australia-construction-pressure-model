Australia Construction Pressure Index A machine learning system that predicts which Australian suburbs will experience a construction boom — before it happens.

Model v8 Results
CV Spearman rank correlation: 0.750 ± 0.001
Top-10 CV precision: 50/50 (10 per fold, across 5 folds)
Random Forest AUC: 0.938
2,442 suburbs analysed nationally
13 new suburb calls for 2026/27 not previously flagged

Data Sources (7.3M+ records)
ABS Building Approvals 2022–26
ABS Regional Population 2023–24
ABS Population History 2001–2024 (23 years)
SEIFA Socioeconomic Index 2021
ABS Building Approvals 2025–26 FYTD

What v8 Found
Mermaid Waters (QLD) is the #1 predicted surge suburb for 2026/27 — 530% approval spike against 21 consecutive growth years
Urban renewal precincts (Rhodes, Zetland, Footscray) are confirmed blind spots — large DA pipelines not yet converting to commencements
Suburbs with 20+ consecutive years of growth remain the strongest signal
v9 will add DA pipeline data to fix urban renewal underweighting

Known Limitations Urban renewal precincts with large approval pipelines but low recorded commencements are currently underscored. This is the primary target for v9.

Tech Stack Python · scikit-learn · XGBoost · Pandas · Streamlit · Joblib
