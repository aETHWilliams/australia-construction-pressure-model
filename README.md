Australia Construction Pressure Model

A machine learning system that predicts which Australian suburbs 
will experience a construction boom — before it happens.

## Results
- Backtest hit rate: 20/20 (100%)
- Random Forest AUC: 0.938
- XGBoost AUC: 0.938
- 2,442 suburbs analysed nationally

## Data Sources (7.3M+ records)
- ABS Building Approvals 2022-2026 (7.3 million records)
- ABS Regional Population 2023-24
- ABS Population History 2001-2024 (23 years)
- SEIFA Socioeconomic Index 2021
- ABS Building Approvals 2025-26 FYTD

## What It Found
- Logan-Ipswich corridor (QLD) is the highest pressure zone nationally
- Perth outer suburbs surging — Baldivis, Byford, Casuarina Wandi
- Suburbs with 20+ consecutive years of growth are the strongest signal
- Lower SEIFA score + high population growth = gentrification precursor

## Tech Stack
Python · scikit-learn · XGBoost · Pandas · GeoPandas · Folium · Streamlit

## Models
| Model | AUC |
|---|---|
| Random Forest | 0.938 |
| XGBoost | 0.938 |

Built with Python and open source tools.
Data sourced from ABS and QLD Open Data Portal.
Development assisted with AI tools.
