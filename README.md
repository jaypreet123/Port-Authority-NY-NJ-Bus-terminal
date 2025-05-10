# Passenger and Traffic Forecasting Project (2025–2030)  
**Port Authority of New York & New Jersey | Group 4 - BANL 6430**

## Project Overview
This project aims to help the Port Authority of NY & NJ forecast passenger traffic and vehicle flow across terminals and bridges from 2025 to 2030. By analyzing historical data from 2005–2024, we provide actionable insights for infrastructure planning, carrier scheduling, and congestion management.

### Key Business Goals
1. **Forecast total passenger volume** at Midtown Bus Terminal (MBT) and George Washington Bridge Bus Station (GWBBS).
2. **Predict passenger volume per carrier** for smarter gate/staff allocation.
3. **Identify busiest time periods** (by week/month/year) for staging facilities.
4. **Compare 2025 traffic volumes** to pre-COVID 2019 benchmarks.
5. **Recommend data-driven strategies** for efficient terminal and bridge operations.

##  Data Sources
We worked with 3 Microsoft Access databases:
- **TBT Traffic Database** – Traffic flow, counts, and congestion metrics.
- **Unit 571 Database** – Passenger counts and bus departure logs.
- **Weather Database** – Climate metrics (temperature, wind, snow, rain).

We performed cleaning, formatting, deduplication, and aggregation using SQL and Python.

## Tools & Technologies
| Tool        | Purpose |
|-------------|---------|
| **SQL**     | Data extraction, preprocessing, joins |
| **Python**  | Forecasting, clustering, classification |
| **Power BI**| Dashboards, visual storytelling |

## Models & Methodologies

### Forecasting Models
- **Prophet** (by Meta) — Time series model used to predict monthly passenger volumes per terminal from 2025–2030.
- **XGBoost Regression** — Carrier-wise forecasting using weather + temporal features.

### Classification Models
- **Random Forest Classifier** — Predicts “High-Traffic” months based on top 25% threshold.

### Clustering Models
- **K-Means Clustering** — Groups terminals and carriers by similar traffic trends for operational alignment.

## Key Insights & Results
- **MBT** expected to serve ~180M passengers; **GWBBS** ~8M from 2025 to 2030.
- **Peak traffic** during summer (May–August) and midday (11 AM–5 PM).
- **Top carriers**: NJ Transit, Greyhound, Hudson Transit; require additional space and gates.
- **GWB and Bayonne Bridge** surpassed 2019 traffic; Holland Tunnel lags behind.

## Power BI Dashboard Highlights
Interactive dashboards visualize:
- Forecasted traffic by terminal and bridge
- Monthly and weekly congestion patterns
- Carrier-specific demand trends
- 2019 vs 2025 recovery comparisons

## Repository Structure
```
project-root/
│
├── data/
│   ├── raw/               # Raw input datasets
│   └── cleaned/           # Cleaned, transformed data
│
├── sql/                  
│   ├── deduplicated_queries.sql
│   └── annotated_deduplicated_queries.sql
│
├── dashboard/            # Power BI files, screenshots, exports
│
├── reports/              # Final reports, presentations
│
├── notebooks/            # Python scripts and models
│
└── README.md             # This file
```


## Recommendations to Port Authority
- Prioritize **summer and midday** staffing at terminals.
- Expand **temporary staging** at MBT for overflow handling.
- Use **carrier clustering** to optimize gate allocation.
- Collect **real-time weather and traffic data** for faster decision-making.
- Factor in **exogenous trends** (e.g., gas prices, remote work, urban development).

## Final Deliverables
-  Clean SQL scripts (deduplicated & annotated)
-  Forecasting models with outputs
-  Power BI dashboards
-  Project reports and presentations
