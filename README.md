# EV Adoption & Infrastructure in Canada (2017–2024)

**Goal.** Analyze the growth of zero-emission vehicles (ZEVs) and charging infrastructure across Canadian provinces and normalize by population for fair comparisons.  
The project produces cleaned and merged datasets, predictive models for EV adoption, and 2025 scenario forecasts.

---

## 📊 Data Sources

### 1. EV Chargers
- **Source:** Natural Resources Canada (NRCan) — Alternative Fuelling Stations Locator  
  🔗 https://natural-resources.canada.ca/energy-efficiency/transportation-energy-efficiency/electric-charging-alternative-fuelling-stationslocator-map#/analyze?country=CA&tab=fuel&ev_levels=all&fuel=ELEC
- **File:** `data/raw/raw_chargers.csv`
- **Processed Output:** `data/processed/chargers_processed.csv`
- **Description:** Active Level 2 and DC Fast charging stations across Canada. Cleaned by removing duplicates, converting dates, and aggregating openings by province and year. Converted to cumulative in-service stocks.

```
chargers_ports = level2_ports + dcfast_ports
fast_share = dcfast_ports / chargers_ports
```

---

### 2. Population
- **Source:** Statistics Canada — Table 17-10-0005-01 (Population estimates by age and sex)  
  🔗 https://www150.statcan.gc.ca/t1/tbl1/en/cv.action?pid=1710000501
- **File:** `data/raw/raw_population_province.csv`
- **Processed Output:** `data/processed/population_processed.csv`
- **Description:** Annual population aged 16+ by province. Used to compute per-capita adoption and infrastructure indicators.

```
ev_per_1k = ev_annual / population_16plus * 1000
ports_per_100k = chargers_ports / population_16plus * 100000
```

---

### 3. Zero-Emission Vehicles (ZEV)
- **Source:** Statistics Canada — Table 20-10-0025-01 (Quarterly motor vehicle registrations by fuel type)  
  🔗 https://www150.statcan.gc.ca/t1/tbl1/en/cv.action?pid=2010002501
- **File:** `data/raw/raw_zev_quarterly.csv`
- **Processed Output:** `data/processed/zev_processed.csv`
- **Description:** Quarterly registrations of BEVs and PHEVs aggregated annually (2017–2024). Interpolated within provinces (no extrapolation).

---

## 🧼 Data Cleaning & Integration

**1️⃣ Cleaning (ETL)**  
Each dataset undergoes parsing, deduplication, missing-value replacement, and type harmonization.

**2️⃣ Annual Aggregation**
- Chargers → yearly openings → cumulative in-service totals.  
- ZEV → quarterly registrations → annual totals.  
- Population → yearly totals (16+).

**3️⃣ Merge**
Merged by (`geo`, `year`) to produce:
`merge_chargers_zev_pop_2017_2024.csv`  
Intermediate outputs: `chargers_processed.csv`, `zev_processed.csv`, `population_processed.csv`.

**4️⃣ Derived Indicators**
- `ev_per_1k`: EVs per 1,000 adults (16+)  
- `ports_per_100k`: Charger ports per 100,000 adults  
- `fast_share`: DC-fast chargers as % of total ports

---

## 🔢 Modeling & Prediction

Two predictive models were trained on 2017–2023 data and tested on 2024:

| Model | Target | Weighted MAE | Weighted R² |
|--------|---------|---------------|-------------|
| **Counts** | `ev_annual` | 3,473 EV | 0.978 |
| **Per-capita** | `ev_per_1k` | 1.28 EV/1k | 0.894 |

**Approach:**
- RidgeCV regression (geo-aware via one-hot provinces)
- Features: `t`, `chargers_ports`, `fast_share`, `ev_lag1`
- Weighted by `population_16plus`

### Forecast Scenarios (2025)
1. **Hold 2024 levels** — keep chargers and fast-share constant.  
2. **Growth scenario** — +20% charger ports, +2pp fast-share.  

Output file: `data/processed/ev_forecast_2025_weighted.csv`

---

## 📈 Tableau Visualizations

Tableau dashboards built from:
- `merge_chargers_zev_pop_2017_2024.csv` (historical)  
- `ev_forecast_2025_weighted.csv` (forecasted)

Dashboards include:
- EV adoption vs. charging density (per province)
- EV/1k and ports/100k trendlines (2017–2024 + forecast)
- Fast vs. Level 2 charger share

🔗 Tableau Dashboard: *[Link coming soon]*

---

## 🧾 Directory Summary

```
data/
├── raw/
│   ├── raw_chargers.csv
│   ├── raw_population_province.csv
│   └── raw_zev_quarterly.csv
└── processed/
    ├── chargers_processed.csv
    ├── zev_processed.csv
    ├── population_processed.csv
    ├── population_age_shares.csv
    ├── merge_chargers_zev_2017_2024.csv
    ├── merge_chargers_zev_pop_2017_2024.csv
    ├── ev_forecast_2025_weighted.csv
```

---

## 🧪 Usage Example (Python)

```python
# Load merged dataset
import pandas as pd

df = pd.read_csv("data/processed/merge_chargers_zev_pop_2017_2024.csv")

# Inspect structure
print(df.columns)
print(df.head())

# Run prediction (after training)
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

features = ["t", "chargers_ports", "fast_share", "ev_lag1", "geo"]
target = "ev_per_1k"

df["t"] = df["year"] - df["year"].min()
df["ev_lag1"] = df.groupby("geo")["ev_annual"].shift(1)

train = df[df["year"] <= 2023].dropna(subset=[target])
test = df[df["year"] == 2024]

pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["geo"])], remainder="passthrough")
model = Pipeline([("prep", pre), ("ridge", RidgeCV(alphas=[0.01, 0.1, 1, 10, 100]))])
model.fit(train[features], train[target])

preds = model.predict(test[features])
print("2024 R²:", model.score(test[features], test[target]))
```

---

## 🙏 Acknowledgements

- Natural Resources Canada — EV Charging Infrastructure  
- Statistics Canada — Population & ZEV Registrations
