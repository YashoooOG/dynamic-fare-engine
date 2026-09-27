# ⚡ Dynamic Fare Engine

ML-based dynamic surge pricing and revenue optimization engine for **Ride-Hailing (Uber/Lyft)** and **Food Delivery** marketplaces.

---

## 🌟 Features

- **Multi-Modal Dynamic Pricing**: Real-time fare calculations for Ride-Hailing (Standard, Premium, Shared) and On-Demand Food Delivery (Distance, Small Basket, Weather & Traffic Surcharges).
- **Machine Learning Layer**: Trained XGBoost and Gradient Boosting models for demand forecasting and delivery delay prediction.
- **Microeconomic Revenue Optimization**: Solves $\arg\max_s E[\text{Revenue}(s)]$ balancing price elasticity with driver fulfillment.
- **Monte Carlo Market Simulation**: Benchmarks Static vs. Dynamic pricing strategies demonstrating $+38\%$ gross revenue lift and improved marketplace fulfillment.
- **Interactive Streamlit Dashboard**: Dark-mode interface with live dispatch simulator, zone maps, 24-hour diurnal demand curves, and transaction logging.
- **Scheduler Architecture Blueprint**: Official periodic batch pipeline specification detailed in comments in `scheduler/hourly_job.py`.
- **Database Layer**: SQLite database (`data/db.sqlite`) tracking urban zones, real-time pricing quote logs, and marketplace time-series snapshots.

---

## 📁 Repository Structure

```
dynamic-fare-engine/
├── app/
│   ├── assets/style.css             # Glassmorphism dark-theme CSS design system
│   ├── components/
│   │   ├── price_display.py         # Dynamic pricing receipt card component
│   │   ├── demand_chart.py          # Altair diurnal demand curves & trade-off plots
│   │   └── zone_map.py              # Geospatial urban zone map & status cards
│   └── streamlit_app.py             # Master multi-tab Streamlit dashboard
├── config.yaml                      # Engine, model, and pricing policy configuration
├── data/
│   ├── raw/                         # Raw ride, delivery, and weather datasets
│   ├── interim/                     # Cleaned interim parquet files
│   ├── processed/                   # ML-ready feature datasets and splits
│   └── db.sqlite                    # SQLite database
├── docs/
│   ├── architecture.md              # System design, Mermaid diagrams & math formulas
│   ├── dataset_notes.md             # Dataset documentation & sourcing notes
│   └── report.md                    # Final project report & benchmark results
├── models/                          # Trained XGBoost .joblib model artifacts
├── notebooks/                       # Exploratory Data Analysis & Simulation notebooks
├── scheduler/
│   └── hourly_job.py                # Hourly batch scheduler blueprint (in comments)
├── src/
│   ├── db/                          # Database connection, schemas, and queries
│   ├── features/                    # Feature engineering & temporal transforms
│   ├── ingestion/                   # Raw data loaders and weather fetchers
│   ├── models/                      # ML model classes, training & evaluation
│   └── pricing/                     # Core surge engine & revenue optimizer
└── tests/                           # Unit & integration test suite
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Feature Engineering & Model Training (Optional - Pretrained models included)
```bash
# Build processed dataset splits
python -m src.features.build_features

# Train ML demand forecasting models
python -m src.models.train
```

### 3. Run the Interactive Web Dashboard
```bash
streamlit run app/streamlit_app.py
```

### 4. Run Test Suite
```bash
python -m unittest discover -s tests
```

---

## 🧪 Testing

All 17 unit and integration tests are verified across feature transformations, machine learning estimators, pricing engine calculations, and database logging:

```bash
Ran 17 tests in 3.845s
OK
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](file:///home/yash/Projects/dynamic-fare-engine/LICENSE) file for details.
