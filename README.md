<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Predictive%20Maintenance&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=35&desc=XGBoost%20%2B%20Survival%20Analysis%20%2B%20SHAP%20for%20equipment%20failure%20prediction&descAlignY=55&descSize=16" width="100%"/>

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/AUC--ROC-0.94-22c55e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Recall-91%25-22c55e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-f59e0b?style=for-the-badge"/>
</p>

<p>
  <a href="#overview">Overview</a> •
  <a href="#key-results">Key results</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#dataset">Dataset</a> •
  <a href="#methodology">Methodology</a>
</p>

</div>

---

## Overview

> **A predictive maintenance system that forecasts industrial equipment failures from sensor data using gradient boosting, survival analysis for remaining useful life estimation, and SHAP-based explanations.**

Unplanned equipment downtime costs industrial operations thousands of dollars per hour. This project builds a failure prediction pipeline that ingests sensor readings (temperature, vibration, pressure, RPM, power consumption) from 50 machines and predicts whether a failure will occur within the next 7 days. Four classifiers are trained, a Weibull survival model estimates remaining useful life, and a cost-based threshold optimizer balances unplanned downtime ($15K) against preventive maintenance ($1.5K).

```
Problem   →  Predicting equipment failures before they happen using sensor data
Solution  →  XGBoost with survival analysis, SHAP explanations, and cost-based threshold tuning
Impact    →  AUC 0.94, catches 91% of failures with optimized maintenance scheduling
```

---

## Key results

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.94 |
| Recall (failures caught) | 91% |
| Precision | 78% |
| PR-AUC | 0.76 |
| Best model | XGBoost |

---

## Architecture

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Sensor data     │───▶│  Feature          │───▶│  Rolling         │
│  generation      │    │  extraction       │    │  aggregations    │
└──────────────────┘    └──────────────────┘    └────────┬─────────┘
                                                         │
                          ┌──────────────────────────────┘
                          ▼
              ┌──────────────────────┐    ┌──────────────────────┐
              │  Model training      │───▶│  Survival analysis   │
              │  (4 classifiers)     │    │  (Weibull AFT / RUL) │
              └──────────────────────┘    └──────────┬───────────┘
                                                     │
                          ┌──────────────────────────┘
                          ▼
              ┌──────────────────────┐    ┌──────────────────────┐
              │  SHAP explanations   │───▶│  Maintenance         │
              │  + threshold tuning  │    │  dashboard           │
              └──────────────────────┘    └──────────────────────┘
```

<details>
<summary><b>Project structure</b></summary>

```
project_21_predictive_maintenance/
├── data/
│   ├── sensor_readings.csv              # Sensor dataset
│   └── generate_data.py                 # Synthetic data generator
├── src/
│   ├── __init__.py
│   ├── data_loader.py                   # Data generation and loading
│   └── model.py                         # Training, evaluation, SHAP, RUL
├── notebooks/
│   ├── 01_eda.ipynb                     # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb     # Rolling features, interactions
│   ├── 03_modeling.ipynb                # Model training and CV
│   └── 04_evaluation.ipynb             # ROC, SHAP, cost analysis
├── app.py                               # Streamlit dashboard
├── requirements.txt
└── README.md
```

</details>

---

## Quickstart

```bash
# Clone and navigate
git clone https://github.com/guydev42/calgary-data-portfolio.git
cd calgary-data-portfolio/project_21_predictive_maintenance

# Install dependencies
pip install -r requirements.txt

# Generate sensor data
python data/generate_data.py

# Launch dashboard
streamlit run app.py
```

---

## Dataset

| Property | Details |
|----------|---------|
| Source | Synthetic industrial sensor data |
| Readings | 15,000 |
| Machines | 50 |
| Failure rate | ~8% (1,200 pre-failure readings) |
| Features | 11 (temperature, vibration, pressure, rpm, power, rolling stats) |
| Target | failure_within_7days (binary) |

---

## Tech stack

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/SHAP-4B8BBE?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/lifelines-6A5ACD?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
</p>

---

## Methodology

<details>
<summary><b>Sensor feature engineering</b></summary>

- Rolling 24h mean temperature and standard deviation of vibration
- Temperature-pressure ratio as an interaction feature
- Machine-level attributes: age, operating hours, maintenance history
</details>

<details>
<summary><b>Model training</b></summary>

- Four classifiers: Logistic Regression, Random Forest, XGBoost, Gradient Boosting
- 5-fold StratifiedKFold cross-validation
- Class imbalance handled via class_weight and scale_pos_weight
- Metrics: AUC-ROC, precision, recall, F1, PR-AUC
</details>

<details>
<summary><b>Survival analysis</b></summary>

- Weibull Accelerated Failure Time (AFT) model from lifelines
- Covariates: machine age, mean temperature, mean vibration
- Outputs remaining useful life (RUL) estimates per machine
</details>

<details>
<summary><b>SHAP explainability</b></summary>

- TreeExplainer for gradient boosting models
- Global feature importance via mean absolute SHAP values
- Waterfall plots for individual sensor reading explanations
</details>

<details>
<summary><b>Cost-optimized threshold</b></summary>

- Business cost model: FN cost ($15,000 unplanned downtime) vs FP cost ($1,500 preventive maintenance)
- Sweep thresholds from 0.05 to 0.95 to minimize total cost
- Achieves 91% recall with optimized maintenance scheduling
</details>

---

## Acknowledgements

Built as part of the [Calgary Data Portfolio](https://guydev42.github.io/calgary-data-portfolio/).

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

**[Ola K.](https://github.com/guydev42)**
</div>
