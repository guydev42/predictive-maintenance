"""Streamlit dashboard for predictive maintenance of industrial equipment."""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score,
)

PROJECT_DIR = os.path.dirname(__file__)
sys.path.insert(0, PROJECT_DIR)

from src.data_loader import generate_sensor_data, load_and_prepare
from src.model import _get_models, RANDOM_STATE

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Predictive maintenance", layout="wide")

GOLD = "#E8C230"
NAVY = "#3B6FD4"
GREEN = "#22c55e"
RED = "#ef4444"
YELLOW = "#f59e0b"


@st.cache_data
def load_data():
    path = os.path.join(PROJECT_DIR, "data", "sensor_readings.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return generate_sensor_data()


@st.cache_resource
def train_models(df):
    """Train all models and return results."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    feature_cols = [c for c in df.columns
                    if c not in ("failure_within_7days", "machine_id")]
    X = df[feature_cols].values.astype(float)
    y = df["failure_within_7days"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models_config = _get_models()
    results = {}
    trained = {}

    for name, config in models_config.items():
        model = config["model"]
        Xtr = X_train_scaled if config["needs_scaling"] else X_train
        Xte = X_test_scaled if config["needs_scaling"] else X_test

        model.fit(Xtr, y_train)
        y_prob = model.predict_proba(Xte)[:, 1]
        y_pred = model.predict(Xte)

        results[name] = {
            "y_prob": y_prob,
            "auc_roc": roc_auc_score(y_test, y_prob),
            "pr_auc": average_precision_score(y_test, y_prob),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }
        trained[name] = {
            "model": model,
            "needs_scaling": config["needs_scaling"],
        }

    return results, trained, X_test, X_test_scaled, y_test, feature_cols, scaler


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Machine health", "Failure timeline", "Sensor trends",
     "Maintenance scheduler", "Feature importance"],
)

df = load_data()
results, trained, X_test, X_test_scaled, y_test, feature_cols, scaler = train_models(df)

best_name = max(results, key=lambda n: results[n]["auc_roc"])


# ---------------------------------------------------------------------------
# Page: Machine health dashboard
# ---------------------------------------------------------------------------
if page == "Machine health":
    st.title("Machine health dashboard")
    st.markdown("Real-time health status for all monitored machines.")

    best_info = trained[best_name]
    model = best_info["model"]

    # Get latest reading per machine and score it
    latest = df.sort_values("operating_hours").groupby("machine_id").last().reset_index()
    X_latest = latest[feature_cols].values.astype(float)
    if best_info["needs_scaling"]:
        X_latest = scaler.transform(X_latest)
    probs = model.predict_proba(X_latest)[:, 1]
    latest["failure_probability"] = probs

    def status_label(p):
        if p < 0.3:
            return "Healthy"
        elif p < 0.6:
            return "Warning"
        else:
            return "Critical"

    def status_color(p):
        if p < 0.3:
            return GREEN
        elif p < 0.6:
            return YELLOW
        else:
            return RED

    latest["status"] = latest["failure_probability"].apply(status_label)
    latest["color"] = latest["failure_probability"].apply(status_color)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    n_healthy = (latest["status"] == "Healthy").sum()
    n_warning = (latest["status"] == "Warning").sum()
    n_critical = (latest["status"] == "Critical").sum()
    col1.metric("Total machines", len(latest))
    col2.metric("Healthy", n_healthy)
    col3.metric("Warning", n_warning)
    col4.metric("Critical", n_critical)

    st.markdown("---")

    # Machine grid
    fig = px.scatter(
        latest, x="machine_id", y="failure_probability",
        color="status",
        color_discrete_map={"Healthy": GREEN, "Warning": YELLOW, "Critical": RED},
        size="failure_probability",
        hover_data=["temperature", "vibration", "pressure", "rpm"],
        title="Machine failure probability overview",
    )
    fig.update_layout(
        xaxis_title="Machine ID", yaxis_title="Failure probability",
        height=450,
    )
    fig.add_hline(y=0.3, line_dash="dash", line_color=YELLOW,
                  annotation_text="Warning threshold")
    fig.add_hline(y=0.6, line_dash="dash", line_color=RED,
                  annotation_text="Critical threshold")
    st.plotly_chart(fig, use_container_width=True)

    # Detailed table
    st.subheader("Machine status details")
    display_cols = ["machine_id", "status", "failure_probability", "temperature",
                    "vibration", "pressure", "rpm", "operating_hours", "age_months"]
    st.dataframe(
        latest[display_cols].sort_values("failure_probability", ascending=False),
        use_container_width=True, hide_index=True,
    )


# ---------------------------------------------------------------------------
# Page: Failure probability timeline
# ---------------------------------------------------------------------------
elif page == "Failure timeline":
    st.title("Failure probability timeline")
    st.markdown("Track how failure probability evolves over time per machine.")

    machine_ids = sorted(df["machine_id"].unique())
    selected = st.multiselect("Select machines", machine_ids,
                              default=machine_ids[:5])

    if not selected:
        st.warning("Select at least one machine.")
    else:
        best_info = trained[best_name]
        model = best_info["model"]

        subset = df[df["machine_id"].isin(selected)].copy()
        X_sub = subset[feature_cols].values.astype(float)
        if best_info["needs_scaling"]:
            X_sub = scaler.transform(X_sub)
        subset["failure_prob"] = model.predict_proba(X_sub)[:, 1]

        subset = subset.sort_values(["machine_id", "operating_hours"])
        subset["reading_index"] = subset.groupby("machine_id").cumcount()

        fig = px.line(
            subset, x="reading_index", y="failure_prob",
            color="machine_id",
            title="Failure probability over readings",
            labels={"reading_index": "Reading sequence", "failure_prob": "P(failure)"},
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color=RED,
                      annotation_text="Decision threshold")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Summary per machine
        summary = subset.groupby("machine_id").agg({
            "failure_prob": ["mean", "max", "last"],
            "failure_within_7days": "sum",
        }).round(3)
        summary.columns = ["Mean P(fail)", "Max P(fail)",
                           "Latest P(fail)", "Actual failures"]
        st.dataframe(summary, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Sensor reading trends
# ---------------------------------------------------------------------------
elif page == "Sensor trends":
    st.title("Sensor reading trends")

    machine_id = st.selectbox("Select machine",
                              sorted(df["machine_id"].unique()))

    machine_data = df[df["machine_id"] == machine_id].sort_values("operating_hours")

    sensor_cols = ["temperature", "vibration", "pressure", "rpm", "power_consumption"]

    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(machine_data, y="temperature",
                      title=f"Machine {machine_id}: temperature",
                      color_discrete_sequence=[NAVY])
        fig.update_layout(height=300, xaxis_title="Reading index")
        st.plotly_chart(fig, use_container_width=True)

        fig = px.line(machine_data, y="pressure",
                      title=f"Machine {machine_id}: pressure",
                      color_discrete_sequence=[GREEN])
        fig.update_layout(height=300, xaxis_title="Reading index")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.line(machine_data, y="vibration",
                      title=f"Machine {machine_id}: vibration",
                      color_discrete_sequence=[GOLD])
        fig.update_layout(height=300, xaxis_title="Reading index")
        st.plotly_chart(fig, use_container_width=True)

        fig = px.line(machine_data, y="rpm",
                      title=f"Machine {machine_id}: RPM",
                      color_discrete_sequence=[RED])
        fig.update_layout(height=300, xaxis_title="Reading index")
        st.plotly_chart(fig, use_container_width=True)

    # Power consumption
    fig = px.line(machine_data, y="power_consumption",
                  title=f"Machine {machine_id}: power consumption",
                  color_discrete_sequence=[NAVY])
    fig.update_layout(height=300, xaxis_title="Reading index")
    st.plotly_chart(fig, use_container_width=True)

    # Statistics
    st.subheader("Sensor statistics")
    stats = machine_data[sensor_cols].describe().round(2)
    st.dataframe(stats, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Maintenance schedule optimizer
# ---------------------------------------------------------------------------
elif page == "Maintenance scheduler":
    st.title("Maintenance schedule optimizer")
    st.markdown("Cost-based threshold tuning to balance unplanned downtime "
                "against preventive maintenance costs.")

    col1, col2 = st.columns(2)
    with col1:
        fn_cost = st.number_input("Cost of unplanned downtime ($)",
                                  1000, 100000, 15000, step=1000)
    with col2:
        fp_cost = st.number_input("Cost of preventive maintenance ($)",
                                  100, 10000, 1500, step=100)

    y_prob = results[best_name]["y_prob"]
    thresholds = np.arange(0.05, 0.96, 0.01)
    records = []

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_test, y_pred_t)
        tn, fp, fn, tp = cm.ravel()
        total_cost = (fn * fn_cost) + (fp * fp_cost)
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        records.append({
            "Threshold": round(t, 3),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Recall": round(rec, 4),
            "Precision": round(prec, 4),
            "FPR": round(fpr, 4),
            "Total cost ($)": total_cost,
        })

    cost_df = pd.DataFrame(records)
    optimal_idx = cost_df["Total cost ($)"].idxmin()
    optimal = cost_df.loc[optimal_idx]

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Optimal threshold", f"{optimal['Threshold']:.3f}")
    col_b.metric("Recall", f"{optimal['Recall']:.1%}")
    col_c.metric("False positive rate", f"{optimal['FPR']:.1%}")
    col_d.metric("Total cost", f"${int(optimal['Total cost ($)']):,}")

    # Cost curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cost_df["Threshold"], y=cost_df["Total cost ($)"],
        mode="lines", name="Total cost", line=dict(color=NAVY, width=2),
    ))
    fig.add_vline(x=optimal["Threshold"], line_dash="dash", line_color="red",
                  annotation_text=f"Optimal: {optimal['Threshold']:.3f}")
    fig.update_layout(
        xaxis_title="Threshold", yaxis_title="Total cost ($)",
        height=400, title="Total cost by decision threshold",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Recall vs FPR
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cost_df["Threshold"], y=cost_df["Recall"],
        mode="lines", name="Recall", line=dict(color=GREEN, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=cost_df["Threshold"], y=cost_df["FPR"],
        mode="lines", name="False positive rate", line=dict(color=RED, width=2),
    ))
    fig.add_vline(x=optimal["Threshold"], line_dash="dash", line_color="red")
    fig.update_layout(
        xaxis_title="Threshold", yaxis_title="Rate",
        height=400, title="Recall and false positive rate by threshold",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Schedule recommendation
    st.subheader("Maintenance schedule recommendation")
    best_info = trained[best_name]
    model = best_info["model"]
    latest = df.sort_values("operating_hours").groupby("machine_id").last().reset_index()
    X_latest = latest[feature_cols].values.astype(float)
    if best_info["needs_scaling"]:
        X_latest = scaler.transform(X_latest)
    probs = model.predict_proba(X_latest)[:, 1]
    latest["failure_prob"] = probs
    latest["schedule"] = latest["failure_prob"].apply(
        lambda p: "Immediate" if p >= optimal["Threshold"]
        else ("Next 7 days" if p >= optimal["Threshold"] * 0.6
              else "Routine")
    )

    schedule_df = latest[["machine_id", "failure_prob", "schedule"]].sort_values(
        "failure_prob", ascending=False
    )
    st.dataframe(schedule_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page: Feature importance
# ---------------------------------------------------------------------------
elif page == "Feature importance":
    st.title("Feature importance analysis")

    import shap

    info = trained[best_name]
    model = info["model"]
    X = X_test_scaled if info["needs_scaling"] else X_test

    sample_size = min(300, X.shape[0])
    np.random.seed(RANDOM_STATE)
    idx = np.random.choice(X.shape[0], sample_size, replace=False)
    X_sample = X[idx]

    if info["needs_scaling"]:
        explainer = shap.LinearExplainer(model, X_sample)
    else:
        explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    # Global feature importance
    st.subheader("Global feature importance (SHAP)")
    mean_abs = np.abs(shap_vals).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Mean |SHAP|": mean_abs,
    }).sort_values("Mean |SHAP|", ascending=True)

    fig = px.bar(importance_df, x="Mean |SHAP|", y="Feature", orientation="h",
                 color_discrete_sequence=[NAVY])
    fig.update_layout(height=450, title="Mean absolute SHAP values")
    st.plotly_chart(fig, use_container_width=True)

    # Model comparison
    st.subheader("Model comparison")
    metrics_df = pd.DataFrame({
        name: {k: v for k, v in r.items() if k != "y_prob"}
        for name, r in results.items()
    }).T.round(4)
    st.dataframe(metrics_df, use_container_width=True)

    # ROC curves
    st.subheader("ROC curves")
    fig = go.Figure()
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name} (AUC={r['auc_roc']:.3f})",
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="gray"), name="Random",
    ))
    fig.update_layout(
        xaxis_title="False positive rate", yaxis_title="True positive rate",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Individual reading explanation
    st.subheader("Individual reading explanation")
    probs = model.predict_proba(X_sample)[:, 1]
    tx_index = st.slider("Select reading index", 0, sample_size - 1, 0)

    st.metric("Failure probability", f"{probs[tx_index]:.1%}")

    contrib_df = pd.DataFrame({
        "Feature": feature_cols,
        "SHAP value": shap_vals[tx_index],
        "Feature value": X_sample[tx_index],
    }).sort_values("SHAP value", key=abs, ascending=True)

    fig = px.bar(contrib_df.tail(11), x="SHAP value", y="Feature", orientation="h",
                 color="SHAP value",
                 color_continuous_scale=["#3B6FD4", "#cccccc", "#E8C230"])
    fig.update_layout(height=400, title="Top feature contributions")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Reading details")
    details = pd.DataFrame({
        "Feature": feature_cols,
        "Value": X_sample[tx_index],
        "SHAP contribution": shap_vals[tx_index],
    })
    st.dataframe(details, use_container_width=True, hide_index=True)
