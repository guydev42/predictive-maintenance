"""
Train, evaluate, and explain predictive maintenance models.
Includes: Logistic Regression, Random Forest, XGBoost, Gradient Boosting.
Generates SHAP explanations, survival analysis for RUL estimation,
and cost-optimized threshold tuning.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
OUTPUTS_DIR = "outputs"


def _ensure_dirs():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


def _get_models():
    """Return model instances configured for imbalanced failure detection."""
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(
                random_state=RANDOM_STATE, max_iter=1000,
                class_weight="balanced",
            ),
            "needs_scaling": True,
        },
        "Random Forest": {
            "model": RandomForestClassifier(
                random_state=RANDOM_STATE, n_estimators=200,
                max_depth=12, min_samples_split=5,
                class_weight="balanced", n_jobs=-1,
            ),
            "needs_scaling": False,
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(
                random_state=RANDOM_STATE, n_estimators=200,
                max_depth=5, learning_rate=0.1,
                min_samples_split=5,
            ),
            "needs_scaling": False,
        },
    }

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = {
            "model": XGBClassifier(
                random_state=RANDOM_STATE, eval_metric="logloss",
                use_label_encoder=False, verbosity=0,
                n_estimators=200, max_depth=6, learning_rate=0.1,
                scale_pos_weight=11,  # ~ratio for 8% failure rate
            ),
            "needs_scaling": False,
        }
    except ImportError:
        print("XGBoost not installed, skipping.")

    return models


def estimate_rul(df, feature_names):
    """
    Estimate remaining useful life (RUL) using a survival analysis approach.
    Uses the Weibull AFT model from lifelines on operating_hours with
    failure as the event indicator.

    Returns:
        rul_df: DataFrame with machine_id, predicted_rul, survival_probability
    """
    try:
        from lifelines import WeibullAFTFitter
    except ImportError:
        print("lifelines not installed, skipping RUL estimation.")
        return None

    survival_df = df[["machine_id", "operating_hours", "age_months",
                       "temperature", "vibration", "failure_within_7days"]].copy()
    survival_df = survival_df.rename(columns={
        "operating_hours": "duration",
        "failure_within_7days": "event",
    })

    # Aggregate per machine: take max duration, mean sensor values
    machine_agg = survival_df.groupby("machine_id").agg({
        "duration": "max",
        "event": "max",
        "age_months": "first",
        "temperature": "mean",
        "vibration": "mean",
    }).reset_index()

    # Ensure positive durations
    machine_agg["duration"] = machine_agg["duration"].clip(lower=1)

    aft = WeibullAFTFitter()
    aft.fit(
        machine_agg[["duration", "event", "age_months", "temperature", "vibration"]],
        duration_col="duration",
        event_col="event",
    )

    # Predict median survival time as RUL proxy
    predicted_median = aft.predict_median(
        machine_agg[["age_months", "temperature", "vibration"]]
    )
    machine_agg["predicted_rul_hours"] = (predicted_median - machine_agg["duration"]).clip(lower=0).round(0)

    # Survival probability at current time
    machine_agg["survival_probability"] = aft.predict_survival_function(
        machine_agg[["age_months", "temperature", "vibration"]],
        times=[machine_agg["duration"].median()]
    ).values[0]

    rul_df = machine_agg[["machine_id", "duration", "predicted_rul_hours",
                           "survival_probability"]].copy()
    rul_df = rul_df.rename(columns={"duration": "current_hours"})

    print("\nRUL estimation (Weibull AFT model):")
    print(f"  Machines analyzed: {len(rul_df)}")
    print(f"  Mean predicted RUL: {rul_df['predicted_rul_hours'].mean():.0f} hours")
    print(f"  Median predicted RUL: {rul_df['predicted_rul_hours'].median():.0f} hours")

    return rul_df


def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names):
    """
    Train all models, evaluate with multiple metrics, generate SHAP plots,
    and run threshold optimization with cost-benefit analysis.

    Returns:
        dict of results per model
    """
    _ensure_dirs()

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models_config = _get_models()
    results = {}
    trained_models = {}

    print("\n" + "=" * 70)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 70)

    for name, config in models_config.items():
        print(f"\n--- {name} ---")

        model = config["model"]
        Xtr = X_train_scaled if config["needs_scaling"] else X_train
        Xte = X_test_scaled if config["needs_scaling"] else X_test

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model, Xtr, y_train, cv=cv, scoring="roc_auc")
        print(f"  CV AUC-ROC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

        # Fit on full training set
        model.fit(Xtr, y_train)
        trained_models[name] = {
            "model": model,
            "needs_scaling": config["needs_scaling"],
        }

        y_pred = model.predict(Xte)
        y_prob = model.predict_proba(Xte)[:, 1]

        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "precision": prec, "recall": rec, "f1": f1,
            "auc_roc": auc, "pr_auc": pr_auc,
            "confusion_matrix": cm, "y_prob": y_prob,
            "cv_auc_mean": cv_scores.mean(), "cv_auc_std": cv_scores.std(),
        }

        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  AUC-ROC:   {auc:.4f}")
        print(f"  PR-AUC:    {pr_auc:.4f}")
        print(f"  Confusion matrix:")
        print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
        print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

    # --- Model comparison ---
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    comparison_df = pd.DataFrame({
        name: {k: v for k, v in r.items()
               if k not in ("confusion_matrix", "y_prob")}
        for name, r in results.items()
    }).T.round(4)
    print(comparison_df.to_string())
    comparison_df.to_csv(os.path.join(OUTPUTS_DIR, "model_comparison.csv"))

    # --- Best model ---
    best_name = max(results, key=lambda n: results[n]["auc_roc"])
    best_auc = results[best_name]["auc_roc"]
    print(f"\nBest model: {best_name} (AUC-ROC = {best_auc:.4f})")

    best_info = trained_models[best_name]
    joblib.dump(best_info["model"], os.path.join(OUTPUTS_DIR, "best_model.joblib"))
    joblib.dump(scaler, os.path.join(OUTPUTS_DIR, "scaler.joblib"))
    joblib.dump(feature_names, os.path.join(OUTPUTS_DIR, "feature_names.joblib"))

    # --- Plots ---
    _plot_roc_curves(results, y_test)
    _plot_pr_curves(results, y_test)
    _plot_confusion_matrices(results)

    # --- SHAP ---
    _generate_shap(trained_models, feature_names, X_test, X_test_scaled, best_name)

    # --- Threshold optimization ---
    best_probs = results[best_name]["y_prob"]
    _threshold_optimization(best_probs, y_test, best_name)

    return results


def _plot_roc_curves(results, y_test):
    """Plot ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={r['auc_roc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves - model comparison")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, "roc_curves.png"), dpi=150)
    plt.close(fig)
    print("Saved ROC curves plot.")


def _plot_pr_curves(results, y_test):
    """Plot precision-recall curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, r in results.items():
        prec, rec, _ = precision_recall_curve(y_test, r["y_prob"])
        ax.plot(rec, prec, label=f"{name} (PR-AUC={r['pr_auc']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves - model comparison")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, "pr_curves.png"), dpi=150)
    plt.close(fig)
    print("Saved precision-recall curves plot.")


def _plot_confusion_matrices(results):
    """Plot confusion matrices side by side."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]
    for ax, (name, r) in zip(axes, results.items()):
        sns.heatmap(
            r["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Failure"],
            yticklabels=["Normal", "Failure"],
            ax=ax,
        )
        ax.set_title(f"{name}")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
    fig.suptitle("Confusion matrices", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, "confusion_matrices.png"), dpi=150)
    plt.close(fig)
    print("Saved confusion matrices plot.")


def _generate_shap(trained_models, feature_names, X_test, X_test_scaled,
                   best_name):
    """Generate SHAP summary and waterfall plots for the best model."""
    print("\nGenerating SHAP explanations...")

    info = trained_models[best_name]
    model = info["model"]
    X = X_test_scaled if info["needs_scaling"] else X_test

    sample_size = min(500, X.shape[0])
    np.random.seed(RANDOM_STATE)
    idx = np.random.choice(X.shape[0], sample_size, replace=False)
    X_sample = X[idx]

    X_df = pd.DataFrame(X_sample, columns=feature_names)

    if info["needs_scaling"]:
        explainer = shap.LinearExplainer(model, X_sample)
    else:
        explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_df, show=False, max_display=11)
    plt.title("SHAP feature impact on failure prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "shap_summary.png"), dpi=150,
                bbox_inches="tight")
    plt.close("all")
    print("Saved SHAP summary plot.")

    # Waterfall plot for a single failure prediction
    try:
        probs = model.predict_proba(X_sample)[:, 1]
        fail_idx = np.where(probs > 0.5)[0]
        if len(fail_idx) > 0:
            single_idx = fail_idx[0]
        else:
            single_idx = np.argmax(probs)

        if isinstance(shap_values, list):
            base_val = explainer.expected_value[1] if isinstance(
                explainer.expected_value, (list, np.ndarray)
            ) else explainer.expected_value
        else:
            base_val = explainer.expected_value if np.isscalar(
                explainer.expected_value
            ) else explainer.expected_value[0]

        explanation = shap.Explanation(
            values=shap_vals[single_idx],
            base_values=base_val,
            data=X_sample[single_idx],
            feature_names=feature_names,
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(explanation, show=False, max_display=11)
        plt.title("SHAP waterfall - single sensor reading")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_DIR, "shap_waterfall.png"), dpi=150,
                    bbox_inches="tight")
        plt.close("all")
        print("Saved SHAP waterfall plot.")
    except Exception as e:
        print(f"Waterfall plot skipped: {e}")

    np.save(os.path.join(OUTPUTS_DIR, "shap_values.npy"), shap_vals)
    np.save(os.path.join(OUTPUTS_DIR, "shap_sample.npy"), X_sample)
    print("Saved SHAP values to disk.")


def _threshold_optimization(y_prob, y_test, model_name):
    """
    Optimize classification threshold based on maintenance costs.

    Assumptions:
    - False negative cost (missed failure / unplanned downtime): $15,000
    - False positive cost (unnecessary preventive maintenance): $1,500
    - Goal: minimize total cost while maintaining high recall
    """
    print("\n" + "=" * 70)
    print("THRESHOLD OPTIMIZATION (COST-BENEFIT ANALYSIS)")
    print("=" * 70)

    fn_cost = 15000   # cost of unplanned downtime
    fp_cost = 1500    # cost of unnecessary preventive maintenance

    thresholds = np.arange(0.05, 0.96, 0.01)
    impact_records = []

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_test, y_pred_t)
        tn, fp, fn, tp = cm.ravel()

        total_cost = (fn * fn_cost) + (fp * fp_cost)
        recall_t = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision_t = tp / (tp + fp) if (tp + fp) > 0 else 0
        fpr_t = fp / (fp + tn) if (fp + tn) > 0 else 0

        impact_records.append({
            "threshold": round(t, 3),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "recall": round(recall_t, 4),
            "precision": round(precision_t, 4),
            "false_positive_rate": round(fpr_t, 4),
            "total_cost": total_cost,
        })

    impact_df = pd.DataFrame(impact_records)

    optimal_idx = impact_df["total_cost"].idxmin()
    optimal = impact_df.loc[optimal_idx]

    print(f"\nModel: {model_name}")
    print(f"Unplanned downtime cost (FN): ${fn_cost:,}")
    print(f"Preventive maintenance cost (FP): ${fp_cost:,}")
    print(f"\nOptimal threshold: {optimal['threshold']:.3f}")
    print(f"  Recall (failures caught): {optimal['recall']:.1%}")
    print(f"  Precision:                {optimal['precision']:.1%}")
    print(f"  False positive rate:      {optimal['false_positive_rate']:.1%}")
    print(f"  Total cost:               ${int(optimal['total_cost']):,}")

    impact_df.to_csv(os.path.join(OUTPUTS_DIR, "threshold_analysis.csv"),
                     index=False)

    # Plot cost curve
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(impact_df["threshold"], impact_df["total_cost"], "b-o",
             markersize=3, label="Total cost ($)")
    ax1.axvline(x=optimal["threshold"], color="red", linestyle="--", alpha=0.7,
                label=f"Optimal ({optimal['threshold']:.3f})")
    ax1.set_xlabel("Classification threshold")
    ax1.set_ylabel("Total cost ($)", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(impact_df["threshold"], impact_df["recall"], "g--s",
             markersize=3, alpha=0.7, label="Recall")
    ax2.set_ylabel("Recall", color="g")
    ax2.tick_params(axis="y", labelcolor="g")
    ax2.legend(loc="upper right")

    plt.title(f"Threshold optimization - {model_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, "threshold_optimization.png"), dpi=150)
    plt.close(fig)
    print("Saved threshold optimization plot.")

    return impact_df, optimal


if __name__ == "__main__":
    from data_loader import load_and_prepare
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare()
    train_and_evaluate(X_train, X_test, y_train, y_test, feature_names)
