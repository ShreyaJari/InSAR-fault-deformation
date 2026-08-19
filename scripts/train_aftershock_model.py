"""
Train a gradient boosting classifier to predict aftershock occurrence
per grid cell / time window, following the framing in DeVries et al.
(2018, Nature) - "Deep learning of aftershock patterns following large
earthquakes."

Usage:
    python3 scripts/train_aftershock_model.py
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import matplotlib.pyplot as plt

DATA_DIR = Path("data")
IN_CSV = DATA_DIR / "aftershock_grid_features.csv"
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIME_BIN_MAP = {"0-1d": 0.5, "1-3d": 2, "3-7d": 5, "7-14d": 10.5, "14-30d": 22}


def load_data():
    df = pd.read_csv(IN_CSV)
    df["time_bin_days"] = df["time_bin"].map(TIME_BIN_MAP)
    features = ["grid_lon", "grid_lat", "dist_from_mainshock_km", "time_bin_days"]
    X = df[features]
    y = df["had_aftershock"]
    return df, X, y, features


def train_and_evaluate():
    df, X, y, features = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Handle class imbalance: weight positive class by the inverse of its
    # frequency, standard practice for XGBoost on imbalanced targets.
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    print("=== Evaluation on held-out test set ===")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")
    print(f"PR-AUC (average precision): {average_precision_score(y_test, y_prob):.3f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["no aftershock", "aftershock"]))

    print("\n=== Feature importance ===")
    for feat, imp in sorted(
        zip(features, model.feature_importances_), key=lambda x: -x[1]
    ):
        print(f"  {feat}: {imp:.3f}")

    # Predict probability over the full grid for one time window, plot it
    # against the observed distribution for a visual sanity check.
    window = "1-3d"
    sub = df[df["time_bin"] == window].copy()
    sub["predicted_prob"] = model.predict_proba(sub[features])[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)

    sc0 = axes[0].scatter(
        sub["grid_lon"], sub["grid_lat"], c=sub["had_aftershock"],
        cmap="Reds", s=25, vmin=0, vmax=1
    )
    axes[0].set_title(f"Observed aftershocks, {window}")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")

    sc1 = axes[1].scatter(
        sub["grid_lon"], sub["grid_lat"], c=sub["predicted_prob"],
        cmap="Reds", s=25, vmin=0, vmax=1
    )
    axes[1].set_title(f"Model-predicted probability, {window}")
    axes[1].set_xlabel("Longitude")
    fig.colorbar(sc1, ax=axes, label="Probability / occurrence", shrink=0.8)

    fig.savefig(OUT_DIR / "aftershock_forecast_comparison.png", dpi=200,
                bbox_inches="tight")
    print(f"\nSaved comparison figure to {OUT_DIR / 'aftershock_forecast_comparison.png'}")

    return model


if __name__ == "__main__":
    train_and_evaluate()