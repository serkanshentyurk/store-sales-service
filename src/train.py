"""Train the store-sales model and persist the fitted pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from src.data import load_clean_data
from src.features import add_features, build_preprocessor

import mlflow

DATA_PATH = "data"
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
MODEL_PATH = MODEL_DIR / "pipeline.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"
HOLDOUT_WEEKS = 8
RANDOM_STATE = 42
TARGET = "Sales"


def rmspe(y_true, y_pred) -> float:
    """Root Mean Squared Percentage Error, ignoring zero-sales rows."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    return float(np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2)))


def temporal_split(df: pd.DataFrame, holdout_weeks: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split at the last `holdout_weeks` of data; assert no temporal overlap."""
    cutoff = df["Date"].max() - pd.to_timedelta(holdout_weeks, unit="W")
    train_df = df[df["Date"] < cutoff]
    test_df = df[df["Date"] >= cutoff]
    assert train_df["Date"].max() < test_df["Date"].min()
    return train_df, test_df


def build_pipeline() -> Pipeline:
    """Preprocessor + log-target gradient boosting regressor."""
    model = TransformedTargetRegressor(
        regressor=HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    return Pipeline(steps=[("prep", build_preprocessor()), ("model", model)])


def baseline_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    """Per-store x day-of-week mean sales, learned on train only."""
    means = train_df.groupby(["Store", "DayOfWeek"])[TARGET].mean()
    preds = test_df.set_index(["Store", "DayOfWeek"]).index.map(means)
    return pd.Series(preds, index=test_df.index).fillna(train_df[TARGET].mean())


def main() -> None:
    df = add_features(load_clean_data(DATA_PATH))
    train_df, test_df = temporal_split(df, HOLDOUT_WEEKS)

    with mlflow.start_run():
        pipe = build_pipeline()
        pipe.fit(train_df, train_df[TARGET])

        model_pred = pipe.predict(test_df)
        base_pred = baseline_predict(train_df, test_df)

        metrics = {
            "model_rmspe": round(rmspe(test_df[TARGET], model_pred), 4),
            "model_mae": round(mean_absolute_error(test_df[TARGET], model_pred), 1),
            "baseline_rmspe": round(rmspe(test_df[TARGET], base_pred), 4),
            "baseline_mae": round(mean_absolute_error(test_df[TARGET], base_pred), 1),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
        }

        mlflow.log_param("holdout_weeks", HOLDOUT_WEEKS)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("model", "HistGradientBoostingRegressor")

        mlflow.log_metric("model_rmspe", metrics["model_rmspe"])
        mlflow.log_metric("model_mae", metrics["model_mae"])
        mlflow.log_metric("baseline_rmspe", metrics["baseline_rmspe"])
        mlflow.log_metric("baseline_mae", metrics["baseline_mae"])

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, MODEL_PATH)
        METRICS_PATH.write_text(json.dumps(metrics, indent=2))

        print(f"model    RMSPE {metrics['model_rmspe']}  MAE {metrics['model_mae']}")
        print(f"baseline RMSPE {metrics['baseline_rmspe']}  MAE {metrics['baseline_mae']}")
        print(f"model -> {MODEL_PATH}")
        print(f"metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()