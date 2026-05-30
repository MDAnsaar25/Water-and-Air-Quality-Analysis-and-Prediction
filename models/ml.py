"""
ML layer: all three tasks.
  1. FORECAST   -> Prophet (next N hours AQI)
  2. CLASSIFY   -> XGBoost (AQI category from pollutant features)
  3. ANOMALY    -> IsolationForest (spike detection on history)

Models are trained on demand and cached. Each function is independent so the
Streamlit app can call only what it needs.
"""
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
import config


# ---------------- 1. FORECAST ----------------
def train_forecast(history: pd.DataFrame):
    """history: DataFrame[ds, y]. Returns fitted Prophet model."""
    m = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.05,
        interval_width=0.85,
    )
    m.fit(history)
    return m


def forecast(model, hours=72):
    future = model.make_future_dataframe(periods=hours, freq="h")
    fc = model.predict(future)
    return fc[["ds", "yhat", "yhat_lower", "yhat_upper"]]


# ---------------- 2. CLASSIFICATION ----------------
def _value_to_class(value, categories):
    for i, (lo, hi, name, _) in enumerate(categories):
        if lo <= value <= hi:
            return i
    return len(categories) - 1


def train_classifier(history: pd.DataFrame, seed=42, categories=None):
    """
    Build a tabular dataset from the history (lag + time features) and train
    XGBoost to predict the quality category.
    `categories` defaults to AQI; pass WQI_CATEGORIES for water.
    """
    if categories is None:
        categories = config.AQI_CATEGORIES
    df = history.copy()
    df["hour"] = df["ds"].dt.hour
    df["dow"] = df["ds"].dt.dayofweek
    df["lag1"] = df["y"].shift(1)
    df["lag3"] = df["y"].shift(3)
    df["lag24"] = df["y"].shift(24)
    df["roll6"] = df["y"].rolling(6).mean()
    df = df.dropna().reset_index(drop=True)

    df["raw_label"] = df["y"].apply(lambda v: _value_to_class(v, categories))
    feats = ["hour", "dow", "lag1", "lag3", "lag24", "roll6"]

    # Remap present classes to a contiguous 0-based range (XGBoost requirement).
    present = sorted(df["raw_label"].unique())
    label_map = {orig: i for i, orig in enumerate(present)}
    inv_map = {i: orig for orig, i in label_map.items()}  # model idx -> real AQI class
    df["label"] = df["raw_label"].map(label_map)

    X, y = df[feats], df["label"]
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=max(len(present), 2),
        random_state=seed,
        n_jobs=2,
        eval_metric="mlogloss",
    )
    model.fit(X, y)
    acc = (model.predict(X) == y).mean()
    return model, feats, acc, inv_map


def classify_latest(model, feats, history, inv_map):
    df = history.copy()
    df["hour"] = df["ds"].dt.hour
    df["dow"] = df["ds"].dt.dayofweek
    df["lag1"] = df["y"].shift(1)
    df["lag3"] = df["y"].shift(3)
    df["lag24"] = df["y"].shift(24)
    df["roll6"] = df["y"].rolling(6).mean()
    df = df.dropna()
    x = df[feats].iloc[[-1]]
    proba = model.predict_proba(x)[0]
    model_idx = int(np.argmax(proba))
    real_cls = inv_map[model_idx]  # map back to true AQI category index
    # spread probabilities onto the real category axis
    full_proba = {inv_map[i]: p for i, p in enumerate(proba)}
    return real_cls, full_proba


# ---------------- 3. ANOMALY ----------------
def detect_anomalies(history: pd.DataFrame, contamination=0.02, seed=42):
    """Flag pollution spikes. Returns history with 'anomaly' bool column."""
    df = history.copy()
    df["lag1"] = df["y"].shift(1).fillna(df["y"])
    df["delta"] = df["y"].diff().fillna(0)
    df["roll6"] = df["y"].rolling(6).mean().bfill()
    df["dev"] = df["y"] - df["roll6"]

    feats = ["y", "delta", "dev"]
    iso = IsolationForest(
        contamination=contamination, random_state=seed, n_estimators=200
    )
    df["anomaly"] = iso.fit_predict(df[feats]) == -1
    df["score"] = iso.decision_function(df[feats])
    return df