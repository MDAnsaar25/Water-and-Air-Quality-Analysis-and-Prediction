"""
Disease risk layer: research-backed rules + ML risk predictor.

Approach:
  1. RULES   -> AQI/WQI level-ku disease risk (research-backed thresholds)
  2. ML      -> rules-la labels uruvaaki, RandomForest train panni,
                pollution value-ku risk level predict pannum.

Air-borne (PM2.5/O3/NO2 driven): Asthma, Bronchitis, COPD, Pneumonia, Lung infection
Water-borne (low WQI driven): Cholera, Typhoid, Diarrhea, Hepatitis A, Dysentery
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# ============================================================
#  DISEASE KNOWLEDGE BASE (research-backed)
# ============================================================
AIR_DISEASES = {
    "low": [],
    "moderate": ["Mild asthma aggravation", "Throat/eye irritation"],
    "high": ["Asthma attacks", "Acute bronchitis", "COPD exacerbation"],
    "severe": ["Severe asthma", "COPD", "Pneumonia", "Lung infection", "Cardiac stress"],
}

WATER_DISEASES = {
    "low": [],
    "moderate": ["Mild gastrointestinal upset"],
    "high": ["Diarrhea", "Dysentery", "Giardiasis"],
    "severe": ["Cholera", "Typhoid", "Hepatitis A", "Severe dysentery"],
}

AIR_PRECAUTIONS = {
    "low": ["Air quality is good — enjoy outdoor activities."],
    "moderate": ["Sensitive groups: limit prolonged outdoor exertion.",
                 "Keep windows closed during peak traffic hours."],
    "high": ["Wear an N95 mask outdoors.",
             "Avoid outdoor exercise.",
             "Use an air purifier indoors.",
             "Asthma/COPD patients: keep inhaler/medication ready."],
    "severe": ["Stay indoors; seal windows.",
               "Run air purifier continuously.",
               "Avoid ALL outdoor activity.",
               "Vulnerable groups: seek medical advice if breathing issues.",
               "Follow local emergency health advisories."],
}

WATER_PRECAUTIONS = {
    "low": ["Water quality is acceptable."],
    "moderate": ["Prefer filtered or boiled water for drinking."],
    "high": ["Boil water before drinking.",
             "Avoid raw food washed in tap water.",
             "Use water purification tablets if needed."],
    "severe": ["Do NOT drink tap water — use bottled/boiled only.",
               "Maintain strict hand hygiene.",
               "Watch for diarrhea/fever — seek care early.",
               "Report contamination to local authorities."],
}


# ============================================================
#  RULES: pollution value -> risk level
# ============================================================
def air_risk_level(aqi):
    """AQI -> risk level (atikam = mosam)."""
    if aqi <= 50:
        return "low"
    elif aqi <= 100:
        return "moderate"
    elif aqi <= 200:
        return "high"
    else:
        return "severe"


def water_risk_level(wqi):
    """WQI -> risk level (kammi = mosam, REVERSE)."""
    if wqi >= 71:
        return "low"
    elif wqi >= 51:
        return "moderate"
    elif wqi >= 26:
        return "high"
    else:
        return "severe"


RISK_ORDER = ["low", "moderate", "high", "severe"]
RISK_COLOR = {"low": "#009966", "moderate": "#ffde33",
              "high": "#ff9933", "severe": "#cc0033"}


# ============================================================
#  ML: train a risk predictor on rule-generated labels
# ============================================================
def train_risk_model(history, kind="air", seed=42):
    """
    history : DataFrame[ds, y]  (y = AQI or WQI)
    kind    : "air" or "water"
    Rules-la label uruvaaki, RandomForest train pannum.
    """
    df = history.copy()
    rule_fn = air_risk_level if kind == "air" else water_risk_level
    df["risk"] = df["y"].apply(rule_fn)

    # features: value + recent context
    df["lag1"] = df["y"].shift(1).fillna(df["y"])
    df["roll6"] = df["y"].rolling(6).mean().bfill()
    df["roll24"] = df["y"].rolling(24).mean().bfill()
    feats = ["y", "lag1", "roll6", "roll24"]

    # labels -> numbers
    label_map = {r: i for i, r in enumerate(RISK_ORDER)}
    df["label"] = df["risk"].map(label_map)

    X, ylab = df[feats], df["label"]
    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=seed, n_jobs=2)
    model.fit(X, ylab)
    acc = (model.predict(X) == ylab).mean()
    return model, feats, acc


def predict_risk(model, feats, value, recent_values):
    """
    value         : forecasted/current pollution value
    recent_values : list of recent values (for lag/roll features)
    """
    rv = list(recent_values) + [value]
    lag1 = rv[-2] if len(rv) >= 2 else value
    roll6 = np.mean(rv[-6:])
    roll24 = np.mean(rv[-24:])
    x = pd.DataFrame([[value, lag1, roll6, roll24]], columns=feats)
    idx = int(model.predict(x)[0])
    proba = model.predict_proba(x)[0]
    return RISK_ORDER[idx], proba


def get_diseases_and_precautions(risk, kind="air"):
    diseases = (AIR_DISEASES if kind == "air" else WATER_DISEASES)[risk]
    precautions = (AIR_PRECAUTIONS if kind == "air" else WATER_PRECAUTIONS)[risk]
    return diseases, precautions