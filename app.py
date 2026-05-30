"""
🌍 Live World Pollution + ML Dashboard (Air + Water)
Run:  streamlit run app.py

  • Air: WAQI live AQI  |  Water: synthetic global WQI
  • Both run Prophet (forecast) + XGBoost (classify) + IsolationForest (anomaly)
"""
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import pydeck as pdk

import config
from utils import data as D       # air
from utils import water as W      # water
from models import ml as ML
from models import disease as DIS

st.set_page_config(page_title="Live Pollution ML Dashboard", page_icon="🌍", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem;}
h1,h2,h3 {letter-spacing:-0.5px;}
</style>
""", unsafe_allow_html=True)

st.title("🌍 Live World Pollution + ML Dashboard")
st.caption("Air (WAQI live) + Water (synthetic WQI) · Prophet · XGBoost · IsolationForest")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Controls")
    if config.WAQI_TOKEN == "demo":
        st.warning("Using WAQI **demo** token (air data limited). Get a free token at aqicn.org/data-platform/token.")
    horizon = st.slider("Forecast horizon (hours)", 12, 72, 48, 6)
    hist_days = st.slider("History window (days)", 30, 180, 120, 10)
    contam = st.slider("Anomaly sensitivity", 0.005, 0.06, 0.02, 0.005)
    auto = st.checkbox("Auto-refresh live data", value=False)
    interval = st.select_slider("Refresh every (sec)", [30, 60, 120, 300], value=60)
    st.divider()
    st.caption("Data: WAQI + synthetic WQI · Models: Prophet / XGBoost / IsolationForest")


# ---------------- DATA (cached) ----------------
@st.cache_data(ttl=60, show_spinner="Fetching live air quality…")
def get_air():
    return D.fetch_all_live()

@st.cache_data(ttl=60, show_spinner="Generating water quality…")
def get_water():
    return W.fetch_all_live()


# =========================================================
#  REUSABLE ML RENDERER  (air + water rendaiyum idhe use pannum)
# =========================================================
def render_ml_tabs(hist, value_label, categories, cat_func, higher_is_worse):
    """
    hist          : DataFrame[ds, y]
    value_label   : "AQI" or "WQI"
    categories    : config.AQI_CATEGORIES or WQI_CATEGORIES
    cat_func      : config.aqi_category or wqi_category
    higher_is_worse: True for AQI (atikam = mosam), False for WQI (atikam = nalla)
    """
    t1, t2, t3 = st.tabs([f"📈 Forecast (Prophet)",
                          f"🏷️ Classify (XGBoost)",
                          f"⚠️ Anomalies (IsolationForest)"])

    # ---- FORECAST ----
    with t1:
        with st.spinner("Training Prophet & forecasting…"):
            m = ML.train_forecast(hist)
            fc = ML.forecast(m, hours=horizon)
        split = hist["ds"].iloc[-1]
        fut = fc[fc["ds"] > split]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["ds"][-240:], y=hist["y"][-240:],
                                 name="History", line=dict(color="#5b9cff")))
        fig.add_trace(go.Scatter(x=fut["ds"], y=fut["yhat"], name="Forecast",
                                 line=dict(color="#ff9933", width=3)))
        fig.add_trace(go.Scatter(
            x=list(fut["ds"]) + list(fut["ds"][::-1]),
            y=list(fut["yhat_upper"]) + list(fut["yhat_lower"][::-1]),
            fill="toself", fillcolor="rgba(255,153,51,0.15)",
            line=dict(width=0), name="85% interval"))
        fig.update_layout(template="plotly_dark", height=420,
                          title=f"Next {horizon}h {value_label} forecast",
                          yaxis_title=value_label)
        st.plotly_chart(fig, use_container_width=True)

        # AQI-la peak (max) mukyam; WQI-la trough (min) mukyam
        if higher_is_worse:
            pt = fut.loc[fut["yhat"].idxmax()]
            label = f"Predicted worst {value_label} (peak)"
        else:
            pt = fut.loc[fut["yhat"].idxmin()]
            label = f"Predicted worst {value_label} (lowest)"
        pname, _ = cat_func(pt["yhat"])
        cA, cB = st.columns(2)
        cA.metric(label, f"{pt['yhat']:.0f}", pname)
        cB.metric("Expected at", pt["ds"].strftime("%b %d, %H:%M"))

    # ---- CLASSIFY ----
    with t2:
        with st.spinner("Training XGBoost classifier…"):
            clf, feats, acc, inv_map = ML.train_classifier(
                hist, categories=categories)
            cls, full_proba = ML.classify_latest(clf, feats, hist, inv_map)
        name = categories[cls][2]
        color = categories[cls][3]
        st.markdown(f"### Current category: "
                    f"<span style='color:{color}'>{name}</span>",
                    unsafe_allow_html=True)
        st.caption(f"Model training accuracy: {acc:.1%}")
        rows = [(categories[k][2], categories[k][3], v)
                for k, v in sorted(full_proba.items())]
        pdf = pd.DataFrame(rows, columns=["Category", "clr", "Probability"])
        fig2 = go.Figure(go.Bar(x=pdf["Probability"], y=pdf["Category"],
                                orientation="h", marker_color=pdf["clr"]))
        fig2.update_layout(template="plotly_dark", height=360,
                           xaxis_title="Probability", title="Class probabilities")
        st.plotly_chart(fig2, use_container_width=True)

    # ---- ANOMALY ----
    with t3:
        with st.spinner("Detecting anomalies…"):
            adf = ML.detect_anomalies(hist, contamination=contam)
        an = adf[adf["anomaly"]]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=adf["ds"], y=adf["y"], name=value_label,
                                  line=dict(color="#5b9cff")))
        fig3.add_trace(go.Scatter(x=an["ds"], y=an["y"], mode="markers",
                                  name="Anomaly", marker=dict(color="#cc0033",
                                  size=9, symbol="x")))
        fig3.update_layout(template="plotly_dark", height=420,
                           title=f"Detected {value_label} anomalies",
                           yaxis_title=value_label)
        st.plotly_chart(fig3, use_container_width=True)
        st.metric("Anomalies detected", len(an))
        if len(an):
            st.dataframe(an[["ds", "y", "score"]].tail(15)
                         .rename(columns={"ds": "time", "y": value_label})
                         .reset_index(drop=True), use_container_width=True)


# =========================================================
#  TOP-LEVEL TABS: AIR | WATER
# =========================================================
air_tab, water_tab, alert_tab = st.tabs(["🌫️ Air Quality", "💧 Water Quality", "🚨 Disease & Alerts"])

# ---------------- AIR ----------------
with air_tab:
    air = get_air()
    if air.empty:
        st.error("No live air data. Check WAQI token / network, then refresh.")
    else:
        air["cat"], air["color"] = zip(*air["aqi"].map(config.aqi_category))
        air["rgb"] = air["color"].map(
            lambda h: [int(h.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)] + [180])

        c1, c2, c3, c4 = st.columns(4)
        worst = air.loc[air["aqi"].idxmax()]
        best = air.loc[air["aqi"].idxmin()]
        c1.metric("Cities live", len(air))
        c2.metric("🌫️ Worst", worst["city"], f"AQI {int(worst['aqi'])}")
        c3.metric("🌿 Cleanest", best["city"], f"AQI {int(best['aqi'])}")
        c4.metric("Global avg AQI", f"{air['aqi'].mean():.0f}")

        st.pydeck_chart(pdk.Deck(
            layers=[pdk.Layer("ScatterplotLayer", data=air,
                              get_position="[lon, lat]", get_fill_color="rgb",
                              get_radius="aqi * 1200", pickable=True, opacity=0.7)],
            initial_view_state=pdk.ViewState(latitude=20, longitude=40, zoom=1.2),
            tooltip={"text": "{city}\nAQI: {aqi} ({cat})\nDominant: {dominant}"},
            map_style="dark"))

        st.dataframe(air[["city", "aqi", "cat", "dominant", "pm25", "pm10"]]
                     .sort_values("aqi", ascending=False).reset_index(drop=True),
                     use_container_width=True)

        st.divider()
        st.subheader("🤖 ML Analysis — Air")
        acity = st.selectbox("City (air ML)", air["city"].tolist(),
                             index=int(air["aqi"].idxmax()), key="air_city")
        seed = float(air.loc[air["city"] == acity, "aqi"].iloc[0])
        ahist = D.build_history(seed_aqi=seed, days=hist_days)
        render_ml_tabs(ahist, "AQI", config.AQI_CATEGORIES,
                       config.aqi_category, higher_is_worse=True)

# ---------------- WATER ----------------
with water_tab:
    water = get_water()
    water["cat"], water["color"] = zip(*water["wqi"].map(config.wqi_category))
    water["rgb"] = water["color"].map(
        lambda h: [int(h.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)] + [180])

    c1, c2, c3, c4 = st.columns(4)
    # WQI-la kammi = mosam, so worst = idxmin
    worst = water.loc[water["wqi"].idxmin()]
    best = water.loc[water["wqi"].idxmax()]
    c1.metric("Cities", len(water))
    c2.metric("🛢️ Worst", worst["city"], f"WQI {worst['wqi']:.0f}")
    c3.metric("💧 Cleanest", best["city"], f"WQI {best['wqi']:.0f}")
    c4.metric("Global avg WQI", f"{water['wqi'].mean():.0f}")

    st.pydeck_chart(pdk.Deck(
        layers=[pdk.Layer("ScatterplotLayer", data=water,
                          get_position="[lon, lat]", get_fill_color="rgb",
                          get_radius="(100 - wqi) * 1500", pickable=True, opacity=0.7)],
        initial_view_state=pdk.ViewState(latitude=20, longitude=40, zoom=1.2),
        tooltip={"text": "{city}\nWQI: {wqi} ({cat})\npH: {ph}  DO: {do}"},
        map_style="dark"))

    st.dataframe(water[["city", "wqi", "cat", "ph", "do", "turbidity"]]
                 .sort_values("wqi").reset_index(drop=True),
                 use_container_width=True)

    st.divider()
    st.subheader("🤖 ML Analysis — Water")
    wcity = st.selectbox("City (water ML)", water["city"].tolist(),
                         index=int(water["wqi"].idxmin()), key="water_city")
    seed = float(water.loc[water["city"] == wcity, "wqi"].iloc[0])
    whist = W.build_history(seed_wqi=seed, days=hist_days)
    render_ml_tabs(whist, "WQI", config.WQI_CATEGORIES,
                   config.wqi_category, higher_is_worse=False)


# =========================================================
#  DISEASE & ALERTS TAB
# =========================================================
def render_alert(city_name, value, recent, kind, value_label):
    """Oru city-ku risk predict panni, alert + diseases + precautions kaattum."""
    hist = (D.build_history(seed_aqi=value, days=hist_days) if kind == "air"
            else W.build_history(seed_wqi=value, days=hist_days))
    model, feats, acc = DIS.train_risk_model(hist, kind=kind)

    # current risk (rule-based, ground truth)
    rule_fn = DIS.air_risk_level if kind == "air" else DIS.water_risk_level
    current_risk = rule_fn(value)

    # ML-predicted risk for the forecasted next-period value
    fc_model = ML.train_forecast(hist)
    fc = ML.forecast(fc_model, hours=horizon)
    future_val = (fc["yhat"].max() if kind == "air" else fc["yhat"].min())
    pred_risk, _ = DIS.predict_risk(model, feats, future_val, recent)

    color = DIS.RISK_COLOR[current_risk]
    icon = {"low": "✅", "moderate": "⚠️", "high": "🔶", "severe": "🚨"}[current_risk]

    st.markdown(
        f"<div style='background:{color}22;border-left:6px solid {color};"
        f"padding:14px 18px;border-radius:8px;margin-bottom:8px'>"
        f"<h3 style='margin:0;color:{color}'>{icon} {city_name} — {current_risk.upper()} RISK</h3>"
        f"<p style='margin:4px 0 0'>Current {value_label}: <b>{value:.0f}</b> · "
        f"Predicted next-{horizon}h worst {value_label}: <b>{future_val:.0f}</b> "
        f"(ML risk: <b>{pred_risk.upper()}</b>)</p></div>",
        unsafe_allow_html=True)
    st.caption(f"Risk model accuracy: {acc:.0%}")

    diseases, precautions = DIS.get_diseases_and_precautions(current_risk, kind)
    cL, cR = st.columns(2)
    with cL:
        st.markdown("**🦠 Potential health risks:**")
        if diseases:
            for d in diseases:
                st.markdown(f"- {d}")
        else:
            st.markdown("- No significant risk at this level.")
    with cR:
        st.markdown("**🛡️ Precautions & advice:**")
        for p in precautions:
            st.markdown(f"- {p}")


with alert_tab:
    st.subheader("🚨 Disease Risk & Health Alerts")
    st.caption("Rule-based risk (research-backed) + ML-predicted risk from pollution forecast")

    air_now = get_air()
    water_now = get_water()

    dom = st.radio("Choose domain", ["🌫️ Air-borne", "💧 Water-borne"],
                   horizontal=True, key="alert_dom")

    if dom == "🌫️ Air-borne":
        if air_now.empty:
            st.error("No live air data available.")
        else:
            ac = st.selectbox("City", air_now["city"].tolist(),
                              index=int(air_now["aqi"].idxmax()), key="alert_air_city")
            val = float(air_now.loc[air_now["city"] == ac, "aqi"].iloc[0])
            recent = D.build_history(seed_aqi=val, days=30)["y"].tail(24).tolist()
            with st.spinner("Assessing health risk…"):
                render_alert(ac, val, recent, "air", "AQI")
    else:
        wc = st.selectbox("City", water_now["city"].tolist(),
                          index=int(water_now["wqi"].idxmin()), key="alert_water_city")
        val = float(water_now.loc[water_now["city"] == wc, "wqi"].iloc[0])
        recent = W.build_history(seed_wqi=val, days=30)["y"].tail(24).tolist()
        with st.spinner("Assessing health risk…"):
            render_alert(wc, val, recent, "water", "WQI")

    st.divider()
    st.caption("⚕️ Informational only — not medical advice. Follow official health authorities.")


# ---------------- AUTO REFRESH ----------------
if auto:
    time.sleep(interval)
    st.cache_data.clear()
    st.rerun()