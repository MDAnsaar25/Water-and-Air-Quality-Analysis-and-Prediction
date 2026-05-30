"""
Data layer: live AQI fetch (WAQI) + history builder for ML training.
"""
import requests
import numpy as np
import pandas as pd
from datetime import datetime
import config


# ---------------- LIVE FETCH ----------------
def fetch_city_live(city_name, lat, lon):
    """Oru city-oda live AQI fetch panrathu. Return dict, illana None."""
    url = f"{config.WAQI_BASE}/feed/geo:{lat};{lon}/"
    try:
        r = requests.get(url, params={"token": config.WAQI_TOKEN}, timeout=10)
        data = r.json()
        if data.get("status") != "ok":
            return None
        d = data["data"]
        aqi = d.get("aqi")
        if not isinstance(aqi, (int, float)):
            return None
        iaqi = d.get("iaqi", {})
        return {
            "city": city_name,
            "lat": lat,
            "lon": lon,
            "aqi": float(aqi),
            "dominant": d.get("dominentpol", "—"),
            "pm25": iaqi.get("pm25", {}).get("v"),
            "pm10": iaqi.get("pm10", {}).get("v"),
            "o3": iaqi.get("o3", {}).get("v"),
            "no2": iaqi.get("no2", {}).get("v"),
            "time": d.get("time", {}).get("s", str(datetime.utcnow())),
        }
    except Exception as e:
        print(f"[fetch error] {city_name}: {e}")
        return None


def fetch_all_live():
    """Ellaa monitored cities-um fetch panni DataFrame return panrathu."""
    rows = []
    for name, lat, lon in config.MONITOR_CITIES:
        rec = fetch_city_live(name, lat, lon)
        if rec:
            rows.append(rec)
    return pd.DataFrame(rows)


# ---------------- HISTORY (model training-ku) ----------------
def build_history(seed_aqi=120, days=120, freq_hours=1, seed=42):
    """
    Realistic synthetic hourly AQI history generate panrathu.
    Live AQI value-a vechu seed pannrom, so forecast continuous ah இருக்கும்.
    """
    rng = np.random.default_rng(seed)
    n = int(days * 24 / freq_hours)
    end = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    idx = pd.date_range(end=end, periods=n, freq=f"{freq_hours}h")

    t = np.arange(n)
    base = seed_aqi + 15 * np.sin(2 * np.pi * t / (24 * 30))   # slow drift
    daily = 25 * np.sin(2 * np.pi * (idx.hour) / 24 - 1.0)     # daily cycle
    weekly = 12 * (idx.dayofweek < 5).astype(float)           # weekday traffic
    noise = rng.normal(0, 10, n)                              # random noise
    spikes = np.zeros(n)                                      # pollution spikes
    spike_idx = rng.choice(n, size=max(3, n // 400), replace=False)
    spikes[spike_idx] = rng.uniform(60, 160, len(spike_idx))

    aqi = base + daily + weekly + noise + spikes
    aqi = np.clip(aqi, 5, 500)

    return pd.DataFrame({"ds": idx, "y": aqi})