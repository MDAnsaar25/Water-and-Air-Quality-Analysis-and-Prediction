"""
Water data layer: synthetic global WQI (Water Quality Index).
Air-oda data.py maathiri, aana WQI-ku tweak pannirukku.
WQI: 0 (worst) to 100 (best) — AQI-ku reverse!
"""
import numpy as np
import pandas as pd
from datetime import datetime
import config


# ---------------- SYNTHETIC LIVE WQI ----------------
def fetch_all_live():
    """Ellaa cities-kum oru synthetic 'live' WQI generate panrathu."""
    rng = np.random.default_rng()  # seed illa -> ovvoru refresh-kum konjam maarum
    rows = []
    for name, lat, lon in config.MONITOR_CITIES:
        # base WQI: konjam realistic-a, city-oda hash vechu stable base
        base = 40 + (hash(name) % 45)          # 40-85 range base
        wqi = float(np.clip(base + rng.normal(0, 8), 0, 100))
        rows.append({
            "city": name,
            "lat": lat,
            "lon": lon,
            "wqi": round(wqi, 1),
            "do": round(rng.uniform(4, 9), 1),      # dissolved oxygen (mg/L)
            "ph": round(rng.uniform(6.2, 8.6), 1),   # pH level
            "turbidity": round(rng.uniform(1, 40), 1),  # NTU
            "time": str(datetime.utcnow()),
        })
    return pd.DataFrame(rows)


# ---------------- HISTORY (model training-ku) ----------------
def build_history(seed_wqi=70, days=120, freq_hours=1, seed=42):
    """
    Synthetic hourly WQI history. Air maathiri patterns,
    aana WQI slow ah maarum (water quality thideer-a maarathu).
    """
    rng = np.random.default_rng(seed)
    n = int(days * 24 / freq_hours)
    end = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    idx = pd.date_range(end=end, periods=n, freq=f"{freq_hours}h")

    t = np.arange(n)
    base = seed_wqi + 10 * np.sin(2 * np.pi * t / (24 * 45))  # slow seasonal drift
    weekly = -5 * (idx.dayofweek < 5).astype(float)          # weekday industrial discharge -> WQI kammi
    noise = rng.normal(0, 4, n)                              # water-la noise kammi (stable)
    # pollution events: WQI thideer-a KEEZHA vizhum (sewage spill, factory dump)
    drops = np.zeros(n)
    drop_idx = rng.choice(n, size=max(3, n // 400), replace=False)
    drops[drop_idx] = -rng.uniform(20, 45, len(drop_idx))

    wqi = base + weekly + noise + drops
    wqi = np.clip(wqi, 0, 100)

    return pd.DataFrame({"ds": idx, "y": wqi})