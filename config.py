"""
Configuration & constants.
Get your FREE WAQI token at: https://aqicn.org/data-platform/token/
Then set it below OR as an environment variable: WAQI_TOKEN
"""
import os

# ---- API CONFIG ----
WAQI_TOKEN = os.getenv("WAQI_TOKEN", "7fadf310029859e2c5c4403e890fceb3a4349d9c")  # replace "demo" with your real token
WAQI_BASE = "https://api.waqi.info"

# ---- AQI CATEGORY THRESHOLDS (US EPA) ----
AQI_CATEGORIES = [
    (0, 50, "Good", "#009966"),
    (51, 100, "Moderate", "#ffde33"),
    (101, 150, "Unhealthy (Sensitive)", "#ff9933"),
    (151, 200, "Unhealthy", "#cc0033"),
    (201, 300, "Very Unhealthy", "#660099"),
    (301, 500, "Hazardous", "#7e0023"),
]

# Cities to monitor (name, lat, lon) — used for the live world map
MONITOR_CITIES = [
    ("Delhi", 28.6139, 77.2090),
    ("Chennai", 13.0827, 80.2707),
    ("Mumbai", 19.0760, 72.8777),
    ("Beijing", 39.9042, 116.4074),
    ("London", 51.5074, -0.1278),
    ("New York", 40.7128, -74.0060),
    ("Tokyo", 35.6762, 139.6503),
    ("Los Angeles", 34.0522, -118.2437),
    ("Lahore", 31.5204, 74.3587),
    ("Jakarta", -6.2088, 106.8456),
    ("Sao Paulo", -23.5505, -46.6333),
    ("Cairo", 30.0444, 31.2357),
]

MODEL_DIR = "models"


def aqi_category(aqi):
    for lo, hi, name, color in AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return name, color
    return "Beyond Index", "#7e0023"

# ---- WQI (Water Quality Index) CATEGORIES ----
# Note: WQI-la ATHIGAM = NALLA (reverse of AQI)
WQI_CATEGORIES = [
    (91, 100, "Excellent",  "#009966"),
    (71, 90,  "Good",       "#a3d977"),
    (51, 70,  "Medium",     "#ffde33"),
    (26, 50,  "Poor",       "#ff9933"),
    (0,  25,  "Very Poor",  "#cc0033"),
]


def wqi_category(wqi):
    for lo, hi, name, color in WQI_CATEGORIES:
        if lo <= wqi <= hi:
            return name, color
    return "Unknown", "#888888"