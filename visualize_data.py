"""
build_history() data-va chart-a paaka oru chinna test script.
Run: python visualize_data.py
"""
import matplotlib.pyplot as plt
from utils import data as D

# 1. History build pannu (Chennai-oda AQI ~150 nu vechukalaam)
hist = D.build_history(seed_aqi=150, days=120)

print("Total data points:", len(hist))
print("\nFirst 5 rows:")
print(hist.head())
print("\nAQI stats:")
print(hist["y"].describe())

# 2. Full history plot
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Mela: full 120 days
axes[0].plot(hist["ds"], hist["y"], color="#5b9cff", linewidth=0.7)
axes[0].set_title("Full 120-day AQI History (synthetic)")
axes[0].set_ylabel("AQI")
axes[0].grid(alpha=0.3)

#last 7 days mattum (daily pattern)
last_week = hist.tail(24 * 7)
axes[1].plot(last_week["ds"], last_week["y"], color="#ff9933", linewidth=1.2)
axes[1].set_title("Last 7 Days (daily up-down pattern theriyum)")
axes[1].set_ylabel("AQI")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("aqi_history.png", dpi=100)
print("\nChart saved as aqi_history.png")
plt.show()