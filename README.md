# 🌍 Live World Pollution + ML Dashboard

An end-to-end, interactive **Streamlit** dashboard that visualizes **live air & water quality** worldwide and applies **three machine-learning models** for forecasting, classification, and anomaly detection.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Features

- 📡 **Live Air Quality** — real-time AQI from the WAQI API across 12 global cities, on an interactive world map
- 💧 **Water Quality** — global Water Quality Index (WQI) with pH, dissolved oxygen & turbidity
- 🤖 **Three ML models**, applied to both air and water:
  | Task | Model | Output |
  |------|-------|--------|
  | Forecasting | Prophet | Next 24–72 h prediction with confidence band |
  | Classification | XGBoost | Quality category + class probabilities |
  | Anomaly detection | IsolationForest | Pollution spikes / drops |
- 🎛️ Fully interactive: pick a city, adjust forecast horizon, anomaly sensitivity, auto-refresh

## 🖥️ Tech Stack

`Streamlit` · `Plotly` · `pydeck` · `Prophet` · `XGBoost` · `scikit-learn` · `pandas` · `NumPy`

## 🚀 Getting Started

### 1. Clone & enter
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your WAQI token (free)
Get a free token at [aqicn.org/data-platform/token](https://aqicn.org/data-platform/token/), then:
```bash
# Windows (PowerShell)
$env:WAQI_TOKEN="your_token_here"
# macOS / Linux
export WAQI_TOKEN="your_token_here"
```
> Without a token, the app falls back to the limited `demo` token.

### 5. Run
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

## 📁 Project Structure
```
.
├── app.py              # Main Streamlit dashboard (Air + Water tabs)
├── config.py           # Settings, AQI/WQI categories, monitored cities
├── requirements.txt
├── utils/
│   ├── data.py         # Live air (WAQI) + history builder
│   └── water.py        # Synthetic water quality data
└── models/
    └── ml.py           # Prophet · XGBoost · IsolationForest
```

## 📝 Notes

- The WAQI free tier provides **live** AQI but not long history, so the time-series models train on a realistic synthetic history seeded by the current live value. Swap `utils/data.py → build_history()` with a real historical source (e.g. OpenAQ) to use true data.
- Water quality data is **synthetic** (global live WQI APIs are not freely available) — designed to demonstrate the ML pipeline. The architecture is ready to drop in a real source (USGS, GEMStat, etc.).

## 📜 License

MIT