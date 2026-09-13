import os
import json
from datetime import datetime
import requests

def save_payload(source_name, precip_window):
    os.makedirs("data/processed", exist_ok=True)
    payload = {
        "source": source_name,
        "retrieved_at": datetime.now().isoformat(),
        "intervals": [15, 30, 45, 60, 75, 90, 105, 120],
        "precipitation_mm": [float(p) for p in precip_window]
    }
    out_path = "data/processed/rainfall_nowcast.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[{source_name}] Updated {out_path}: {precip_window}")

def fetch_live_rainfall():
    lat, lon = 19.0182, 72.8434
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&minutely_15=precipitation&forecast_days=1&timezone=Asia%2FKolkata"
    )
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()
        m15_times = data.get("minutely_15", {}).get("time", [])
        m15_precip = data.get("minutely_15", {}).get("precipitation", [])
        
        now_iso = datetime.now().strftime("%Y-%m-%dT%H:%M")
        start_idx = 0
        for i, t in enumerate(m15_times):
            if t >= now_iso:
                start_idx = i
                break
        window = [float(p) for p in m15_precip[start_idx : start_idx + 8]]
        while len(window) < 8:
            window.append(0.0)
        save_payload("Open-Meteo Live", window)
    except Exception as e:
        print(f"Failed live pull: {e}. Falling back to nominal rain.")
        save_payload("Open-Meteo Fallback", [2.0, 4.0, 3.5, 1.0, 0.5, 0.0, 0.0, 0.0])

def generate_scenarios():
    scenarios = {
        "live": None, # dynamically pulled
        "monsoon_heavy": [12.0, 18.5, 26.0, 22.0, 15.0, 9.0, 4.0, 2.0],
        "cloudburst": [18.0, 42.0, 78.0, 65.0, 38.0, 20.0, 10.0, 4.0]  # Mumbai extreme event curve
    }
    with open("data/processed/scenarios.json", "w") as f:
        json.dump(scenarios, f, indent=2)

if __name__ == "__main__":
    generate_scenarios()
    fetch_live_rainfall()