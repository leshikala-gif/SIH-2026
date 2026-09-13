import time
import subprocess
import sys
from datetime import datetime

POLL_INTERVAL_SECONDS = 900  # 15 minutes

def cycle():
    print("\n" + "=" * 50)
    print(f"NOWCAST UPDATE CYCLE STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    # 1. Fetch live rainfall
    res_rain = subprocess.run([sys.executable, "src/pipeline/fetch_rainfall.py"])
    if res_rain.returncode != 0:
        print("[WARN] Rainfall fetch encountered an issue; proceeding with cached values.")

    # 2. Re-simulate surface ponding
    subprocess.run([sys.executable, "src/hydrologic/simulate_runoff.py"], check=True)

    print("\n[CYCLE COMPLETE] Frontend data refreshed.")

def main():
    print("Starting Dadar Flood Nowcasting background daemon...")
    while True:
        cycle()
        print(f"\nSleeping for {POLL_INTERVAL_SECONDS // 60} minutes until next cycle... (Press Ctrl+C to stop)")
        try:
            time.sleep(POLL_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            print("\nDaemon stopped by user.")
            break

if __name__ == "__main__":
    main()