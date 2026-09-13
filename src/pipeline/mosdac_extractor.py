import os
import glob
import numpy as np
from datetime import datetime, timezone

try:
    import h5py
except ImportError:
    h5py = None

class MOSDACReader:
    def __init__(self, data_dir: str = "data/live_radar"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def get_latest_h5_file(self):
        """Finds the most recently downloaded MOSDAC HDF5 product in the data directory."""
        files = glob.glob(os.path.join(self.data_dir, "*.h5")) + glob.glob(os.path.join(self.data_dir, "*.HDF5"))
        if not files:
            return None
        return max(files, key=os.path.getmtime)

    def get_live_precipitation_rate(self, target_lat: float, target_lon: float) -> dict:
        """
        Extracts real-time rainfall rate (mm/hr) from the active MOSDAC Hydro-Estimator granule.
        """
        h5_path = self.get_latest_h5_file()
        if not h5_path or h5py is None:
            return None

        try:
            with h5py.File(h5_path, "r") as h5f:
                # Standard INSAT HEM dataset layout: datasets contain rain rates and lat/lon navigation arrays
                dataset_keys = list(h5f.keys())
                
                # Dynamic key resolution for varying INSAT-3D/3DR/3DS versions
                rain_key = next((k for k in ["rain_rate", "RAIN", "HEM_RAIN", "precipitation"] if k in dataset_keys), None)
                lat_key = next((k for k in ["Latitude", "LATITUDE", "lat"] if k in dataset_keys), None)
                lon_key = next((k for k in ["Longitude", "LONGITUDE", "lon"] if k in dataset_keys), None)

                if not rain_key:
                    return None

                rain_arr = h5f[rain_key][:]
                
                # Check if coordinates are 1D arrays or 2D grids
                if lat_key and lon_key:
                    lats = h5f[lat_key][:]
                    lons = h5f[lon_key][:]

                    if lats.ndim == 1 and lons.ndim == 1:
                        lat_idx = np.abs(lats - target_lat).argmin()
                        lon_idx = np.abs(lons - target_lon).argmin()
                        rate = float(rain_arr[lat_idx, lon_idx])
                    else:
                        # 2D coordinate mesh
                        dist = (lats - target_lat)**2 + (lons - target_lon)**2
                        min_idx = np.unravel_index(dist.argmin(), dist.shape)
                        rate = float(rain_arr[min_idx])
                else:
                    # Fallback index mapping for standard ISRO Indian region bounding projection
                    lat_idx = int(np.clip((target_lat - 5.0) / 35.0 * rain_arr.shape[0], 0, rain_arr.shape[0] - 1))
                    lon_idx = int(np.clip((target_lon - 65.0) / 35.0 * rain_arr.shape[1], 0, rain_arr.shape[1] - 1))
                    rate = float(rain_arr[lat_idx, lon_idx])

                if np.isnan(rate) or rate < 0:
                    rate = 0.0

                return {
                    "source": "ISRO_MOSDAC_HEM",
                    "file": os.path.basename(h5_path),
                    "lat": target_lat,
                    "lon": target_lon,
                    "rate_mm_hr": round(rate, 2),
                    "timestamp": datetime.now(timezone.utc)
                }
        except Exception as e:
            print(f"[MOSDAC Extraction Error] {e}")
            return None