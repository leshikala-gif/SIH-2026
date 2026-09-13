import numpy as np

class PrecipitationFusionEngine:
    def __init__(self):
        # Base initialization for MOSDAC, GPM, and ERA5 integration layers
        pass

    def inverse_distance_weighting(self, target_lat: float, target_lon: float, grid_points: list, power: float = 2.0) -> float:
        """
        Computes Inverse Distance Weighting (IDW) interpolation for precipitation 
        when exact coordinate grid cells are missing from MOSDAC/satellite feeds.
        
        :param target_lat: Latitude of the requested target location
        :param target_lon: Longitude of the requested target location
        :param grid_points: List of dicts [{"lat": float, "lon": float, "rate_mm_hr": float}, ...]
        :param power: Distance friction parameter (default = 2.0)
        :return: Interpolated rainfall rate in mm/hr
        """
        if not grid_points:
            return 20.0  # Safe default baseline convective rainfall rate
        
        weights_sum = 0.0
        value_weights_sum = 0.0
        
        for pt in grid_points:
            lat_diff = target_lat - pt["lat"]
            lon_diff = target_lon - pt["lon"]
            # Euclidean/Haversine distance approximation for local degree grids
            distance = np.sqrt(lat_diff**2 + lon_diff**2)
            
            # Handle exact coordinate match condition
            if distance == 0.0:
                return float(pt["rate_mm_hr"])
            
            weight = 1.0 / (distance ** power)
            weights_sum += weight
            value_weights_sum += weight * float(pt["rate_mm_hr"])
            
        if weights_sum == 0.0:
            return float(grid_points[0]["rate_mm_hr"])
            
        return round(value_weights_sum / weights_sum, 2)

    def fuse(self, lat: float, lon: float, dwr_data: dict, hem_data: dict, gpm_data: dict) -> dict:
        """
        Fuses multi-source precipitation feeds with automated IDW fallback for unmapped coordinates.
        """
        available_readings = []
        
        for source_data in [dwr_data, hem_data, gpm_data]:
            if source_data and "rate_mm_hr" in source_data and source_data["rate_mm_hr"] is not None:
                available_readings.append({
                    "lat": source_data.get("lat", lat),
                    "lon": source_data.get("lon", lon),
                    "rate_mm_hr": source_data["rate_mm_hr"]
                })
                
        if not available_readings:
            # Fallback baseline model simulation profile
            fused_rate = 18.5
        else:
            # Apply IDW to smooth spatial variations across available stations/satellites
            fused_rate = self.inverse_distance_weighting(lat, lon, available_readings)
            
        return {
            "lat": lat,
            "lon": lon,
            "fused_rain_rate_mm_hr": fused_rate,
            "source_count": len(available_readings)
        }

    def generate_step_series(self, base_rate_mm_hr: float) -> list:
        """Generates a 120-minute forward precipitation timeline (8 steps x 15 mins)."""
        curve_multipliers = [0.3, 0.7, 1.0, 0.85, 0.6, 0.35, 0.15, 0.05]
        return [round(base_rate_mm_hr * m, 1) for m in curve_multipliers]