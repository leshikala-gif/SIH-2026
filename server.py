import os
import sys
import glob
import math
import re
import time
import requests
from datetime import datetime, timezone
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point
import rasterio
from rasterio.sample import sample_gen
from pyproj import Transformer

sys.path.append(".")
try:
    from src.pipeline.precipitation_fusion import PrecipitationFusionEngine
    fusion_engine = PrecipitationFusionEngine()
except ImportError:
    fusion_engine = None

app = FastAPI(title="JalMarg Urban Flood Intelligence API", version="5.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ZONE_DEFAULTS = {
    "coastal": {"C": 0.88, "drainage_mm_hr": 20.0, "carryover": 0.80},
    "plain":   {"C": 0.90, "drainage_mm_hr": 14.0, "carryover": 0.86},
    "plateau": {"C": 0.80, "drainage_mm_hr": 18.0, "carryover": 0.68}
}

DRAINAGE_CAPACITY = {
    "motorway": 50.0, "trunk": 45.0, "primary": 40.0, "secondary": 30.0,
    "tertiary": 22.0, "residential": 14.0, "service": 10.0,
    "living_street": 8.0, "unclassified": 15.0
}

CONCENTRATION_FACTOR = {
    "motorway": 1.8, "trunk": 2.2, "primary": 2.8, "secondary": 3.6,
    "tertiary": 4.5, "residential": 5.2, "service": 5.8,
    "living_street": 6.0, "unclassified": 4.8
}

SCENARIO_PROFILES = {
    "cloudburst": [25.0, 55.0, 85.0, 70.0, 45.0, 25.0, 10.0, 5.0],
    "deluge":     [15.0, 30.0, 45.0, 35.0, 20.0, 10.0, 4.0, 1.0],
    "moderate":   [4.0, 8.0, 14.0, 10.0, 5.0, 2.0, 0.0, 0.0]
}

CACHE_DIR = "data/processed/geojson_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
MAX_CACHE_FILES = 80

STATE_RTO_MAP = {
    "DL": {"state": "Delhi", "city": "New Delhi", "lat": 28.6315, "lon": 77.2167},
    "MH": {"state": "Maharashtra", "city": "Mumbai", "lat": 19.0182, "lon": 72.8434},
    "MP": {"state": "Madhya Pradesh", "city": "Bhopal", "lat": 23.2064, "lon": 77.4601},
    "KA": {"state": "Karnataka", "city": "Bengaluru", "lat": 12.9716, "lon": 77.5946},
    "TN": {"state": "Tamil Nadu", "city": "Chennai", "lat": 13.0827, "lon": 80.2707},
    "RJ": {"state": "Rajasthan", "city": "Jaipur", "lat": 26.9124, "lon": 75.7873},
    "TS": {"state": "Telangana", "city": "Hyderabad", "lat": 17.3850, "lon": 78.4867},
    "WB": {"state": "West Bengal", "city": "Kolkata", "lat": 22.5726, "lon": 88.3639}
}

MOCK_VAHAN_DATABASE: Dict[str, dict] = {
    "MH01AB1234": {
        "owner": "Rajesh Sharma",
        "phone": "+91 98201 44821",
        "masked_phone": "+91 98*** **821",
        "model": "Maruti Suzuki Swift",
        "vehicle_type": "car",
        "wading_depth_cm": 20.0,
        "wading_depth_mm": 200.0,
        "city": "Mumbai",
        "state": "Maharashtra"
    },
    "MH02CD5678": {
        "owner": "Pooja Deshmukh",
        "phone": "+91 98332 11904",
        "masked_phone": "+91 98*** **904",
        "model": "Tata Nexon",
        "vehicle_type": "suv",
        "wading_depth_cm": 30.0,
        "wading_depth_mm": 300.0,
        "city": "Mumbai",
        "state": "Maharashtra"
    },
    "MP04TA5510": {
        "owner": "Vikram Patel",
        "phone": "+91 94250 88219",
        "masked_phone": "+91 94*** **219",
        "model": "Mahindra Scorpio",
        "vehicle_type": "suv",
        "wading_depth_cm": 35.0,
        "wading_depth_mm": 350.0,
        "city": "Bhopal",
        "state": "Madhya Pradesh"
    },
    "DL01AM0911": {
        "owner": "Lifeline Emergency Services",
        "phone": "+91 99112 00102",
        "masked_phone": "+91 99*** **102",
        "model": "Force Traveller (Ambulance)",
        "vehicle_type": "ambulance",
        "wading_depth_cm": 45.0,
        "wading_depth_mm": 450.0,
        "city": "Delhi",
        "state": "Delhi"
    }
}

USERS_DATABASE: Dict[str, dict] = {}
LAST_ALERT_TIMESTAMPS: Dict[str, float] = {}
ALERT_COOLDOWN_SECONDS = 180
ACTIVE_RESCUE_REQUESTS: List[dict] = []

class SignUpRequest(BaseModel):
    vehicle_number: str
    phone: str
    full_name: Optional[str] = "Citizen Driver"
    channels: List[str] = ["sms", "whatsapp"]

class SignInRequest(BaseModel):
    identifier: str
    otp: str

class SimulateAlertRequest(BaseModel):
    vehicle_number: str
    current_water_depth_cm: float
    corridor_name: str

class LiveTelemetryPing(BaseModel):
    vehicle_number: str
    lat: float
    lon: float
    speed_kmh: Optional[float] = 0.0
    heading: Optional[float] = 0.0

class EmergencySOSRequest(BaseModel):
    vehicle_number: str
    driver_name: str
    phone: str
    lat: float
    lon: float
    vehicle_model: str
    wading_depth_mm: float

class TacticalRescueReportResponse(BaseModel):
    coordinates: List[float]
    elevation_m: float
    slope_pct: float
    current_ponding_cm: float
    peak_forecast_cm: float
    inundation_trend: str
    deployable_asset: str
    recommended_ingress_road: str
    safest_approach_depth_cm: float
    estimated_sump_volume_m3: float
    deoc_dispatch_priority: str

def sanitize_plate(plate: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", plate).upper()

def get_or_derive_vahan_profile(plate: str) -> dict:
    clean_plate = sanitize_plate(plate)
    if clean_plate in MOCK_VAHAN_DATABASE:
        data = MOCK_VAHAN_DATABASE[clean_plate].copy()
        data["plate"] = clean_plate
        return data

    is_suv = any(k in clean_plate for k in ["04", "SUV", "THAR", "SCORP"])
    is_amb = any(k in clean_plate for k in ["AM", "0911", "EMERG"])
    depth_mm = 450.0 if is_amb else (350.0 if is_suv else 200.0)

    prefix = clean_plate[:2]
    loc_meta = STATE_RTO_MAP.get(prefix, {"state": "India", "city": "Current Region"})

    return {
        "plate": clean_plate,
        "owner": "Registered Driver",
        "phone": "+91 98765 43210",
        "masked_phone": "+91 98*** **210",
        "model": "Force Traveller (Ambulance)" if is_amb else ("Mahindra Scorpio" if is_suv else "Standard Hatchback"),
        "vehicle_type": "ambulance" if is_amb else ("suv" if is_suv else "car"),
        "wading_depth_cm": depth_mm / 10.0,
        "wading_depth_mm": depth_mm,
        "city": loc_meta["city"],
        "state": loc_meta["state"]
    }

def classify_zone(lat: float, lon: float):
    if (lat < 21.0 and lon < 74.0) or lon > 85.0:
        return ZONE_DEFAULTS["coastal"]
    elif lat > 24.0 and lon < 85.0:
        return ZONE_DEFAULTS["plain"]
    return ZONE_DEFAULTS["plateau"]

def sample_smoothed_slope(geom, src, transformer=None, n_points=5):
    if geom is None or geom.is_empty:
        return 0.6
    fractions = np.linspace(0.1, 0.9, n_points)
    sample_pts = [geom.interpolate(f, normalized=True) for f in fractions]
    coords = []
    for p in sample_pts:
        x, y = p.x, p.y
        if transformer:
            x, y = transformer.transform(x, y)
        coords.append((x, y))
    
    bounds = src.bounds
    sampled_vals = []
    for (x, y) in coords:
        if bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top:
            try:
                val = list(sample_gen(src, [(x, y)]))[0][0]
                if val is not None and not np.isnan(val):
                    sampled_vals.append(float(val))
            except Exception:
                continue
    if not sampled_vals:
        return 0.6
    return round(float(np.mean(sampled_vals)), 2)

def attach_slope_from_dem(gdf_edges, dem_path="data/processed/slope_pct.tif", n_points=5):
    if not os.path.exists(dem_path) or gdf_edges is None or len(gdf_edges) == 0:
        gdf_edges["slope_pct"] = 0.6
        return gdf_edges

    try:
        with rasterio.open(dem_path) as src:
            first_geom = gdf_edges.geometry.iloc[0]
            pt = first_geom.interpolate(0.5, normalized=True)
            if not (src.bounds.left <= pt.x <= src.bounds.right and src.bounds.bottom <= pt.y <= src.bounds.top):
                gdf_edges["slope_pct"] = 0.6
                return gdf_edges

            sampling_gdf = gdf_edges
            transformer = None
            if sampling_gdf.crs is not None and src.crs is not None and sampling_gdf.crs != src.crs:
                transformer = Transformer.from_crs(sampling_gdf.crs, src.crs, always_xy=True)

            slopes = [
                sample_smoothed_slope(geom, src, transformer, n_points=n_points)
                for geom in sampling_gdf.geometry
            ]
            gdf_edges["slope_pct"] = slopes
    except Exception:
        gdf_edges["slope_pct"] = 0.6
    return gdf_edges

def generate_natural_corridor_fallback(lat: float, lon: float, dist_m: float = 1200):
    delta_deg = max(min(dist_m, 1400), 800) / 111320.0
    features = []
    steps = 6
    offsets = np.linspace(-delta_deg, delta_deg, steps)
    
    for i, dy in enumerate(offsets):
        y = lat + dy
        features.append({
            "geometry": LineString([(lon - delta_deg, y), (lon + delta_deg, y)]),
            "name": f"Avenue Link {i + 1}",
            "highway": "primary" if i in [1, 4] else "secondary",
            "slope_pct": 0.6
        })
        
    for j, dx in enumerate(offsets):
        x = lon + dx
        features.append({
            "geometry": LineString([(x, lat - delta_deg), (x, lat + delta_deg)]),
            "name": f"Sector Cross {j + 1}",
            "highway": "primary" if j in [1, 4] else "residential",
            "slope_pct": 0.5
        })

    return gpd.GeoDataFrame(features, crs="EPSG:4326")

def get_or_fetch_edges_geojson(lat: float, lon: float, dist_m: float = 1000):
    cache_key = f"roads_{round(lat, 3)}_{round(lon, 3)}.geojson"
    cache_file = os.path.join(CACHE_DIR, cache_key)

    if os.path.exists(cache_file):
        try:
            gdf = gpd.read_file(cache_file)
            if len(gdf) > 0:
                os.utime(cache_file, None)
                return gdf, "cached_disk"
        except Exception:
            pass

    delta_deg = min(dist_m, 900) / 111320.0
    south, north = round(lat - delta_deg, 5), round(lat + delta_deg, 5)
    west, east = round(lon - delta_deg, 5), round(lon + delta_deg, 5)

    overpass_query = f"""[out:json][timeout:25];
(
  way["highway"~"motorway|trunk|primary|secondary|tertiary|residential"]({south},{west},{north},{east});
);
out geom qt;
"""

    headers = {
        "User-Agent": "JalMarg-CivicHydrology-App/5.1 (contact@jalmarg.org)",
        "Accept": "application/json"
    }
    
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter"
    ]

    elements = None
    for ep in endpoints:
        try:
            resp = requests.post(ep, data={"data": overpass_query}, headers=headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                el = data.get("elements", [])
                if el:
                    elements = el
                    break
        except Exception:
            continue

    if elements:
        features = []
        for el in elements:
            if el.get("type") == "way" and "geometry" in el:
                pts = [(pt["lon"], pt["lat"]) for pt in el["geometry"]]
                if len(pts) >= 2:
                    features.append({
                        "geometry": LineString(pts),
                        "name": el.get("tags", {}).get("name", "Arterial Link"),
                        "highway": el.get("tags", {}).get("highway", "residential"),
                        "slope_pct": 0.6
                    })
        if len(features) >= 5:
            gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
            try:
                gdf.to_file(cache_file, driver="GeoJSON")
            except Exception:
                pass
            return gdf, "live_overpass"

    return generate_natural_corridor_fallback(lat, lon, dist_m=dist_m), "telemetry_fallback"

def resolve_rain_series(lat: float, lon: float, scenario: str):
    if scenario in SCENARIO_PROFILES:
        return [float(p) for p in SCENARIO_PROFILES[scenario]]
    
    if scenario == "mosdac" and fusion_engine is not None:
        try:
            now = datetime.now(timezone.utc)
            dwr_data = {"rate_mm_hr": 45.0, "timestamp": now}
            hem_data = {"rate_mm_hr": 38.0, "timestamp": now}
            gpm_data = {"rate_mm_hr": 32.0, "timestamp": now}
            fused_res = fusion_engine.fuse(lat, lon, dwr_data, hem_data, gpm_data)
            return fusion_engine.generate_step_series(fused_res["fused_rain_rate_mm_hr"])
        except Exception:
            pass

    try:
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&hourly=precipitation&minutely_15=precipitation"
            f"&forecast_days=2&timezone=auto"
        )
        r = requests.get(weather_url, timeout=3.0).json()
        m15 = r.get("minutely_15", {})
        precip = m15.get("precipitation", [])
        times = m15.get("time", [])
        
        now_iso = datetime.now().strftime("%Y-%m-%dT%H")
        start_idx = 0
        for idx, t in enumerate(times):
            if t.startswith(now_iso):
                start_idx = idx
                break
        
        slice_15 = [float(p) for p in precip[start_idx : start_idx + 8]] if precip else []
        if sum(slice_15) == 0:
            hourly = r.get("hourly", {}).get("precipitation", [])
            h_times = r.get("hourly", {}).get("time", [])
            h_start = 0
            for idx, t in enumerate(h_times):
                if t.startswith(now_iso):
                    h_start = idx
                    break
            h1 = float(hourly[h_start]) if len(hourly) > h_start else 0.0
            h2 = float(hourly[h_start + 1]) if len(hourly) > h_start + 1 else 0.0
            rain_series = [round(h1 / 4.0, 2)] * 4 + [round(h2 / 4.0, 2)] * 4
        else:
            rain_series = slice_15
    except Exception:
        rain_series = [4.0, 8.0, 14.0, 10.0, 5.0, 2.0, 0.0, 0.0]

    while len(rain_series) < 8:
        rain_series.append(0.0)
    return rain_series

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.post("/api/auth/signup")
async def auth_signup(payload: SignUpRequest):
    clean_plate = sanitize_plate(payload.vehicle_number)
    vahan_info = get_or_derive_vahan_profile(clean_plate)
    depth_mm = vahan_info.get("wading_depth_mm", 200.0)
    
    user_record = {
        "user_id": f"usr_{len(USERS_DATABASE) + 101}",
        "full_name": payload.full_name or vahan_info["owner"],
        "phone": payload.phone or vahan_info["phone"],
        "masked_phone": (payload.phone[:5] + "*** **" + payload.phone[-3:]) if len(payload.phone) >= 10 else vahan_info["masked_phone"],
        "vehicle_number": clean_plate,
        "model": vahan_info["model"],
        "vehicle_type": vahan_info["vehicle_type"],
        "wading_depth_cm": depth_mm / 10.0,
        "wading_depth_mm": depth_mm,
        "city": vahan_info["city"],
        "channels": payload.channels,
        "alerts_inbox": []
    }
    
    USERS_DATABASE[clean_plate] = user_record
    return {"status": "success", "message": "Vehicle registered successfully", "user": user_record}

@app.post("/api/auth/signin")
async def auth_signin(payload: SignInRequest):
    clean_id = sanitize_plate(payload.identifier)
    
    if payload.otp != "7429":
        return {"status": "error", "message": "Invalid verification code. Use demo code: 7429"}

    user = None
    if clean_id in USERS_DATABASE:
        user = USERS_DATABASE[clean_id]
    else:
        for u in USERS_DATABASE.values():
            if clean_id in sanitize_plate(u["phone"]):
                user = u
                break
                
    if not user:
        vahan_info = get_or_derive_vahan_profile(clean_id)
        depth_mm = vahan_info.get("wading_depth_mm", 200.0)
        user = {
            "user_id": f"usr_{len(USERS_DATABASE) + 101}",
            "full_name": vahan_info["owner"],
            "phone": vahan_info["phone"],
            "masked_phone": vahan_info["masked_phone"],
            "vehicle_number": clean_id,
            "model": vahan_info["model"],
            "vehicle_type": vahan_info["vehicle_type"],
            "wading_depth_cm": depth_mm / 10.0,
            "wading_depth_mm": depth_mm,
            "city": vahan_info["city"],
            "channels": ["sms", "whatsapp"],
            "alerts_inbox": []
        }
        USERS_DATABASE[clean_id] = user

    return {"status": "success", "message": "Signed in successfully", "user": user}

@app.post("/api/user/simulate-alert")
async def user_simulate_alert(payload: SimulateAlertRequest):
    clean_plate = sanitize_plate(payload.vehicle_number)
    user = USERS_DATABASE.get(clean_plate, get_or_derive_vahan_profile(clean_plate))
    
    wading_limit_mm = user.get("wading_depth_mm", 200.0)
    water_depth_mm = round(payload.current_water_depth_cm * 10.0)
    is_hazard = water_depth_mm >= wading_limit_mm

    alert_item = {
        "id": f"alt_{datetime.now().strftime('%H%M%S')}",
        "time": "Just now",
        "corridor": payload.corridor_name,
        "water_depth_mm": water_depth_mm,
        "severity": "CRITICAL" if is_hazard else "PASSABLE",
        "message": (
            f"⚠️ Peak ponding at {payload.corridor_name} reached {water_depth_mm:.0f} mm (Limit: {wading_limit_mm:.0f} mm). Divert immediately!"
            if is_hazard else
            f"ℹ️ {payload.corridor_name} has {water_depth_mm:.0f} mm standing water. Passable for your {user.get('model', 'vehicle')} (Limit: {wading_limit_mm:.0f} mm)."
        )
    }

    if "alerts_inbox" in user:
        user["alerts_inbox"].insert(0, alert_item)

    return {
        "status": "success",
        "alert": alert_item,
        "dispatched_to": user.get("phone", "+91 98201 44821"),
        "channels": user.get("channels", ["sms", "whatsapp"])
    }

@app.post("/api/user/live-telemetry")
async def ingest_live_telemetry(payload: LiveTelemetryPing):
    clean_plate = sanitize_plate(payload.vehicle_number)
    user = USERS_DATABASE.get(clean_plate, get_or_derive_vahan_profile(clean_plate))
    
    user_wading_limit_mm = user.get("wading_depth_mm", 200.0)
    user_location = Point(payload.lon, payload.lat)

    gdf, _ = get_or_fetch_edges_geojson(payload.lat, payload.lon, dist_m=800)
    rain_series = resolve_rain_series(payload.lat, payload.lon, scenario="mosdac")
    
    zone = classify_zone(payload.lat, payload.lon)
    C = zone["C"]
    drainage_step = zone["drainage_mm_hr"] * (15.0 / 60.0)

    hazard_found = False
    critical_street = ""
    critical_depth_mm = 0.0

    if gdf is not None and not gdf.empty:
        search_geom = user_location.buffer(0.0045)
        nearby_edges = gdf[gdf.geometry.intersects(search_geom)]

        for _, row in nearby_edges.iterrows():
            st_name = str(row.get("name", "Active Road Corridor"))
            if st_name in ["Active Road Corridor", "Urban Corridor", "Arterial Link", "Avenue Link", "Sector Cross"]:
                continue
            
            p_mm = rain_series[0] if rain_series else 15.0
            excess_mm = max(0.0, (max(0.0, p_mm - 5.0) * C) - drainage_step)
            depth_mm = excess_mm * 4.0 * 10.0

            if depth_mm >= user_wading_limit_mm:
                hazard_found = True
                critical_street = st_name
                critical_depth_mm = round(depth_mm)
                break

    now = time.time()
    last_sent = LAST_ALERT_TIMESTAMPS.get(clean_plate, 0)
    should_dispatch_external = hazard_found and (now - last_sent > ALERT_COOLDOWN_SECONDS)

    alert_payload = None
    if hazard_found:
        alert_payload = {
            "title": "FLOOD HAZARD IN VICINITY",
            "message": (
                f"⚠️ Water level at {critical_street} reached {critical_depth_mm:.0f} mm, "
                f"exceeding your {user['model']} clearance limit ({user_wading_limit_mm:.0f} mm). "
                f"Reroute immediately via JalMarg."
            ),
            "corridor": critical_street,
            "depth_mm": critical_depth_mm,
            "wading_limit_mm": user_wading_limit_mm
        }

        if should_dispatch_external:
            LAST_ALERT_TIMESTAMPS[clean_plate] = now

    return {
        "status": "synchronized",
        "hazard_detected": hazard_found,
        "alert": alert_payload
    }

@app.post("/api/user/sos")
async def handle_emergency_sos(payload: EmergencySOSRequest):
    sos_entry = {
        "id": f"sos_{int(time.time())}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "driver": payload.driver_name,
        "phone": payload.phone,
        "vehicle": f"{payload.vehicle_model} ({payload.vehicle_number})",
        "coordinates": [payload.lat, payload.lon],
        "status": "DISPATCH_PENDING"
    }
    ACTIVE_RESCUE_REQUESTS.insert(0, sos_entry)
    return {"status": "dispatched", "sos_id": sos_entry["id"], "helpline_routed": "112/1077"}

@app.get("/api/rescue/tactical-report", response_model=TacticalRescueReportResponse)
def get_tactical_rescue_report(
    lat: float = Query(..., description="Victim / Incident Latitude"),
    lon: float = Query(..., description="Victim / Incident Longitude"),
    scenario: str = Query("mosdac", description="Precipitation scenario")
):
    gdf, _ = get_or_fetch_edges_geojson(lat, lon, dist_m=800)
    gdf = attach_slope_from_dem(gdf)
    rain_series = resolve_rain_series(lat, lon, scenario)
    zone = classify_zone(lat, lon)
    C = zone["C"]
    carryover = zone["carryover"]

    victim_pt = Point(lon, lat)
    approach_roads = []
    
    if gdf is not None and not gdf.empty:
        for _, row in gdf.iterrows():
            if row.geometry is None:
                continue
            dist_to_pt = row.geometry.distance(victim_pt) * 111320.0
            if dist_to_pt <= 400:
                hw_type = str(row.get("highway", "residential"))
                drain_rate = DRAINAGE_CAPACITY.get(hw_type, zone["drainage_mm_hr"])
                drain_step = drain_rate * (15.0 / 60.0)
                gutter = CONCENTRATION_FACTOR.get(hw_type, 4.0)
                slope = float(row.get("slope_pct", 0.6))
                retention = max(0.20, min(1.35, 1.35 - (max(0.1, min(12.0, slope)) / 3.0)))

                cum_depth = 0.0
                depth_timeline = []
                for p in rain_series:
                    excess = max(0.0, (max(0.0, p - 5.0) * C) - drain_step)
                    cum_depth = (cum_depth * carryover) + (excess * retention * gutter * 0.35)
                    depth_timeline.append(round(cum_depth / 10.0, 1))

                st_name = str(row.get("name", "Arterial Access"))
                approach_roads.append({
                    "name": st_name,
                    "slope": slope,
                    "depth_now": depth_timeline[0],
                    "peak_depth": max(depth_timeline),
                    "timeline": depth_timeline
                })

    if not approach_roads:
        approach_roads = [{
            "name": "Local Arterial Link",
            "slope": 0.5,
            "depth_now": 14.0,
            "peak_depth": 28.0,
            "timeline": [14.0, 22.0, 28.0, 24.0, 16.0, 8.0, 2.0, 0.0]
        }]

    approach_roads.sort(key=lambda x: x["peak_depth"])
    best_ingress = approach_roads[0]
    worst_point = max(approach_roads, key=lambda x: x["peak_depth"])

    curr_d = worst_point["depth_now"]
    peak_d = worst_point["peak_depth"]

    trend = "STABLE"
    if len(worst_point["timeline"]) >= 3:
        if worst_point["timeline"][2] > curr_d + 3.0:
            trend = "SURGING / RISING"
        elif worst_point["timeline"][2] < curr_d - 2.0:
            trend = "RECEDING"

    if peak_d >= 50.0:
        asset = "Inflatable Rescue Boat (IRB / OBM) + NDRF Flood Team"
        priority = "P1_CRITICAL"
    elif peak_d >= 25.0:
        asset = "High-Clearance Heavy Tactical Truck (NDRF Tatra / Fire Tender)"
        priority = "P1_CRITICAL"
    elif peak_d >= 15.0:
        asset = "4x4 Emergency Response Vehicle + 10HP Submersible Pump"
        priority = "P2_ELEVATED"
    else:
        asset = "Standard Response Vehicle / Ambulance"
        priority = "P3_MONITOR"

    sump_volume = round((peak_d / 100.0) * 150.0 * 10.5)

    return {
        "coordinates": [round(lat, 5), round(lon, 5)],
        "elevation_m": round(214.0 - (lat * 0.4), 1),
        "slope_pct": best_ingress["slope"],
        "current_ponding_cm": curr_d,
        "peak_forecast_cm": peak_d,
        "inundation_trend": trend,
        "deployable_asset": asset,
        "recommended_ingress_road": best_ingress["name"],
        "safest_approach_depth_cm": best_ingress["peak_depth"],
        "estimated_sump_volume_m3": sump_volume,
        "deoc_dispatch_priority": priority
    }

@app.get("/api/nowcast")
def dynamic_nowcast(
    lat: float = Query(28.6315, description="Center Latitude"),
    lon: float = Query(77.2167, description="Center Longitude"),
    dist_m: float = Query(1000, description="Corridor radius in meters"),
    scenario: str = Query("mosdac", description="Precipitation mode")
):
    zone = classify_zone(lat, lon)
    rain_series = resolve_rain_series(lat, lon, scenario)
    intervals = [15, 30, 45, 60, 75, 90, 105, 120]

    gdf_edges = None
    data_source = "live_overpass"

    if abs(lat - 19.018) < 0.05 and abs(lon - 72.843) < 0.05 and os.path.exists("data/processed/pilot_roads.geojson"):
        try:
            gdf_edges = gpd.read_file("data/processed/pilot_roads.geojson")
            data_source = "static_dadar"
        except Exception:
            pass

    if gdf_edges is None:
        gdf_edges, source_type = get_or_fetch_edges_geojson(lat, lon, dist_m=dist_m)
        data_source = source_type

    gdf_edges = attach_slope_from_dem(gdf_edges)

    if gdf_edges.crs is not None and gdf_edges.crs != "EPSG:4326":
        try:
            gdf_edges = gdf_edges.to_crs(epsg=4326)
        except Exception:
            pass

    C = zone["C"]
    carryover = zone["carryover"]
    initial_abstraction_mm = 5.0

    features = []
    severe_streets = set()
    high_streets = set()

    for idx, row in gdf_edges.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue

        raw_name = row.get("name", "Active Road Corridor")
        st_name = str(raw_name[0]) if isinstance(raw_name, (list, np.ndarray)) else str(raw_name or "Active Road Corridor")

        raw_hw = row.get("highway", "residential")
        hw_type = str(raw_hw[0]) if isinstance(raw_hw, (list, np.ndarray)) else str(raw_hw or "residential")

        road_drain_rate = DRAINAGE_CAPACITY.get(hw_type, zone["drainage_mm_hr"])
        drainage_step = road_drain_rate * (15.0 / 60.0)
        gutter_factor = CONCENTRATION_FACTOR.get(hw_type, 4.0)

        try:
            slope = float(row.get("slope_pct", 0.6))
        except (ValueError, TypeError):
            slope = 0.6

        clamped_slope = max(0.1, min(12.0, slope))
        retention = max(0.20, min(1.35, 1.35 - (clamped_slope / 3.0)))
        cumulative_pond_mm = 0.0
        depth_timeline = []

        for p_mm in rain_series:
            effective_rain = max(0.0, p_mm - initial_abstraction_mm)
            excess = max(0.0, (effective_rain * C) - drainage_step)
            net_water = excess * retention * gutter_factor
            cumulative_pond_mm = (cumulative_pond_mm * carryover) + (net_water * 0.35)
            depth_timeline.append(round(cumulative_pond_mm / 10.0, 1))

        max_d = max(depth_timeline) if depth_timeline else 0.0
        if max_d >= 25.0 and st_name not in ["Active Road Corridor", "Urban Corridor", "Arterial Link", "Avenue Link", "Sector Cross"]:
            severe_streets.add(st_name)
        elif max_d >= 15.0 and st_name not in ["Active Road Corridor", "Urban Corridor", "Arterial Link", "Avenue Link", "Sector Cross"]:
            high_streets.add(st_name)

        features.append({
            "type": "Feature",
            "geometry": row.geometry.__geo_interface__,
            "properties": {
                "name": st_name,
                "highway": hw_type,
                "slope_pct": round(slope, 2),
                "depth_timeline_cm": depth_timeline
            }
        })

    max_forecast_depth = max([max(f["properties"]["depth_timeline_cm"]) for f in features]) if features else 0.0
    alerts = []

    if max_forecast_depth >= 25.0:
        alerts.append({
            "severity": "critical",
            "title": "Severe Inundation & Corridor Failure Warning",
            "source": f"IMD / MOSDAC Nowcast ({scenario.upper()})",
            "message": f"Peak street water depth reaching {max_forecast_depth:.1f} cm. Structural underpasses compromised.",
            "corridors": list(severe_streets)[:4] or ["Central Transit Network"],
            "safety_action": "Evacuate low ground. Divert all non-emergency traffic away from low-lying arterials."
        })
    elif max_forecast_depth >= 15.0:
        alerts.append({
            "severity": "warning",
            "title": "Elevated Surface Runoff Warning",
            "source": f"Open-Meteo & MOSDAC ({scenario.upper()})",
            "message": f"Surface accumulation between 15-25 cm detected. Traffic deceleration expected across local links.",
            "corridors": list(high_streets)[:4] or ["Secondary Transit Corridors"],
            "safety_action": "High-clearance emergency vehicles only. Exercise extreme caution at junctions."
        })
    else:
        alerts.append({
            "severity": "safe",
            "title": "Normal Operational Drainage Flow",
            "source": "IMD Real-time Telemetry",
            "message": f"Predicted ponding levels (<{max_forecast_depth:.1f} cm) remain within baseline municipal stormwater capacity.",
            "corridors": ["All corridors operational"],
            "safety_action": "Standard transit procedures active. Monitor nowcast timeline."
        })

    return {
        "type": "FeatureCollection",
        "data_source": data_source,
        "graph_fetch_error": None,
        "edge_count": len(gdf_edges),
        "intervals_min": intervals,
        "rain_series": rain_series,
        "alerts": alerts,
        "features": features
    }

@app.get("/api/route")
def calculate_safe_route(
    start_lat: float = Query(..., description="Origin Latitude"),
    start_lon: float = Query(..., description="Origin Longitude"),
    end_lat: float = Query(..., description="Destination Latitude"),
    end_lon: float = Query(..., description="Destination Longitude"),
    forecast_step: int = Query(2, ge=0, le=7, description="Step index (0=15m, 2=45m)"),
    scenario: str = Query("mosdac", description="Precipitation mode"),
    profile: str = Query("car", description="Vehicle profile: car, ambulance, fire_tender, ndrf_truck")
):
    mid_lat = (start_lat + end_lat) / 2.0
    mid_lon = (start_lon + end_lon) / 2.0

    gdf, _ = get_or_fetch_edges_geojson(mid_lat, mid_lon, dist_m=1500)

    fallback_geojson = {
        "type": "LineString",
        "coordinates": [[start_lon, start_lat], [end_lon, end_lat]]
    }

    profile_limits = {
        "car":         {"max_depth": 30.0, "warn_depth": 15.0, "penalty_mult": 12.0},
        "ambulance":   {"max_depth": 40.0, "warn_depth": 25.0, "penalty_mult": 6.0},
        "fire_tender": {"max_depth": 45.0, "warn_depth": 30.0, "penalty_mult": 4.0},
        "ndrf_truck":  {"max_depth": 60.0, "warn_depth": 40.0, "penalty_mult": 2.0}
    }
    lim = profile_limits.get(profile, profile_limits["car"])

    zone = classify_zone(mid_lat, mid_lon)
    rain_series = resolve_rain_series(mid_lat, mid_lon, scenario)
    C = zone["C"]
    carryover = zone["carryover"]

    G = nx.Graph()
    if gdf is not None and not gdf.empty:
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.geom_type != "LineString":
                continue
            coords = list(geom.coords)
            if len(coords) < 2:
                continue

            hw_type = str(row.get("highway", "residential"))
            drain_rate = DRAINAGE_CAPACITY.get(hw_type, zone["drainage_mm_hr"])
            drain_step = drain_rate * (15.0 / 60.0)
            gutter = CONCENTRATION_FACTOR.get(hw_type, 4.0)

            slope_val = float(row.get("slope_pct", 0.6))
            clamped_slope = max(0.1, min(12.0, slope_val))
            retention = max(0.20, min(1.35, 1.35 - (clamped_slope / 3.0)))

            cum_depth = 0.0
            depth_at_step = 0.0
            for p_idx, p in enumerate(rain_series[:forecast_step + 1]):
                excess = max(0.0, (max(0.0, p - 5.0) * C) - drain_step)
                cum_depth = (cum_depth * carryover) + (excess * retention * gutter * 0.35)
                if p_idx == forecast_step:
                    depth_at_step = round(cum_depth / 10.0, 1)

            for u, v in zip(coords[:-1], coords[1:]):
                length = math.hypot(v[0] - u[0], v[1] - u[1]) * 111320.0
                if depth_at_step >= lim["max_depth"]:
                    cost = length * 50.0
                elif depth_at_step >= lim["warn_depth"]:
                    cost = length * (1.0 + lim["penalty_mult"] * ((depth_at_step / lim["warn_depth"]) ** 2))
                else:
                    cost = length
                G.add_edge(u, v, weight=cost, length=length, depth=depth_at_step)

    if len(G.nodes) == 0:
        return {
            "status": "success",
            "vehicle_profile": profile,
            "forecast_window_min": (forecast_step + 1) * 15,
            "nav_waypoints": [{"lat": round(mid_lat, 5), "lon": round(mid_lon, 5)}],
            "default_route": {
                "geometry": fallback_geojson,
                "max_water_depth_cm": 14.5,
                "status": "Submerged Corridor"
            },
            "safe_route": {
                "geometry": fallback_geojson,
                "max_water_depth_cm": 3.2,
                "status": "Safe Emergency Vector"
            }
        }

    nodes = list(G.nodes)
    orig_node = min(nodes, key=lambda n: math.hypot(n[0] - start_lon, n[1] - start_lat))
    dest_node = min(nodes, key=lambda n: math.hypot(n[0] - end_lon, n[1] - end_lat))

    default_geojson = fallback_geojson
    default_max_depth = 8.5
    try:
        def_path = nx.shortest_path(G, orig_node, dest_node, weight="length")
        default_geojson = LineString(def_path).__geo_interface__
        depths = [G.get_edge_data(u, v).get("depth", 0.0) for u, v in zip(def_path[:-1], def_path[1:])]
        if depths:
            default_max_depth = max(depths)
    except Exception:
        pass

    safe_geojson = fallback_geojson
    safe_max_depth = 2.1
    try:
        alt_path = nx.shortest_path(G, orig_node, dest_node, weight="weight")
        safe_geojson = LineString(alt_path).__geo_interface__
        depths = [G.get_edge_data(u, v).get("depth", 0.0) for u, v in zip(alt_path[:-1], alt_path[1:])]
        if depths:
            safe_max_depth = max(depths)
    except Exception:
        try:
            alt_path = nx.shortest_path(G, orig_node, dest_node, weight="length")
            safe_geojson = LineString(alt_path).__geo_interface__
            depths = [G.get_edge_data(u, v).get("depth", 0.0) for u, v in zip(alt_path[:-1], alt_path[1:])]
            if depths:
                safe_max_depth = max(depths)
        except Exception:
            mid_pt_lon = (start_lon + end_lon) / 2.0 + 0.0008
            mid_pt_lat = (start_lat + end_lat) / 2.0 + 0.0008
            safe_geojson = LineString([(start_lon, start_lat), (mid_pt_lon, mid_pt_lat), (end_lon, end_lat)]).__geo_interface__
            safe_max_depth = 4.2

    nav_waypoints = []
    if safe_geojson and "coordinates" in safe_geojson:
        coords = safe_geojson["coordinates"]
        if len(coords) > 2:
            step_count = min(5, len(coords) - 2)
            indices = np.linspace(1, len(coords) - 2, step_count, dtype=int)
            for idx in indices:
                lon_pt, lat_pt = coords[idx]
                nav_waypoints.append({"lat": round(lat_pt, 5), "lon": round(lon_pt, 5)})

    return {
        "status": "success",
        "vehicle_profile": profile,
        "forecast_window_min": (forecast_step + 1) * 15,
        "nav_waypoints": nav_waypoints,
        "default_route": {
            "geometry": default_geojson,
            "max_water_depth_cm": round(default_max_depth, 1),
            "status": "Submerged Corridor" if default_max_depth >= lim["warn_depth"] else "Passable"
        },
        "safe_route": {
            "geometry": safe_geojson,
            "max_water_depth_cm": round(safe_max_depth, 1),
            "status": "Safe Emergency Vector" if safe_max_depth < lim["warn_depth"] else "Caution Advised"
        }
    }

@app.get("/api/municipal/pumps")
def get_pump_dispatch(
    lat: float = Query(28.6315, description="Center Latitude"),
    lon: float = Query(77.2167, description="Center Longitude"),
    scenario: str = Query("mosdac", description="Precipitation mode"),
    forecast_step: int = Query(2, ge=0, le=7, description="Forecast step index")
):
    gdf, _ = get_or_fetch_edges_geojson(lat, lon, dist_m=1000)
    if gdf is None or len(gdf) == 0:
        return {"dispatch_orders": []}

    gdf = attach_slope_from_dem(gdf)
    zone = classify_zone(lat, lon)
    rain_series = resolve_rain_series(lat, lon, scenario)
    C = zone["C"]
    carryover = zone["carryover"]

    dispatch_orders = []

    for idx, row in gdf.iterrows():
        raw_name = row.get("name", "")
        st_name = str(raw_name[0]) if isinstance(raw_name, (list, np.ndarray)) else str(raw_name or "")
        
        if not st_name or st_name in ["Active Road Corridor", "Urban Corridor", "Arterial Link", "Avenue Link", "Sector Cross"]:
            continue

        hw_type = str(row.get("highway", "residential"))
        drain_rate = DRAINAGE_CAPACITY.get(hw_type, zone["drainage_mm_hr"])
        drain_step = drain_rate * (15.0 / 60.0)
        gutter = CONCENTRATION_FACTOR.get(hw_type, 4.0)

        slope_val = float(row.get("slope_pct", 0.6))
        clamped_slope = max(0.1, min(12.0, slope_val))
        retention = max(0.20, min(1.35, 1.35 - (clamped_slope / 3.0)))

        cum_depth = 0.0
        depth_at_step = 0.0

        for p_idx, p in enumerate(rain_series[:forecast_step + 1]):
            excess = max(0.0, (max(0.0, p - 5.0) * C) - drain_step)
            cum_depth = (cum_depth * carryover) + (excess * retention * gutter * 0.35)
            if p_idx == forecast_step:
                depth_at_step = round(cum_depth / 10.0, 1)

        if depth_at_step >= 12.0:
            road_length_m = row.geometry.length * 111320.0 if row.geometry else 150.0
            road_width_m = 14.0 if hw_type in ["primary", "trunk", "motorway"] else 7.0
            volume_m3 = round((depth_at_step / 100.0) * road_length_m * road_width_m)
            pumps_needed = max(1, min(8, math.ceil(volume_m3 / 150.0)))
            
            action = "Deploy standard high-flow submersible pump units"
            if depth_at_step >= 25.0:
                action = "Critical: Immediate high-capacity suction & barrier deployment"
            elif depth_at_step >= 18.0:
                action = "Deploy medium-capacity diesel dewatering pump"

            dispatch_orders.append({
                "corridor": st_name,
                "highway_class": hw_type,
                "inundation_volume_m3": volume_m3,
                "predicted_depth_cm": depth_at_step,
                "pumps_allocated": pumps_needed,
                "recommended_action": action
            })

    dispatch_orders.sort(key=lambda x: x["predicted_depth_cm"], reverse=True)
    return {"dispatch_orders": dispatch_orders[:8]}

@app.get("/")
def serve_home():
    response = FileResponse("static/index.html")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

app.mount("/static", StaticFiles(directory="static"), name="static")
if os.path.exists("data"):
    app.mount("/data", StaticFiles(directory="data"), name="data")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)