from fastapi.testclient import TestClient
from server import app, classify_zone, resolve_rain_series

client = TestClient(app)

def test_root_endpoint_availability():
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")

def test_zone_classification_logic():
    # Test coastal coordinate classification
    coastal_zone = classify_zone(19.0182, 72.8434)
    assert coastal_zone["C"] == 0.88
    
    # Test plain coordinate classification
    plain_zone = classify_zone(26.9124, 75.7873)
    assert plain_zone["C"] == 0.90

def test_precipitation_resolution_fallback():
    # Verify cloudburst scenario profile returns explicit step array
    series = resolve_rain_series(19.0182, 72.8434, "cloudburst")
    assert isinstance(series, list)
    assert len(series) == 8
    assert max(series) == 85.0

def test_nowcast_endpoint_payload():
    # Test nowcast calculations for standard urban coordinates
    response = client.get("/api/nowcast?lat=19.0182&lon=72.8434&scenario=cloudburst")
    assert response.status_code == 200
    data = response.json()
    
    assert "features" in data
    assert "intervals_min" in data
    assert "rain_series" in data
    assert len(data["features"]) > 0
    
    # Validate structure of individual street corridor feature
    sample_feature = data["features"][0]
    assert "geometry" in sample_feature
    assert "properties" in sample_feature
    assert "depth_timeline_cm" in sample_feature["properties"]
    assert len(sample_feature["properties"]["depth_timeline_cm"]) == 8

def test_safe_routing_endpoint():
    # Test safe navigation diversion logic between coordinates
    response = client.get(
        "/api/route"
        "?start_lat=19.0182&start_lon=72.8434"
        "&end_lat=19.0250&end_lon=72.8500"
        "&forecast_step=2&scenario=cloudburst"
    )
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "success"
    assert "default_route" in data
    assert "safe_route" in data
    assert "max_water_depth_cm" in data["default_route"]
    assert "max_water_depth_cm" in data["safe_route"]

def test_municipal_pumps_dispatch_module():
    # Test ICCC dewatering pump allocation scheduler
    response = client.get("/api/municipal/pumps?lat=19.0182&lon=72.8434&scenario=cloudburst")
    assert response.status_code == 200
    data = response.json()
    
    assert "dispatch_orders" in data
    assert isinstance(data["dispatch_orders"], list)
    if len(data["dispatch_orders"]) > 0:
        order = data["dispatch_orders"][0]
        assert "corridor" in order
        assert "pumps_allocated" in order
        assert "recommended_action" in order

def test_uncached_location_resiliency():
    # Test system resilience and fallback grid generation for arbitrary/remote coordinates
    response = client.get("/api/nowcast?lat=34.0837&lon=74.7973&scenario=moderate")
    assert response.status_code == 200
    data = response.json()
    assert len(data["features"]) >= 15