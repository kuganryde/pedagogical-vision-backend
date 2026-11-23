import pytest
from fastapi.testclient import TestClient
from src.api.server import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert len(data["layers"]) == 5

def test_config_endpoints():
    response = client.get("/api/v1/config")
    assert response.status_code == 200
    data = response.json()
    assert "configuration" in data

    # Update sliding window
    put_res = client.put("/api/v1/config/sliding-window?window_seconds=12.5")
    assert put_res.status_code == 200
    assert put_res.json()["sliding_window_seconds"] == 12.5

def test_analyze_frame_api():
    payload = {
        "frame_id": 1,
        "image_base64": None
    }
    response = client.post("/api/v1/analyze/frame", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["frame_id"] == 1
    assert data["total_students_detected"] > 0
    assert "mean_engagement_score" in data

def test_session_lifecycle_api():
    # 1. Start session
    start_payload = {
        "session_id": "api_test_session_1",
        "classroom_name": "Hall 1",
        "sliding_window_seconds": 10.0
    }
    res_start = client.post("/api/v1/session/start", json=start_payload)
    assert res_start.status_code == 200

    # 2. Ingest frame
    frame_payload = {
        "session_id": "api_test_session_1",
        "frame_id": 1
    }
    res_frame = client.post("/api/v1/session/api_test_session_1/frame", json=frame_payload)
    assert res_frame.status_code == 200

    # 3. Get metrics
    res_metrics = client.get("/api/v1/session/api_test_session_1/metrics")
    assert res_metrics.status_code == 200
    assert res_metrics.json()["total_frames_processed"] >= 1

    # 4. Stop session
    res_stop = client.post("/api/v1/session/api_test_session_1/stop")
    assert res_stop.status_code == 200
    assert res_stop.json()["session_id"] == "api_test_session_1"

def test_websocket_stream():
    with client.websocket_connect("/ws/stream/ws_test_sess") as websocket:
        websocket.send_text('{"frame_id": 1}')
        data = websocket.receive_text()
        assert "mean_engagement_score" in data
