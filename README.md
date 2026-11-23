# Pedagogical Vision & Behavioral Analytics Engine

> High-throughput, modular AI backend and API service implementing the recommended 5-layer pedagogical vision stack for automated lecture hall and classroom engagement analysis.

---

## 🎯 Modular Model Stack Overview

| # | Detection Layer | Recommended Models | Target Pedagogical & Behavioral Cues |
|---|---|---|---|
| **1** | **Student Detection & Tracking** | `YOLOv8` / `YOLO11` + `ByteTrack` | Isolates individual students in dense lecture halls and assigns persistent IDs across frames without losing track during occlusion. |
| **2** | **Body Posture & Gestures** | `YOLOv8-Pose` / `MediaPipe Holistic` | Tracks skeletal keypoints to identify hand-raising, leaning forward (high engagement), slumping/sleeping on desk, or turning away. |
| **3** | **Head Pose & Gaze Direction** | `6DRepNet` / `L2CS-Net` | Estimates pitch, yaw, and roll to verify whether students are tracking the lecturer/board, looking at notes, or distracted by phones. |
| **4** | **Facial Affect & Micro-Reactions** | `HSEmotion (EfficientNet-B0)`, `POSTER++`, `OpenFace 2.0` | Classifies classroom-specific affective states (`confusion`, `boredom`, `delight`, `frustration`, `concentration`) and Facial Action Units (AU01, AU02, AU04, AU12, AU15, AU43) rather than generic happy/sad emotions. |
| **5** | **Temporal Sentiment Fusion** | `ST-GCN`, `Temporal Transformer`, `XGBoost` | Aggregates movement, gaze, and facial vectors over a 5–15 second sliding window to output an accurate fused engagement and sentiment score. |

---

## 🔌 Connection Options

The service is engineered to be attached to **any** frontend, edge device, microservice, or LMS:

### 1. REST API (`/api/v1`)
- `POST /api/v1/analyze/frame`: Ingest a single frame (base64 JSON or image bytes).
- `POST /api/v1/analyze/upload`: Multipart file upload for IP cameras / cURL.
- `POST /api/v1/session/start`: Initialize a lecture session with persistent tracking.
- `POST /api/v1/session/{session_id}/frame`: Ingest frame into active session (maintains 5–15s sliding window history).
- `GET /api/v1/session/{session_id}/metrics`: Live classroom dashboard statistics.
- `POST /api/v1/session/{session_id}/stop`: Terminate session and export pedagogical summary report.
- `GET /api/v1/config` & `PUT /api/v1/config/sliding-window`: Runtime parameter tuning.

### 2. Low-Latency WebSocket Stream (`/ws/stream/{session_id}`)
- Connect via `ws://<host>:8000/ws/stream/<session_id>`
- Send JPEG frames or JSON frame payloads at 5–30 FPS.
- Receive immediate per-frame JSON analytics containing all 5 layers and real-time alerts.

### 3. Outbound Webhooks & Event Dispatcher
- Real-time webhook push when pedagogical anomalies trigger:
  - `STUDENT_HAND_RAISE`: Instant notification when a student raises their hand to ask a question.
  - `LECTURE_CONFUSION_SPIKE`: Alerts the lecturer or dashboard when >30% of the class shows cognitive impasse.
  - `PERSISTENT_DISENGAGEMENT`: Flags students looking at smartphone screens or sleeping on desks > 8s.

### 4. Python SDK (`EduVisionClient`)
Ready-to-use client for Python apps (Django, Flask, FastAPI, Streamlit, Celery):
```python
from src.sdk.client import EduVisionClient

client = EduVisionClient(base_url="http://localhost:8000")

# Start session
client.start_session("hall_101", classroom_name="Lecture Hall A")

# Send frame (sliding window maintains state per student)
result = client.send_session_frame("hall_101", frame_id=1, image_bytes=frame_bytes)
print(f"Classroom Engagement: {result['mean_engagement_score']}")
print(f"Active Alerts: {result['active_pedagogical_alerts']}")

# Stop session
summary = client.stop_session("hall_101")
print(summary["pedagogical_insights"])
```

### 5. Node.js / TypeScript Integration
```javascript
const res = await fetch('http://localhost:8000/api/v1/session/start', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    session_id: 'lecture_room_a',
    classroom_name: 'Room A',
    sliding_window_seconds: 10.0
  })
});
```

---

## ⚙️ Easy Configuration

All parameters are configured in [`config/config.yaml`](file:///root/edu-vision-backend/config/config.yaml) or via environment variables:

```yaml
layers:
  detection_and_tracking:
    model_name: "yolo11m"
    confidence_threshold: 0.45
    track_buffer: 30
  
  body_posture:
    model_name: "yolov8m-pose"
    hand_raise_wrist_nose_margin: 15.0
    lean_forward_torso_deg: 18.0
    
  head_pose:
    model_name: "6DRepNet"
    gaze_zones:
      board_lecturer:
        yaw_range: [-22.0, 22.0]
        pitch_range: [-15.0, 15.0]
      notebook_desk:
        pitch_range: [-45.0, -15.0]
      phone_lap:
        pitch_range: [-90.0, -45.0]

  facial_affect:
    model_name: "HSEmotion-B0"
    emotions: ["concentration", "confusion", "boredom", "delight", "frustration", "neutral"]

  temporal_fusion:
    sliding_window_seconds: 10.0  # 5-15s sliding window
    sample_rate_fps: 5.0
    weights:
      attention_gaze: 0.40
      facial_affect: 0.35
      body_posture: 0.25
    alert_thresholds:
      confusion_alert_rate: 0.30
      prolonged_distraction_seconds: 8.0
```

---

## 🚀 Quickstart

### 1. Local Run
```bash
cd /root/edu-vision-backend
source venv/bin/activate
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

Interactive OpenAPI Swagger UI is available at `http://localhost:8000/docs`.

### 2. Docker Deployment
```bash
docker compose up -d --build
```

### 3. Run Test Suite
```bash
pytest tests/ -v
```
