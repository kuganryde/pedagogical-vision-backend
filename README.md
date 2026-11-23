# Pedagogical Vision & Behavioral Analytics Engine

Modular computer vision microservice implementing a multi-stage model pipeline for classroom and lecture hall behavior analytics.

---

## Modular Architecture Overview

The system pipeline is structured into 5 decoupled analytical layers:

1. **Student Detection & Tracking**: Isolates individual students using bounding box detection and ByteTrack association. Tracks persistent IDs across frames with occlusion state management.
2. **Body Posture & Gestures**: Extracts 17 skeletal keypoints to detect hand-raising for queries, forward torso inclination (high focus), desk slumping, and turning away.
3. **Head Pose & Gaze Target**: Regresses Euler angles (pitch, yaw, roll) to classify gaze focus into distinct zones: lecturer/board, notebook/reading, mobile phone in lap, or lateral distraction.
4. **Facial Affect & Action Units**: Analyzes facial cues to determine classroom-specific cognitive states (concentration, confusion, boredom, delight, frustration) and Action Units (AU01, AU02, AU04, AU12, AU15, AU43).
5. **Temporal Sentiment Fusion**: Fuses movement, gaze stability, and affective distributions over a configurable 5–15 second sliding buffer into an overall engagement index (0.00 to 1.00) and triggers pedagogical alerts.

---

## Real-World Application Usecases

### 1. In-Classroom Dashboard Integration (e.g. EduSphere AI)
- Ingests overhead CCTV, IP camera feeds (via RTSP), or recorded video.
- Provides real-time engagement velocity metrics to instructors.
- Alerts the teacher when cognitive confusion exceeds configurable thresholds.

### 2. Remote & Hybrid Learning Platforms
- Ingests student webcam video via WebRTC or canvas frames over WebSocket.
- Detects attention drop-offs, phone distraction, or prolonged absence.

### 3. Post-Lecture Analytics & Instructional Auditing
- Processes recorded video sessions to generate automated pedagogical reports.
- Pinpoints the exact moments of highest engagement versus student confusion.

### 4. Event-Driven Webhooks
- Automatically dispatches structured HTTP POST alerts to learning management systems (Canvas, Moodle) or automation platforms (n8n).

---

## Integration Guide

### Python Client SDK
```python
from src.sdk.client import EduVisionClient

client = EduVisionClient(base_url="http://localhost:8088")

# Start classroom session
client.start_session(session_id="lecture_101", classroom_name="Room A", sliding_window_seconds=10.0)

# Process frame
result = client.send_session_frame(session_id="lecture_101", frame_id=1, image_bytes=frame_bytes)
print("Engagement Score:", result["mean_engagement_score"])
print("Alerts:", result["active_pedagogical_alerts"])

# Conclude session and export report
summary = client.stop_session("lecture_101")
print("Summary:", summary["pedagogical_insights"])
```

### Node.js / TypeScript
```javascript
const res = await fetch("http://localhost:8088/api/v1/session/lecture_101/frame", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    session_id: "lecture_101",
    frame_id: 1,
    image_base64: base64Data
  })
});
const data = await res.json();
console.log("Mean Engagement:", data.mean_engagement_score);
```

### Low-Latency WebSocket Stream
Connect to `ws://localhost:8088/ws/stream/:sessionId` to stream raw JPEG frames or JSON payloads and receive immediate per-frame analysis.

---

## Configuration (`config/config.yaml`)

- `server.port`: Default `8088`.
- `layers.temporal_fusion.sliding_window_seconds`: Default `10.0` (adjustable from 5.0 to 15.0).
- `layers.temporal_fusion.alert_thresholds.confusion_alert_rate`: Default `0.30` (30% threshold).
- `layers.temporal_fusion.weights`: Adjustable relative weights for gaze, affect, and posture.

---

## Running the Engine

### Local Setup
```bash
cd /root/edu-vision-backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn src.api.server:app --host 0.0.0.0 --port 8088
```
Interactive OpenAPI documentation will be accessible at `http://localhost:8088/docs`.

### Automated Testing
```bash
./run_tests.sh
```
Runs the 14 automated unit and pipeline tests, tests the WebSocket endpoint, and verifies the SDK.
