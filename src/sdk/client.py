import base64
from typing import Dict, Any, Optional, List
import httpx

class EduVisionClient:
    """
    Python SDK Client for the Pedagogical Vision & Behavioral Analytics Backend.
    Enables any backend or application to attach to the 5-layer pedagogical model stack.
    
    Usage:
        client = EduVisionClient(base_url="http://localhost:8000")
        
        # Analyze a single frame
        res = client.analyze_frame(image_bytes=b"...")
        print(res["mean_engagement_score"])
        
        # Or run within a session context (5-15s sliding window)
        client.start_session("lecture_hall_101", classroom_name="Auditorium B")
        res = client.send_session_frame("lecture_hall_101", frame_id=1, image_bytes=b"...")
        summary = client.stop_session("lecture_hall_101")
    """

    def __init__(self, base_url: str = "http://localhost:8088", timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def health(self) -> Dict[str, Any]:
        resp = self.client.get("/health")
        resp.raise_for_status()
        return resp.json()

    def analyze_frame(
        self,
        image_bytes: Optional[bytes] = None,
        frame_id: int = 1,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """Analyzes a single frame across all 5 detection layers."""
        payload: Dict[str, Any] = {"frame_id": frame_id}
        if image_bytes is not None:
            payload["image_base64"] = base64.b64encode(image_bytes).decode("utf-8")
        if timestamp is not None:
            payload["timestamp"] = timestamp

        resp = self.client.post("/api/v1/analyze/frame", json=payload)
        resp.raise_for_status()
        return resp.json()

    def start_session(
        self,
        session_id: str,
        classroom_name: str = "Auditorium",
        course_name: str = "Subject",
        sliding_window_seconds: float = 10.0,
        webhook_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Initializes a lecture session with persistent tracking."""
        payload = {
            "session_id": session_id,
            "classroom_name": classroom_name,
            "course_name": course_name,
            "sliding_window_seconds": sliding_window_seconds,
            "webhook_url": webhook_url
        }
        resp = self.client.post("/api/v1/session/start", json=payload)
        resp.raise_for_status()
        return resp.json()

    def send_session_frame(
        self,
        session_id: str,
        frame_id: int,
        image_bytes: Optional[bytes] = None,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """Sends frame into session with sliding window temporal fusion."""
        payload: Dict[str, Any] = {"frame_id": frame_id, "session_id": session_id}
        if image_bytes is not None:
            payload["image_base64"] = base64.b64encode(image_bytes).decode("utf-8")
        if timestamp is not None:
            payload["timestamp"] = timestamp

        resp = self.client.post(f"/api/v1/session/{session_id}/frame", json=payload)
        resp.raise_for_status()
        return resp.json()

    def get_session_metrics(self, session_id: str) -> Dict[str, Any]:
        """Retrieves real-time dashboard metrics."""
        resp = self.client.get(f"/api/v1/session/{session_id}/metrics")
        resp.raise_for_status()
        return resp.json()

    def stop_session(self, session_id: str) -> Dict[str, Any]:
        """Closes session and returns summary analytics report."""
        resp = self.client.post(f"/api/v1/session/{session_id}/stop")
        resp.raise_for_status()
        return resp.json()

    def update_sliding_window(self, window_seconds: float) -> Dict[str, Any]:
        """Tuning parameter: adjust temporal window (5-15s)."""
        resp = self.client.put(f"/api/v1/config/sliding-window?window_seconds={window_seconds}")
        resp.raise_for_status()
        return resp.json()

    def close(self):
        self.client.close()
