import base64
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends
from src.schemas.pedagogy import (
    SessionStartRequest,
    SessionSummaryResponse,
    ClassroomFrameAnalysis,
    FrameAnalyzeRequest
)
from src.session.manager import SessionManager
from src.config import settings

router = APIRouter(prefix="/session", tags=["Classroom Sessions"])
session_manager = SessionManager(settings.layers)

@router.post("/start", summary="Start a Classroom Lecture Session")
async def start_session(request: SessionStartRequest):
    """
    Initializes a tracked lecture session.
    Maintains persistent student IDs, occlusion recovery, and temporal sliding windows.
    """
    session = session_manager.get_or_create(
        session_id=request.session_id,
        classroom_name=request.classroom_name,
        course_name=request.course_name or "General Lecture",
        sliding_window=request.sliding_window_seconds,
        webhook_url=request.webhook_url
    )
    return {
        "status": "active",
        "session_id": session.session_id,
        "classroom_name": session.classroom_name,
        "sliding_window_seconds": request.sliding_window_seconds,
        "message": "Classroom session initialized successfully."
    }

@router.post("/{session_id}/frame", response_model=ClassroomFrameAnalysis, summary="Ingest Frame into Active Session")
async def ingest_session_frame(session_id: str, request: FrameAnalyzeRequest):
    """
    Feeds a video frame into the active classroom session.
    Maintains 5-15s sliding window history across frames for each tracked student.
    """
    session = session_manager.get(session_id)
    if not session:
        # Auto-create if not explicitly started
        session = session_manager.get_or_create(session_id)

    frame_bytes = None
    if request.image_base64:
        try:
            frame_bytes = base64.b64decode(request.image_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 payload")

    analysis = session.process_frame(
        frame_input=frame_bytes,
        frame_id=request.frame_id,
        timestamp=request.timestamp
    )

    # Trigger alerts to registered webhook if pedagogical anomalies occur
    if analysis.active_pedagogical_alerts and session.dispatcher.webhook_urls:
        for alert in analysis.active_pedagogical_alerts:
            await session.dispatcher.dispatch_event("PEDAGOGICAL_ALERT", {
                "session_id": session_id,
                "frame_id": request.frame_id,
                "alert": alert,
                "mean_engagement": analysis.mean_engagement_score,
                "confusion_percentage": analysis.confusion_percentage
            })

    return analysis

@router.get("/{session_id}/metrics", summary="Get Live Classroom Engagement Metrics")
async def get_session_metrics(session_id: str):
    """Returns real-time classroom statistics from the latest frame."""
    session = session_manager.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    latest = session.history[-1] if session.history else None
    return {
        "session_id": session_id,
        "classroom_name": session.classroom_name,
        "total_frames_processed": len(session.history),
        "unique_students_tracked": len(session.unique_track_ids),
        "latest_metrics": latest
    }

@router.post("/{session_id}/stop", response_model=SessionSummaryResponse, summary="End Classroom Session & Export Report")
async def stop_session(session_id: str):
    """Terminates session and generates comprehensive pedagogical summary."""
    summary = session_manager.close_session(session_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Session not found or already closed")
    return summary
