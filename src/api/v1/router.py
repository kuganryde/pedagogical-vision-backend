from typing import Dict, Any
from fastapi import APIRouter
from src.api.v1.analyze import router as analyze_router
from src.api.v1.session import router as session_router
from src.config import settings

api_router = APIRouter()
api_router.include_router(analyze_router)
api_router.include_router(session_router)

@api_router.get("/config", summary="Get Current Modular Model Configuration")
async def get_configuration():
    """Returns current active settings for the 5 modular layers."""
    return {
        "status": "success",
        "configuration": {
            "detection_and_tracking": settings.layers.detection_and_tracking.model_dump(),
            "body_posture": settings.layers.body_posture.model_dump(),
            "head_pose": settings.layers.head_pose.model_dump(),
            "facial_affect": settings.layers.facial_affect.model_dump(),
            "temporal_fusion": settings.layers.temporal_fusion.model_dump(),
        }
    }

@api_router.put("/config/sliding-window", summary="Update Temporal Sliding Window Duration (5-15s)")
async def update_sliding_window(window_seconds: float):
    """Dynamically tunes the temporal sentiment sliding window."""
    clamped = max(5.0, min(15.0, window_seconds))
    settings.layers.temporal_fusion.sliding_window_seconds = clamped
    return {
        "status": "success",
        "sliding_window_seconds": clamped,
        "message": f"Temporal fusion window updated to {clamped} seconds."
    }
