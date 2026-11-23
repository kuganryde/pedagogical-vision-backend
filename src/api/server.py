from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config import settings
from src.api.v1.router import api_router
from src.api.v1.stream_ws import router as ws_router

def create_app() -> FastAPI:
    """Factory function for FastAPI application."""
    app = FastAPI(
        title="Pedagogical Vision & Behavioral Analytics API",
        description="""
Modular AI Backend Service implementing the 5-Layer Pedagogical Model Stack:
1. **Student Detection & Tracking**: YOLOv8 / YOLO11 + ByteTrack (persistent IDs across occlusion)
2. **Body Posture & Gestures**: YOLOv8-Pose / MediaPipe Holistic (hand-raising, leaning forward, slumping, turning away)
3. **Head Pose & Gaze Direction**: 6DRepNet / L2CS-Net (pitch, yaw, roll for board vs notes vs smartphone distraction)
4. **Facial Affect & Micro-Reactions**: HSEmotion / POSTER++ / OpenFace 2.0 (confusion, boredom, delight, frustration, Action Units)
5. **Temporal Sentiment Fusion**: ST-GCN / Temporal Transformer / XGBoost (5–15s sliding window engagement & sentiment scoring)
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS configuration for easy frontend attachment
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health & status check
    @app.get("/health", tags=["System"])
    async def health_check():
        return {
            "status": "healthy",
            "service": "pedagogical-vision-backend",
            "layers": [
                "1. Student Detection & Tracking (YOLOv8/11 + ByteTrack)",
                "2. Body Posture & Gestures (YOLOv8-Pose / MediaPipe)",
                "3. Head Pose & Gaze Direction (6DRepNet / L2CS-Net)",
                "4. Facial Affect & Micro-Reactions (HSEmotion / POSTER++)",
                "5. Temporal Sentiment Fusion (ST-GCN / Temporal Transformer)"
            ]
        }

    # Mount REST and WebSocket routers
    app.include_router(api_router, prefix=settings.server.api_prefix)
    app.include_router(ws_router)

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.server:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.debug
    )
