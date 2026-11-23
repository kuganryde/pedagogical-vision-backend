from fastapi import FastAPI, Request
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

    # Middleware to ensure proper routing across Vercel serverless rewrites
    @app.middleware("http")
    async def handle_serverless_routing(request: Request, call_next):
        query_str = request.scope.get("query_string", b"").decode("utf-8")
        if "__path__=" in query_str:
            import urllib.parse
            params = urllib.parse.parse_qs(query_str)
            if "__path__" in params and params["__path__"]:
                raw_path = params["__path__"][0]
                if not raw_path.startswith("/"):
                    raw_path = "/" + raw_path
                request.scope["path"] = raw_path
                # Clean up query string so child endpoints receive standard query params
                clean_params = {k: v for k, v in params.items() if k != "__path__"}
                request.scope["query_string"] = urllib.parse.urlencode(clean_params, doseq=True).encode("utf-8")
        elif request.scope.get("path", "").startswith("/api/index.py"):
            new_path = request.scope["path"][len("/api/index.py"):] or "/"
            request.scope["path"] = new_path
        elif request.scope.get("path", "") == "/api/index":
            request.scope["path"] = "/"

        return await call_next(request)

    @app.api_route("/api/debug-route", methods=["GET", "POST"])
    @app.api_route("/debug-route", methods=["GET", "POST"])
    async def debug_route(request: Request):
        return {
            "scope_path": request.scope.get("path"),
            "url_path": request.url.path,
            "headers": dict(request.headers),
            "method": request.method
        }

    # CORS configuration for easy frontend attachment
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", tags=["System"])
    async def root():
        return {
            "status": "healthy",
            "service": "pedagogical-vision-backend",
            "documentation": "/docs",
            "health_check": "/health",
            "api_prefix": settings.server.api_prefix
        }

    # Health & status check
    @app.get("/health", tags=["System"])
    @app.get("/api/health", tags=["System"])
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
