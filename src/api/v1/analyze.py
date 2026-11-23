import base64
from typing import Optional, List
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from src.schemas.pedagogy import FrameAnalyzeRequest, ClassroomFrameAnalysis
from src.pipeline.orchestrator import ModularPedagogicalPipeline
from src.config import settings

router = APIRouter(prefix="/analyze", tags=["Frame Analysis"])
pipeline_instance = ModularPedagogicalPipeline(settings.layers)

@router.post("/frame", response_model=ClassroomFrameAnalysis, summary="Analyze a Single Lecture Frame")
async def analyze_frame(request: FrameAnalyzeRequest):
    """
    Executes the complete 5-layer pedagogical model stack on a frame:
    1. Student Detection & Tracking (YOLOv8/11 + ByteTrack)
    2. Body Posture & Gestures (YOLOv8-Pose / MediaPipe)
    3. Head Pose & Gaze Direction (6DRepNet / L2CS-Net)
    4. Facial Affect & Micro-Reactions (HSEmotion / POSTER++ / OpenFace)
    5. Temporal Sentiment Fusion (ST-GCN / Temporal Transformer / XGBoost)
    """
    frame_data = None
    if request.image_base64:
        try:
            frame_data = base64.b64decode(request.image_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 encoded image string")

    result = pipeline_instance.process_frame(
        frame_input=frame_data,
        frame_id=request.frame_id,
        timestamp=request.timestamp
    )
    return result

@router.post("/upload", response_model=ClassroomFrameAnalysis, summary="Analyze Frame via Multipart File Upload")
async def upload_frame(
    file: UploadFile = File(...),
    frame_id: int = Form(1),
    timestamp: Optional[float] = Form(None)
):
    """Convenience endpoint to upload image bytes directly from IP cameras or curl."""
    contents = await file.read()
    result = pipeline_instance.process_frame(
        frame_input=contents,
        frame_id=frame_id,
        timestamp=timestamp
    )
    return result
