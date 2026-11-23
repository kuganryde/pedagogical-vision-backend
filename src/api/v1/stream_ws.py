import json
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from src.session.manager import SessionManager
from src.config import settings

logger = logging.getLogger("edu_vision.websocket")
router = APIRouter(tags=["WebSocket Real-Time Stream"])

session_manager = SessionManager(settings.layers)

@router.websocket("/ws/stream/{session_id}")
async def websocket_stream_endpoint(websocket: WebSocket, session_id: str):
    """
    Real-Time WebSocket Stream Connection Option:
    Enables low-latency bidirectional communication for:
    - IP Cameras & RTSP streamers
    - Edge inference devices / Raspberry Pi / Jetson
    - Web browser WebRTC / MediaStream capture
    
    Receives: Frame payload (binary JPEG bytes or JSON string with frame data)
    Sends: Instant JSON analysis with 5-layer pedagogical metrics per student and classroom aggregate
    """
    await websocket.accept()
    session = session_manager.get_or_create(session_id)
    frame_counter = 0

    try:
        while True:
            # Can receive text (JSON) or binary (JPEG/PNG bytes)
            message = await websocket.receive()

            frame_input = None
            if "bytes" in message and message["bytes"]:
                frame_input = message["bytes"]
            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    frame_input = payload.get("image")
                except Exception:
                    frame_input = None

            frame_counter += 1
            analysis = session.process_frame(
                frame_input=frame_input,
                frame_id=frame_counter
            )

            # Transmit structured analysis back through websocket
            await websocket.send_text(analysis.model_dump_json())

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error in session {session_id}: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
