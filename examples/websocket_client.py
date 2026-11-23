"""
Real-time WebSocket Streaming Client Example:
Connects to ws://localhost:8000/ws/stream/{session_id} and streams frames.
"""
import asyncio
import json
import websockets

async def stream_frames():
    uri = "ws://localhost:8088/ws/stream/live_hall_alpha"
    print(f"Connecting to WebSocket stream at {uri}...")

    async with websockets.connect(uri) as ws:
        print("Connected to WebSocket server!")

        for frame_id in range(1, 6):
            payload = {
                "frame_id": frame_id,
                "timestamp": asyncio.get_event_loop().time()
            }
            await ws.send(json.dumps(payload))
            response_text = await ws.recv()
            analysis = json.loads(response_text)

            print(f"Received Analysis Frame #{analysis['frame_id']}:")
            print(f"  Students Count: {analysis['total_students_detected']}")
            print(f"  Engagement: {analysis['mean_engagement_score']}")
            print(f"  Confusion: {analysis['confusion_percentage']}%")
            if analysis['students']:
                first_student = analysis['students'][0]
                print(f"  Sample Student #{first_student['track_id']}:")
                print(f"    Gaze: {first_student['head_pose']['gaze_target']}")
                print(f"    Posture: {first_student['body_posture']['posture']}")
                print(f"    Affect: {first_student['facial_affect']['dominant_affect']}")
                print(f"    Cues: {first_student['behavioral_cues']}")

            await asyncio.sleep(0.5)

if __name__ == "__main__":
    asyncio.run(stream_frames())
