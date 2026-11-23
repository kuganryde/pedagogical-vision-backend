"""
Example Client Demo:
Demonstrates attaching to the Pedagogical Vision Backend via Python SDK.
"""
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sdk.client import EduVisionClient

def run_demo():
    print("=== Pedagogical Vision Backend SDK Demo ===")
    client = EduVisionClient(base_url="http://localhost:8088")

    # 1. Health check
    print("\n1. Checking API Health...")
    health = client.health()
    print(f"Status: {health['status']}")
    for layer in health['layers']:
        print(f"  ✓ {layer}")

    # 2. Start a classroom lecture session
    session_id = f"lecture_math_{int(time.time())}"
    print(f"\n2. Starting Session: {session_id}...")
    start_resp = client.start_session(
        session_id=session_id,
        classroom_name="Hall 402 - Linear Algebra",
        course_name="MATH 201",
        sliding_window_seconds=10.0
    )
    print(f"Session started: {start_resp['message']}")

    # 3. Simulate streaming 5 frames into session
    print("\n3. Streaming simulated classroom video frames...")
    for frame_idx in range(1, 6):
        res = client.send_session_frame(
            session_id=session_id,
            frame_id=frame_idx,
            timestamp=time.time()
        )
        print(f"Frame #{frame_idx}: Detected {res['total_students_detected']} students | "
              f"Mean Engagement: {res['mean_engagement_score']} | "
              f"Hand-Raises: {res['hand_raises_count']} | "
              f"Confusion: {res['confusion_percentage']}%")

        if res['active_pedagogical_alerts']:
            for alert in res['active_pedagogical_alerts']:
                print(f"   ⚠️  Alert: {alert}")

        time.sleep(0.3)

    # 4. Get live session metrics
    print("\n4. Fetching Real-time Session Dashboard Metrics...")
    metrics = client.get_session_metrics(session_id)
    print(f"Total Frames: {metrics['total_frames_processed']}, "
          f"Unique Students Tracked: {metrics['unique_students_tracked']}")

    # 5. Stop session and export pedagogical summary
    print("\n5. Stopping Session and Exporting Pedagogical Summary...")
    summary = client.stop_session(session_id)
    print(f"Duration: {summary['duration_seconds']}s")
    print(f"Average Classroom Engagement: {summary['average_classroom_engagement']}")
    print(f"Confusion Spikes Count: {summary['confusion_spikes_count']}")
    print(f"Total Hand Raises: {summary['total_hand_raises']}")
    print("Pedagogical Insights:")
    for ins in summary['pedagogical_insights']:
        print(f"  • {ins}")

    print("\nDemo completed successfully!")

if __name__ == "__main__":
    run_demo()
