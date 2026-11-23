import time
from typing import Dict, Any, Optional, List
from src.config import LayersConfig
from src.pipeline.orchestrator import ModularPedagogicalPipeline
from src.schemas.pedagogy import ClassroomFrameAnalysis, SessionSummaryResponse
from src.events.dispatcher import EventDispatcher

class ClassroomSession:
    """Represents an active lecture or classroom video session."""

    def __init__(
        self,
        session_id: str,
        classroom_name: str,
        course_name: str,
        layers_config: LayersConfig,
        sliding_window_seconds: float = 10.0,
        webhook_url: Optional[str] = None
    ):
        self.session_id = session_id
        self.classroom_name = classroom_name
        self.course_name = course_name
        self.created_at = time.time()
        self.last_activity = time.time()
        self.pipeline = ModularPedagogicalPipeline(layers_config)
        self.pipeline.set_sliding_window(sliding_window_seconds)

        self.dispatcher = EventDispatcher()
        if webhook_url:
            self.dispatcher.register_webhook(webhook_url)

        # Historical metrics collection
        self.history: List[ClassroomFrameAnalysis] = []
        self.unique_track_ids = set()
        self.confusion_spikes = 0
        self.total_hand_raises_detected = 0

    def process_frame(
        self,
        frame_input: Any = None,
        raw_detections: Optional[List[Dict[str, Any]]] = None,
        frame_id: int = 1,
        timestamp: Optional[float] = None
    ) -> ClassroomFrameAnalysis:
        self.last_activity = time.time()
        result = self.pipeline.process_frame(
            frame_input=frame_input,
            raw_detections=raw_detections,
            frame_id=frame_id,
            timestamp=timestamp
        )

        self.history.append(result)
        for s in result.students:
            self.unique_track_ids.add(s.track_id)
            if s.is_actively_participating:
                self.total_hand_raises_detected += 1

        if result.confusion_percentage >= 30.0:
            self.confusion_spikes += 1

        return result

    def generate_summary(self) -> SessionSummaryResponse:
        duration = max(1.0, time.time() - self.created_at)
        total_frames = len(self.history)
        if total_frames > 0:
            avg_eng = round(sum(f.mean_engagement_score for f in self.history) / total_frames, 2)
            # Find min and max engagement frames
            highest_frame = max(self.history, key=lambda f: f.mean_engagement_score)
            lowest_frame = min(self.history, key=lambda f: f.mean_engagement_score)
            high_ts = highest_frame.timestamp
            low_ts = lowest_frame.timestamp
        else:
            avg_eng = 0.0
            high_ts = None
            low_ts = None

        insights: List[str] = []
        if avg_eng >= 0.75:
            insights.append("Outstanding classroom engagement sustained throughout lecture.")
        elif avg_eng <= 0.45:
            insights.append("Low overall engagement; consider incorporating interactive polls or break sessions.")

        if self.confusion_spikes > 3:
            insights.append(f"Recorded {self.confusion_spikes} major confusion spikes during complex topic presentation.")

        if self.total_hand_raises_detected > 0:
            insights.append(f"Active student participation observed with {self.total_hand_raises_detected} question hand-raises.")

        return SessionSummaryResponse(
            session_id=self.session_id,
            classroom_name=self.classroom_name,
            duration_seconds=round(duration, 1),
            total_frames_analyzed=total_frames,
            total_unique_students=len(self.unique_track_ids),
            average_classroom_engagement=avg_eng,
            highest_engagement_timestamp=high_ts,
            lowest_engagement_timestamp=low_ts,
            confusion_spikes_count=self.confusion_spikes,
            total_hand_raises=self.total_hand_raises_detected,
            pedagogical_insights=insights
        )

class SessionManager:
    """Thread-safe session registry."""

    def __init__(self, layers_config: LayersConfig):
        self.layers_config = layers_config
        self.sessions: Dict[str, ClassroomSession] = {}

    def get_or_create(
        self,
        session_id: str,
        classroom_name: str = "Default Hall",
        course_name: str = "General",
        sliding_window: float = 10.0,
        webhook_url: Optional[str] = None
    ) -> ClassroomSession:
        if session_id not in self.sessions:
            self.sessions[session_id] = ClassroomSession(
                session_id=session_id,
                classroom_name=classroom_name,
                course_name=course_name,
                layers_config=self.layers_config,
                sliding_window_seconds=sliding_window,
                webhook_url=webhook_url
            )
        return self.sessions[session_id]

    def get(self, session_id: str) -> Optional[ClassroomSession]:
        return self.sessions.get(session_id)

    def close_session(self, session_id: str) -> Optional[SessionSummaryResponse]:
        if session_id in self.sessions:
            session = self.sessions.pop(session_id)
            return session.generate_summary()
        return None
