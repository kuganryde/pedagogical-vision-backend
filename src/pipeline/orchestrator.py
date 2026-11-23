import time
from typing import List, Dict, Any, Optional

from src.config import LayersConfig
from src.schemas.pedagogy import ClassroomFrameAnalysis, TemporalStudentAnalysis
from src.layers.layer1_student_detector import StudentDetectorAndTracker
from src.layers.layer2_body_pose import BodyPoseEstimator
from src.layers.layer3_head_pose import HeadPoseGazeEstimator
from src.layers.layer4_facial_affect import FacialAffectClassifier
from src.layers.layer5_temporal_fusion import TemporalFusionEngine

class ModularPedagogicalPipeline:
    """
    Unified 5-Layer Pipeline Orchestrator:
    1. Student Detection & Tracking (YOLOv8/YOLO11 + ByteTrack)
    2. Body Posture & Gestures (YOLOv8-Pose / MediaPipe Holistic)
    3. Head Pose & Gaze Direction (6DRepNet / L2CS-Net)
    4. Facial Affect & Micro-Reactions (HSEmotion / POSTER++ / OpenFace 2.0)
    5. Temporal Sentiment Fusion (ST-GCN / Temporal Transformer / XGBoost)
    """

    def __init__(self, config: LayersConfig):
        self.config = config
        self.layer1_detector = StudentDetectorAndTracker(config.detection_and_tracking)
        self.layer2_pose = BodyPoseEstimator(config.body_posture)
        self.layer3_head_pose = HeadPoseGazeEstimator(config.head_pose)
        self.layer4_affect = FacialAffectClassifier(config.facial_affect)
        self.layer5_temporal = TemporalFusionEngine(config.temporal_fusion)

    def process_frame(
        self,
        frame_input: Any = None,
        raw_detections: Optional[List[Dict[str, Any]]] = None,
        frame_id: int = 1,
        timestamp: Optional[float] = None,
        fps: float = 5.0
    ) -> ClassroomFrameAnalysis:
        """
        Runs the 5-layer pipeline on a single video frame.
        """
        current_time = timestamp if timestamp is not None else time.time()

        # 1. Student Detection & Tracking
        students = self.layer1_detector.process(
            frame_input=frame_input,
            raw_detections=raw_detections,
            frame_id=frame_id,
            fps=fps
        )

        analyzed_students: List[TemporalStudentAnalysis] = []
        hand_raises = 0
        confused_count = 0
        distracted_count = 0

        for student in students:
            # 2. Body Posture & Gestures
            posture = self.layer2_pose.process(student, frame_input=frame_input)
            if posture.hand_raised:
                hand_raises += 1

            # 3. Head Pose & Gaze Direction
            head_pose = self.layer3_head_pose.process(student, frame_input=frame_input)

            # 4. Facial Affect & Micro-Reactions
            affect = self.layer4_affect.process(student, frame_input=frame_input)
            if affect.dominant_affect.value == "confusion":
                confused_count += 1

            # 5. Temporal Sentiment Fusion (5-15s sliding window)
            temporal_analysis = self.layer5_temporal.process(
                current_time=current_time,
                detection=student,
                posture=posture,
                head_pose=head_pose,
                facial_affect=affect
            )
            if temporal_analysis.is_at_risk_disengaged:
                distracted_count += 1

            analyzed_students.append(temporal_analysis)

        total_students = len(analyzed_students)
        mean_engagement = (
            round(sum(s.engagement_score for s in analyzed_students) / max(1, total_students), 2)
            if total_students > 0 else 0.0
        )
        confusion_pct = round((confused_count / max(1, total_students)) * 100, 1) if total_students > 0 else 0.0
        distraction_pct = round((distracted_count / max(1, total_students)) * 100, 1) if total_students > 0 else 0.0

        alerts: List[str] = []
        if total_students > 0 and (confused_count / total_students) >= self.config.temporal_fusion.alert_thresholds.confusion_alert_rate:
            alerts.append(f"CLASSROOM_ALERT: High Confusion Spike ({confusion_pct}% of students indicate cognitive impasse)")

        if hand_raises > 0:
            alerts.append(f"PEDAGOGICAL_PROMPT: {hand_raises} student(s) actively raising hand with questions")

        if total_students > 0 and (distracted_count / total_students) > 0.40:
            alerts.append(f"CLASSROOM_ALERT: High Disengagement ({distraction_pct}% distracted/sleeping)")

        return ClassroomFrameAnalysis(
            frame_id=frame_id,
            timestamp=current_time,
            total_students_detected=total_students,
            mean_engagement_score=mean_engagement,
            hand_raises_count=hand_raises,
            confusion_percentage=confusion_pct,
            distraction_percentage=distraction_pct,
            students=analyzed_students,
            active_pedagogical_alerts=alerts
        )

    def set_sliding_window(self, seconds: float):
        """Dynamically reconfigures temporal window (5-15s)."""
        self.layer5_temporal.update_config(seconds)
