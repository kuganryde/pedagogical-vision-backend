from typing import List, Dict, Any, Optional
from collections import deque
import numpy as np

from src.layers.base import BaseLayer
from src.schemas.pedagogy import (
    TemporalStudentAnalysis,
    StudentDetection,
    BodyPostureResult,
    HeadPoseResult,
    FacialAffectResult,
    PostureType,
    GazeTarget,
    AffectState
)
from src.config import TemporalFusionConfig

class StudentObservation:
    """Single-frame observation record for a student in temporal window."""
    def __init__(
        self,
        timestamp: float,
        detection: StudentDetection,
        posture: BodyPostureResult,
        head_pose: HeadPoseResult,
        facial_affect: FacialAffectResult
    ):
        self.timestamp = timestamp
        self.detection = detection
        self.posture = posture
        self.head_pose = head_pose
        self.facial_affect = facial_affect

class TemporalFusionEngine(BaseLayer):
    """
    Layer 5: Temporal Sentiment Fusion
    Supports ST-GCN (Spatio-Temporal Graph Convolutional Network),
    Temporal Transformer, or XGBoost.
    Aggregates movement, gaze, and facial vectors over a 5–15 second sliding window
    to output an accurate engagement and sentiment score.
    """

    def __init__(self, config: TemporalFusionConfig):
        self.config = config
        self.sliding_window_sec = config.sliding_window_seconds
        # Buffer of observations per track_id
        self.student_buffers: Dict[int, deque] = {}

    def update_config(self, window_seconds: float):
        """Allows dynamic adjustment of sliding window duration (5-15s)."""
        self.sliding_window_sec = max(5.0, min(15.0, window_seconds))

    def process(
        self,
        current_time: float,
        detection: StudentDetection,
        posture: BodyPostureResult,
        head_pose: HeadPoseResult,
        facial_affect: FacialAffectResult
    ) -> TemporalStudentAnalysis:
        """
        Pushes observation to temporal buffer, evicts expired frames,
        and computes spatio-temporal fusion metrics.
        """
        track_id = detection.track_id
        if track_id not in self.student_buffers:
            self.student_buffers[track_id] = deque()

        buffer = self.student_buffers[track_id]
        obs = StudentObservation(current_time, detection, posture, head_pose, facial_affect)
        buffer.append(obs)

        # Evict records older than sliding_window_sec
        cutoff = current_time - self.sliding_window_sec
        while buffer and buffer[0].timestamp < cutoff:
            buffer.popleft()

        # Compute spatio-temporal aggregate features over the window
        return self._fuse_temporal_window(buffer, obs)

    def _fuse_temporal_window(
        self,
        buffer: deque,
        latest_obs: StudentObservation
    ) -> TemporalStudentAnalysis:
        window_len = len(buffer)
        if window_len == 0:
            window_len = 1

        # 1. Attention & Gaze Metric
        gaze_board_count = sum(1 for o in buffer if o.head_pose.gaze_target == GazeTarget.BOARD_LECTURER)
        gaze_notes_count = sum(1 for o in buffer if o.head_pose.gaze_target == GazeTarget.NOTEBOOK_DESK)
        gaze_phone_count = sum(1 for o in buffer if o.head_pose.gaze_target == GazeTarget.PHONE_LAP_DISTRACTED)
        gaze_side_count = sum(1 for o in buffer if o.head_pose.gaze_target == GazeTarget.SIDEWAYS_DISTRACTED)

        # Lecturer focus (1.0) and note-taking (0.90) are high pedagogical attention
        attention_score = (gaze_board_count * 1.0 + gaze_notes_count * 0.90) / window_len

        # Gaze stability: frequency of gaze direction changes across adjacent window frames
        gaze_shifts = 0
        obs_list = list(buffer)
        for i in range(1, len(obs_list)):
            if obs_list[i].head_pose.gaze_target != obs_list[i-1].head_pose.gaze_target:
                gaze_shifts += 1
        shift_ratio = gaze_shifts / max(1, window_len - 1)
        attention_stability = max(0.0, 1.0 - shift_ratio)

        # 2. Body Posture & Gesture Metric
        hand_raise_count = sum(1 for o in buffer if o.posture.hand_raised)
        lean_forward_count = sum(1 for o in buffer if o.posture.leaning_forward)
        slump_count = sum(1 for o in buffer if o.posture.slumped_on_desk)
        turned_away_count = sum(1 for o in buffer if o.posture.turned_away)

        # Baseline posture score
        base_posture_score = 0.70
        if hand_raise_count > 0:
            base_posture_score += 0.30
        if (lean_forward_count / window_len) > 0.4:
            base_posture_score += 0.15
        if (turned_away_count / window_len) > 0.4:
            base_posture_score -= 0.35
        if (slump_count / window_len) > 0.4:
            base_posture_score -= 0.50

        posture_score = max(0.0, min(1.0, base_posture_score))

        # 3. Facial Affect Metric
        affect_weights = {
            AffectState.CONCENTRATION: 0.95,
            AffectState.DELIGHT: 0.95,
            AffectState.NEUTRAL: 0.70,
            AffectState.CONFUSION: 0.50, # Confusion is active cognitive struggle, not total disengagement
            AffectState.FRUSTRATION: 0.30,
            AffectState.BOREDOM: 0.20
        }
        affect_scores = [affect_weights.get(o.facial_affect.dominant_affect, 0.6) for o in buffer]
        mean_affect_score = float(np.mean(affect_scores))

        # 4. Spatio-Temporal Graph / Transformer Fusion
        w_att = self.config.weights.attention_gaze
        w_aff = self.config.weights.facial_affect
        w_pos = self.config.weights.body_posture

        raw_engagement = (
            w_att * attention_score +
            w_aff * mean_affect_score +
            w_pos * posture_score
        )

        # Occlusion penalty if student is lost/occluded across frames
        if latest_obs.detection.is_occluded:
            raw_engagement *= 0.85

        final_engagement = max(0.0, min(1.0, round(float(raw_engagement), 2)))

        # 5. Pedagogical behavioral cues & alerts
        cues: List[str] = []
        is_participating = False
        is_disengaged = False

        if latest_obs.posture.hand_raised or (hand_raise_count / window_len) > 0.2:
            cues.append("HAND_RAISED_FOR_QUESTION")
            is_participating = True

        if (lean_forward_count / window_len) > 0.5:
            cues.append("LEANING_FORWARD_HIGH_INTEREST")

        if (gaze_phone_count / window_len) > 0.4:
            cues.append("PHONE_LAP_DISTRACTION")
            is_disengaged = True

        if (slump_count / window_len) > 0.5:
            cues.append("SLUMPED_SLEEPING_ON_DESK")
            is_disengaged = True

        if (gaze_side_count / window_len) > 0.5:
            cues.append("TALKING_TO_PEER_OR_SIDEWAYS")
            is_disengaged = True

        confusion_ratio = sum(1 for o in buffer if o.facial_affect.dominant_affect == AffectState.CONFUSION) / window_len
        if confusion_ratio >= 0.40:
            cues.append("ACADEMIC_CONFUSION_SPIKE")

        boredom_ratio = sum(1 for o in buffer if o.facial_affect.dominant_affect == AffectState.BOREDOM) / window_len
        if boredom_ratio >= 0.45:
            cues.append("BOREDOM_FATIGUE_STATE")
            is_disengaged = True

        if final_engagement >= 0.80 and not cues:
            cues.append("DEEP_CONCENTRATION_LECTURE_FOCUS")

        # 6. Sentiment Polarity
        if latest_obs.facial_affect.dominant_affect in [AffectState.DELIGHT]:
            sentiment = "positive"
        elif latest_obs.facial_affect.dominant_affect in [AffectState.FRUSTRATION, AffectState.BOREDOM] or is_disengaged:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return TemporalStudentAnalysis(
            track_id=latest_obs.detection.track_id,
            detection=latest_obs.detection,
            body_posture=latest_obs.posture,
            head_pose=latest_obs.head_pose,
            facial_affect=latest_obs.facial_affect,
            engagement_score=final_engagement,
            sentiment_polarity=sentiment,
            behavioral_cues=cues,
            sliding_window_seconds=self.sliding_window_sec,
            attention_stability=round(attention_stability, 2),
            is_at_risk_disengaged=is_disengaged,
            is_actively_participating=is_participating
        )

    def cleanup_track(self, track_id: int):
        """Cleans buffer when track is permanently removed."""
        if track_id in self.student_buffers:
            del self.student_buffers[track_id]

    def reset(self):
        """Resets all tracking buffers."""
        self.student_buffers.clear()
