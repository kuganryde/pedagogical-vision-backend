import time
import pytest
from src.config import settings
from src.schemas.pedagogy import (
    BoundingBox,
    StudentDetection,
    BodyPostureResult,
    HeadPoseResult,
    FacialAffectResult,
    PostureType,
    GazeTarget,
    AffectState
)
from src.layers.layer5_temporal_fusion import TemporalFusionEngine

def test_layer5_temporal_sliding_window():
    engine = TemporalFusionEngine(settings.layers.temporal_fusion)
    engine.update_config(window_seconds=10.0)
    
    t0 = time.time()
    det = StudentDetection(track_id=101, bbox=BoundingBox(x1=10, y1=10, x2=50, y2=80, width=40, height=70), confidence=0.9)
    posture = BodyPostureResult(posture=PostureType.UPRIGHT_ATTENTIVE, hand_raised=False, leaning_forward=False)
    head = HeadPoseResult(pitch=0.0, yaw=0.0, roll=0.0, gaze_target=GazeTarget.BOARD_LECTURER)
    affect = FacialAffectResult(dominant_affect=AffectState.CONCENTRATION)

    # Process consecutive observations across 5 seconds
    analysis = None
    for sec in range(5):
        analysis = engine.process(
            current_time=t0 + sec,
            detection=det,
            posture=posture,
            head_pose=head,
            facial_affect=affect
        )

    assert analysis is not None
    assert 0.0 <= analysis.engagement_score <= 1.0
    assert analysis.sliding_window_seconds == 10.0
    assert analysis.track_id == 101

def test_layer5_hand_raise_behavioral_cue():
    engine = TemporalFusionEngine(settings.layers.temporal_fusion)
    t0 = time.time()
    det = StudentDetection(track_id=202, bbox=BoundingBox(x1=10, y1=10, x2=50, y2=80, width=40, height=70), confidence=0.9)
    posture = BodyPostureResult(posture=PostureType.UPRIGHT_ATTENTIVE, hand_raised=True)
    head = HeadPoseResult(pitch=0.0, yaw=0.0, roll=0.0, gaze_target=GazeTarget.BOARD_LECTURER)
    affect = FacialAffectResult(dominant_affect=AffectState.CONCENTRATION)

    analysis = engine.process(t0, det, posture, head, affect)
    assert analysis.is_actively_participating is True
    assert "HAND_RAISED_FOR_QUESTION" in analysis.behavioral_cues

def test_layer5_phone_distraction_cue():
    engine = TemporalFusionEngine(settings.layers.temporal_fusion)
    t0 = time.time()
    det = StudentDetection(track_id=303, bbox=BoundingBox(x1=10, y1=10, x2=50, y2=80, width=40, height=70), confidence=0.9)
    posture = BodyPostureResult(posture=PostureType.UPRIGHT_ATTENTIVE, hand_raised=False)
    head = HeadPoseResult(pitch=-55.0, yaw=0.0, roll=0.0, gaze_target=GazeTarget.PHONE_LAP_DISTRACTED)
    affect = FacialAffectResult(dominant_affect=AffectState.BOREDOM)

    for i in range(4):
        analysis = engine.process(t0 + i, det, posture, head, affect)

    assert analysis.is_at_risk_disengaged is True
    assert "PHONE_LAP_DISTRACTION" in analysis.behavioral_cues
