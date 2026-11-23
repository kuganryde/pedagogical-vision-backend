import pytest
from src.config import settings
from src.schemas.pedagogy import BoundingBox, StudentDetection, PostureType, GazeTarget, AffectState
from src.layers.layer1_student_detector import StudentDetectorAndTracker
from src.layers.layer2_body_pose import BodyPoseEstimator
from src.layers.layer3_head_pose import HeadPoseGazeEstimator
from src.layers.layer4_facial_affect import FacialAffectClassifier

def test_layer1_student_detection_and_tracking():
    tracker = StudentDetectorAndTracker(settings.layers.detection_and_tracking)
    
    # First frame
    frame1_res = tracker.process(frame_id=1)
    assert len(frame1_res) > 0
    initial_ids = [s.track_id for s in frame1_res]
    
    # Second frame (tracks must persist IDs)
    frame2_res = tracker.process(frame_id=2)
    assert len(frame2_res) == len(frame1_res)
    second_ids = [s.track_id for s in frame2_res]
    assert initial_ids == second_ids, "Track IDs must persist across frames"

def test_layer2_body_pose_estimation():
    estimator = BodyPoseEstimator(settings.layers.body_posture)
    student = StudentDetection(
        track_id=1,
        bbox=BoundingBox(x1=100, y1=100, x2=200, y2=300, width=100, height=200),
        confidence=0.95
    )
    
    result = estimator.process(student)
    assert result.keypoints_detected > 0
    assert result.posture in [PostureType.UPRIGHT_ATTENTIVE, PostureType.LEANING_FORWARD, PostureType.SLUMPING_SLEEPING, PostureType.TURNING_AWAY]

def test_layer3_head_pose_gaze_direction():
    estimator = HeadPoseGazeEstimator(settings.layers.head_pose)
    student = StudentDetection(
        track_id=1,
        bbox=BoundingBox(x1=100, y1=100, x2=200, y2=300, width=100, height=200),
        confidence=0.95
    )
    
    result = estimator.process(student)
    assert result.gaze_target in [
        GazeTarget.BOARD_LECTURER,
        GazeTarget.NOTEBOOK_DESK,
        GazeTarget.PHONE_LAP_DISTRACTED,
        GazeTarget.SIDEWAYS_DISTRACTED
    ]
    assert -90.0 <= result.pitch <= 90.0
    assert -180.0 <= result.yaw <= 180.0

def test_layer4_facial_affect_and_action_units():
    classifier = FacialAffectClassifier(settings.layers.facial_affect)
    student = StudentDetection(
        track_id=1,
        bbox=BoundingBox(x1=100, y1=100, x2=200, y2=300, width=100, height=200),
        confidence=0.95
    )
    
    result = classifier.process(student)
    assert result.dominant_affect in [
        AffectState.CONCENTRATION,
        AffectState.CONFUSION,
        AffectState.BOREDOM,
        AffectState.DELIGHT,
        AffectState.FRUSTRATION,
        AffectState.NEUTRAL
    ]
    assert "AU04" in result.action_units
    assert "AU12" in result.action_units
    assert "AU43" in result.action_units
    assert 0.0 <= result.affect_confidence <= 1.0
