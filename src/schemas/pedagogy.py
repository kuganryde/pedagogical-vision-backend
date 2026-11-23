from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field

class PostureType(str, Enum):
    UPRIGHT_ATTENTIVE = "UPRIGHT_ATTENTIVE"
    LEANING_FORWARD = "LEANING_FORWARD"
    SLUMPING_SLEEPING = "SLUMPING_SLEEPING"
    TURNING_AWAY = "TURNING_AWAY"
    UNKNOWN = "UNKNOWN"

class GazeTarget(str, Enum):
    BOARD_LECTURER = "BOARD_LECTURER"
    NOTEBOOK_DESK = "NOTEBOOK_DESK"
    PHONE_LAP_DISTRACTED = "PHONE_LAP_DISTRACTED"
    SIDEWAYS_DISTRACTED = "SIDEWAYS_DISTRACTED"
    UNKNOWN = "UNKNOWN"

class AffectState(str, Enum):
    CONCENTRATION = "concentration"
    CONFUSION = "confusion"
    BOREDOM = "boredom"
    DELIGHT = "delight"
    FRUSTRATION = "frustration"
    NEUTRAL = "neutral"

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float

class Keypoint(BaseModel):
    id: int
    name: str
    x: float
    y: float
    confidence: float

# Layer 1: Student Detection & Tracking
class StudentDetection(BaseModel):
    track_id: int
    bbox: BoundingBox
    confidence: float
    is_occluded: bool = False
    time_tracked_seconds: float = 0.0

# Layer 2: Body Posture & Gestures
class BodyPostureResult(BaseModel):
    posture: PostureType
    hand_raised: bool = False
    leaning_forward: bool = False
    slumped_on_desk: bool = False
    turned_away: bool = False
    torso_angle: float = 0.0
    posture_confidence: float = 0.9
    keypoints_detected: int = 17

# Layer 3: Head Pose & Gaze Direction
class HeadPoseResult(BaseModel):
    pitch: float = Field(..., description="Looking up (+)/down (-) in degrees")
    yaw: float = Field(..., description="Looking right (+)/left (-) in degrees")
    roll: float = Field(..., description="Head tilt in degrees")
    gaze_target: GazeTarget
    gaze_confidence: float = 0.9

# Layer 4: Facial Affect & Micro-Reactions
class FacialAffectResult(BaseModel):
    dominant_affect: AffectState
    affect_confidence: float = 0.85
    affect_distribution: Dict[str, float] = Field(
        default_factory=lambda: {
            "concentration": 0.0,
            "confusion": 0.0,
            "boredom": 0.0,
            "delight": 0.0,
            "frustration": 0.0,
            "neutral": 0.0,
        }
    )
    action_units: Dict[str, float] = Field(
        default_factory=lambda: {
            "AU01": 0.0,
            "AU02": 0.0,
            "AU04": 0.0,
            "AU12": 0.0,
            "AU15": 0.0,
            "AU43": 0.0,
        }
    )

# Layer 5: Temporal Sentiment Fusion (Aggregate Per Student)
class TemporalStudentAnalysis(BaseModel):
    track_id: int
    detection: StudentDetection
    body_posture: BodyPostureResult
    head_pose: HeadPoseResult
    facial_affect: FacialAffectResult

    # Layer 5 Fused Metrics
    engagement_score: float = Field(..., ge=0.0, le=1.0, description="Fused 0.0-1.0 engagement score")
    sentiment_polarity: str = Field(default="neutral", description="positive, neutral, negative")
    behavioral_cues: List[str] = Field(default_factory=list)
    sliding_window_seconds: float = 10.0
    attention_stability: float = 1.0
    is_at_risk_disengaged: bool = False
    is_actively_participating: bool = False

# Classroom Session Snapshot
class ClassroomFrameAnalysis(BaseModel):
    frame_id: int
    timestamp: float
    total_students_detected: int
    mean_engagement_score: float
    hand_raises_count: int
    confusion_percentage: float
    distraction_percentage: float
    students: List[TemporalStudentAnalysis]
    active_pedagogical_alerts: List[str] = Field(default_factory=list)

# API Requests & Responses
class FrameAnalyzeRequest(BaseModel):
    session_id: Optional[str] = "default_session"
    frame_id: int = 1
    timestamp: Optional[float] = None
    image_base64: Optional[str] = None
    # Support simulated/injected inputs for microservices sending pre-cropped/raw vectors
    simulated_students_count: Optional[int] = None

class SessionStartRequest(BaseModel):
    session_id: str
    classroom_name: str = "Lecture Hall A"
    course_name: Optional[str] = "Computer Science 101"
    sliding_window_seconds: float = 10.0
    webhook_url: Optional[str] = None

class SessionSummaryResponse(BaseModel):
    session_id: str
    classroom_name: str
    duration_seconds: float
    total_frames_analyzed: int
    total_unique_students: int
    average_classroom_engagement: float
    highest_engagement_timestamp: Optional[float] = None
    lowest_engagement_timestamp: Optional[float] = None
    confusion_spikes_count: int
    total_hand_raises: int
    pedagogical_insights: List[str]
