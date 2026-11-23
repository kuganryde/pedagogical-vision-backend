import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class DetectionConfig(BaseModel):
    engine: str = "heuristic"
    model_name: str = "yolo11m"
    device: str = "cpu"
    confidence_threshold: float = 0.45
    iou_threshold: float = 0.50
    track_thresh: float = 0.25
    match_thresh: float = 0.80
    track_buffer: int = 30

class BodyPoseConfig(BaseModel):
    engine: str = "heuristic"
    model_name: str = "yolov8m-pose"
    keypoint_threshold: float = 0.50
    hand_raise_wrist_nose_margin: float = 15.0
    lean_forward_torso_deg: float = 18.0
    slump_desk_collapse_ratio: float = 0.55
    turning_away_angle_deg: float = 35.0

class GazeZoneConfig(BaseModel):
    yaw_range: Optional[List[float]] = None
    pitch_range: Optional[List[float]] = None
    yaw_threshold: Optional[float] = None

class HeadPoseConfig(BaseModel):
    engine: str = "heuristic"
    model_name: str = "6DRepNet"
    gaze_zones: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: {
        "board_lecturer": {"yaw_range": [-22.0, 22.0], "pitch_range": [-15.0, 15.0]},
        "notebook_desk": {"yaw_range": [-30.0, 30.0], "pitch_range": [-45.0, -15.0]},
        "phone_lap": {"yaw_range": [-35.0, 35.0], "pitch_range": [-90.0, -45.0]},
        "sideways_distracted": {"yaw_threshold": 30.0}
    })

class FacialAffectConfig(BaseModel):
    engine: str = "heuristic"
    model_name: str = "HSEmotion-B0"
    emotions: List[str] = [
        "concentration", "confusion", "boredom", "delight", "frustration", "neutral"
    ]
    action_units_tracked: List[str] = [
        "AU01_inner_brow_raiser", "AU02_outer_brow_raiser", "AU04_brow_lowerer",
        "AU12_lip_corner_puller", "AU15_lip_corner_depressor", "AU43_eyes_closed"
    ]

class TemporalFusionWeights(BaseModel):
    attention_gaze: float = 0.40
    facial_affect: float = 0.35
    body_posture: float = 0.25

class TemporalFusionAlerts(BaseModel):
    confusion_alert_rate: float = 0.30
    prolonged_distraction_seconds: float = 8.0
    sleeping_desk_seconds: float = 10.0

class TemporalFusionConfig(BaseModel):
    engine: str = "heuristic"
    sliding_window_seconds: float = 10.0
    sample_rate_fps: float = 5.0
    weights: TemporalFusionWeights = Field(default_factory=TemporalFusionWeights)
    alert_thresholds: TemporalFusionAlerts = Field(default_factory=TemporalFusionAlerts)

class LayersConfig(BaseModel):
    detection_and_tracking: DetectionConfig = Field(default_factory=DetectionConfig)
    body_posture: BodyPoseConfig = Field(default_factory=BodyPoseConfig)
    head_pose: HeadPoseConfig = Field(default_factory=HeadPoseConfig)
    facial_affect: FacialAffectConfig = Field(default_factory=FacialAffectConfig)
    temporal_fusion: TemporalFusionConfig = Field(default_factory=TemporalFusionConfig)

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8088
    workers: int = 1
    debug: bool = False
    cors_origins: List[str] = ["*"]
    api_prefix: str = "/api/v1"

class SessionConfig(BaseModel):
    max_active_sessions: int = 100
    session_timeout_minutes: int = 120
    export_analytics_on_close: bool = True

class EventsConfig(BaseModel):
    webhooks_enabled: bool = True
    webhook_endpoints: List[str] = []
    publish_events: List[str] = [
        "STUDENT_HAND_RAISE",
        "LECTURE_CONFUSION_SPIKE",
        "PERSISTENT_DISENGAGEMENT",
        "CLASS_ENGAGEMENT_UPDATE"
    ]

class AppSettings(BaseSettings):
    server: ServerConfig = Field(default_factory=ServerConfig)
    layers: LayersConfig = Field(default_factory=LayersConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    events: EventsConfig = Field(default_factory=EventsConfig)

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore"
    )

def load_settings(config_path: Optional[str] = None) -> AppSettings:
    """Load configuration from YAML file and merge with environment variables."""
    default_path = Path(__file__).parent.parent / "config" / "config.yaml"
    cfg_file = Path(config_path) if config_path else default_path

    yaml_data: Dict[str, Any] = {}
    if cfg_file.exists():
        with open(cfg_file, "r") as f:
            loaded = yaml.safe_load(f)
            if loaded and isinstance(loaded, dict):
                yaml_data = loaded

    return AppSettings(**yaml_data)

settings = load_settings()
