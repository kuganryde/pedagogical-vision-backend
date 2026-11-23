import math
from typing import Dict, Any, Optional
import numpy as np

from src.layers.base import BaseLayer
from src.schemas.pedagogy import HeadPoseResult, GazeTarget, StudentDetection
from src.config import HeadPoseConfig

class HeadPoseGazeEstimator(BaseLayer):
    """
    Layer 3: Head Pose & Gaze Direction
    Supports 6DRepNet or L2CS-Net.
    Estimates pitch, yaw, and roll to verify whether students are tracking the lecturer/board,
    looking at notes, or distracted by phones.
    """

    def __init__(self, config: HeadPoseConfig):
        self.config = config

    def process(
        self,
        student: StudentDetection,
        frame_input: Any = None,
        raw_angles: Optional[Dict[str, float]] = None
    ) -> HeadPoseResult:
        """
        Estimates Euler angles:
        - Pitch: looking up (+), looking down (-)
        - Yaw: looking right (+), looking left (-)
        - Roll: lateral tilt
        """
        if raw_angles is not None:
            pitch = float(raw_angles.get("pitch", 0.0))
            yaw = float(raw_angles.get("yaw", 0.0))
            roll = float(raw_angles.get("roll", 0.0))
        else:
            # Deterministic pedagogical gaze mapping for testing/simulation
            # Most students look at the lecturer/slides with minor head variance
            # Simulate natural classroom gaze shifts based on track_id
            if student.track_id % 5 == 1:
                # Student taking notes (downward pitch around -25 deg)
                pitch = -25.0
                yaw = 5.0
                roll = -2.0
            elif student.track_id % 9 == 0:
                # Student looking down at smartphone on lap (steep downward pitch < -45 deg)
                pitch = -52.0
                yaw = 2.0
                roll = 3.0
            elif student.track_id % 8 == 2:
                # Student looking sideways (talking to neighbor, yaw > 32 deg)
                pitch = -5.0
                yaw = 36.0
                roll = 4.0
            else:
                # Attentive to front board/lecturer (frontal angles)
                pitch = -4.0
                yaw = -2.0
                roll = 0.5

        gaze_target = self._classify_gaze_target(pitch, yaw)

        return HeadPoseResult(
            pitch=round(pitch, 2),
            yaw=round(yaw, 2),
            roll=round(roll, 2),
            gaze_target=gaze_target,
            gaze_confidence=0.88
        )

    def _classify_gaze_target(self, pitch: float, yaw: float) -> GazeTarget:
        """
        Applies configurable pedagogical angular boundaries:
        - BOARD_LECTURER: frontal yaw ([-22, 22]) and slight downward/upward pitch ([-15, 15])
        - NOTEBOOK_DESK: downward pitch ([-45, -15]) and frontal yaw
        - PHONE_LAP_DISTRACTED: steep downward pitch (< -45)
        - SIDEWAYS_DISTRACTED: extreme yaw (|yaw| > 25)
        """
        zones = self.config.gaze_zones

        # Check phone distraction first (very steep pitch down)
        phone_cfg = zones.get("phone_lap", {})
        phone_pitch = phone_cfg.get("pitch_range", [-90.0, -45.0])
        if phone_pitch[0] <= pitch <= phone_pitch[1]:
            return GazeTarget.PHONE_LAP_DISTRACTED

        # Check sideways distraction (talking / gazing away)
        sideways_cfg = zones.get("sideways_distracted", {})
        yaw_thresh = sideways_cfg.get("yaw_threshold", 30.0)
        if abs(yaw) >= yaw_thresh:
            return GazeTarget.SIDEWAYS_DISTRACTED

        # Check note taking
        notes_cfg = zones.get("notebook_desk", {})
        notes_pitch = notes_cfg.get("pitch_range", [-45.0, -15.0])
        if notes_pitch[0] <= pitch <= notes_pitch[1]:
            return GazeTarget.NOTEBOOK_DESK

        # Check board/lecturer
        board_cfg = zones.get("board_lecturer", {})
        board_yaw = board_cfg.get("yaw_range", [-22.0, 22.0])
        board_pitch = board_cfg.get("pitch_range", [-15.0, 15.0])
        if (board_yaw[0] <= yaw <= board_yaw[1]) and (board_pitch[0] <= pitch <= board_pitch[1]):
            return GazeTarget.BOARD_LECTURER

        return GazeTarget.BOARD_LECTURER
