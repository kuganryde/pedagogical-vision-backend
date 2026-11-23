import pytest
from src.config import settings
from src.pipeline.orchestrator import ModularPedagogicalPipeline
from src.session.manager import SessionManager

def test_full_pipeline_orchestration():
    pipeline = ModularPedagogicalPipeline(settings.layers)
    
    # Process simulated frame
    result = pipeline.process_frame(frame_id=1)
    
    assert result.frame_id == 1
    assert result.total_students_detected > 0
    assert 0.0 <= result.mean_engagement_score <= 1.0
    assert 0.0 <= result.confusion_percentage <= 100.0
    assert 0.0 <= result.distraction_percentage <= 100.0
    assert len(result.students) == result.total_students_detected

def test_session_manager_lifecycle():
    manager = SessionManager(settings.layers)
    session = manager.get_or_create("sess_test_100", classroom_name="Auditorium C")
    
    # Send 3 frames
    for i in range(1, 4):
        session.process_frame(frame_id=i)
        
    summary = manager.close_session("sess_test_100")
    assert summary is not None
    assert summary.session_id == "sess_test_100"
    assert summary.total_frames_analyzed == 3
    assert summary.average_classroom_engagement >= 0.0
