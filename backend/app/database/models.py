from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.sql import func
from app.database.database import Base
import uuid

def generate_uuid():
    return str(uuid.uuid4())

class AnalysisRecord(Base):
    __tablename__ = "analyses"

    id = Column(String, primary_key=True, index=True, default=generate_uuid)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Image Paths
    original_image_path = Column(String, nullable=False)
    heatmap_image_path = Column(String, nullable=True)
    
    # Telemetry
    people_count = Column(Integer, default=0)
    density_score = Column(Float, default=0.0)
    hotspot_zone = Column(String, nullable=True)
    max_zone_count = Column(Integer, default=0)
    
    # Risk & Alerts
    risk_level = Column(String, nullable=True)
    congestion_score = Column(Integer, default=0)
    future_people_count = Column(Integer, default=0)
    stampede_risk = Column(Integer, default=0)
    alert_level = Column(String, nullable=True)
    warning_message = Column(String, nullable=True)
    requires_intervention = Column(Boolean, default=False)
    
    # LLM Advice
    llm_risk_summary = Column(Text, nullable=True)
    llm_safety_recommendations = Column(Text, nullable=True)
    llm_incident_report = Column(Text, nullable=True)
