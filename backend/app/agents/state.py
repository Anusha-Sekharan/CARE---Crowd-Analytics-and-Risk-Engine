from typing import TypedDict, List, Dict, Any, Optional
import numpy as np

class CrowdState(TypedDict):
    # Core inputs
    image_path: str
    image_array: np.ndarray
    
    # Agent 1: Detection Output
    people_count: int
    density_score: float
    bounding_boxes: List[Any]
    
    # Agent 2: Hotspot Output
    zone_counts: Dict[str, int]
    hotspot_zone: str
    max_zone_count: int
    heatmap_array: np.ndarray
    heatmap_image_path: str
    
    # Agent 3: Risk Output
    risk_level: str
    congestion_score: int
    hotspot_concentration: float
    
    # Agent 4: Prediction Output
    future_people_count: int
    stampede_risk: int
    
    # Agent 5: Alert Output
    alert_level: str
    warning_message: str
    requires_intervention: bool
    
    # Agent 6: Advisor Output
    llm_risk_summary: str
    llm_safety_recommendations: str
    llm_incident_report: str
    
    # Output Control
    errors: List[str]
