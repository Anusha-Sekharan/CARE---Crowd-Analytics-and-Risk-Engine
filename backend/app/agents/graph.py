import os
import cv2
import logging
from langgraph.graph import StateGraph, START, END

from app.agents.state import CrowdState
from app.agents.detection_agent import CrowdDetectionAgent
from app.agents.hotspot_agent import HotspotDetectionAgent, HotspotInput
from app.agents.risk_agent import RiskAnalysisAgent, RiskInput
from app.agents.prediction_agent import PredictionAgent, PredictionInput
from app.agents.alert_agent import AlertAgent, AlertInput
from app.agents.advisor_agent import LLMSafetyAdvisorAgent, AdvisorInput

logger = logging.getLogger(__name__)

# Initialize singletons for agents
agent_detection = CrowdDetectionAgent()
agent_hotspot = HotspotDetectionAgent(grid_rows=3, grid_cols=3)
agent_risk = RiskAnalysisAgent()
agent_prediction = PredictionAgent()
agent_alert = AlertAgent()
agent_advisor = LLMSafetyAdvisorAgent()

def node_detection(state: CrowdState) -> CrowdState:
    logger.info("--- NODE: Detection ---")
    try:
        out = agent_detection.process_image(state["image_array"])
        state["people_count"] = out.people_count
        state["density_score"] = out.density_score
        state["bounding_boxes"] = out.bounding_boxes
    except Exception as e:
        state.setdefault("errors", []).append(f"Detection Error: {e}")
    return state

def node_hotspot(state: CrowdState) -> CrowdState:
    logger.info("--- NODE: Hotspot ---")
    if "errors" in state and state["errors"]:
        return state
        
    try:
        h, w = state["image_array"].shape[:2]
        hotspot_in = HotspotInput(
            people_count=state["people_count"],
            bounding_boxes=state["bounding_boxes"],
            image_width=w,
            image_height=h,
            density_map=getattr(agent_detection, 'last_density_map', None)
        )
        
        # Analytics
        out = agent_hotspot.analyze_zones(hotspot_in)
        state["zone_counts"] = out.zone_counts
        state["hotspot_zone"] = out.hotspot_zone
        state["max_zone_count"] = out.max_zone_count
        
        # Heatmap overlay
        heatmap_img = agent_hotspot.generate_heatmap_overlay(state["image_array"], hotspot_in)
        state["heatmap_array"] = heatmap_img
        
        # Save heatmap
        base_name = os.path.basename(state["image_path"])
        heatmap_path = os.path.join("data", "images", f"heatmap_{base_name}")
        cv2.imwrite(heatmap_path, heatmap_img)
        state["heatmap_image_path"] = heatmap_path
        
    except Exception as e:
        state.setdefault("errors", []).append(f"Hotspot Error: {e}")
    return state

def node_risk(state: CrowdState) -> CrowdState:
    logger.info("--- NODE: Risk ---")
    if "errors" in state and state["errors"]: return state
    try:
        r_in = RiskInput(
            people_count=state["people_count"],
            density_score=state["density_score"],
            max_zone_count=state["max_zone_count"]
        )
        out = agent_risk.analyze_risk(r_in)
        state["risk_level"] = out.risk_level.value
        state["congestion_score"] = out.congestion_score
        state["hotspot_concentration"] = out.hotspot_concentration
    except Exception as e:
        state.setdefault("errors", []).append(f"Risk Error: {e}")
    return state

def node_prediction(state: CrowdState) -> CrowdState:
    logger.info("--- NODE: Prediction ---")
    if "errors" in state and state["errors"]: return state
    try:
        p_in = PredictionInput(
            current_people_count=state["people_count"],
            congestion_score=state["congestion_score"],
            hotspot_concentration=state["hotspot_concentration"],
            time_horizon_mins=15
        )
        out = agent_prediction.predict_future_state(p_in)
        state["future_people_count"] = out.future_people_count
        state["stampede_risk"] = out.stampede_risk
    except Exception as e:
        state.setdefault("errors", []).append(f"Prediction Error: {e}")
    return state

def node_alert(state: CrowdState) -> CrowdState:
    logger.info("--- NODE: Alert ---")
    if "errors" in state and state["errors"]: return state
    try:
        a_in = AlertInput(
            risk_level=state["risk_level"],
            hotspot_zone=state["hotspot_zone"],
            stampede_risk=state["stampede_risk"]
        )
        out = agent_alert.generate_alert(a_in)
        state["alert_level"] = out.alert_level.value
        state["warning_message"] = out.warning_message
        state["requires_intervention"] = out.requires_intervention
    except Exception as e:
        state.setdefault("errors", []).append(f"Alert Error: {e}")
    return state

def node_advisor(state: CrowdState) -> CrowdState:
    logger.info("--- NODE: LLM Advisor ---")
    if "errors" in state and state["errors"]: return state
    try:
        adv_in = AdvisorInput(
            alert_level=state["alert_level"],
            warning_message=state["warning_message"],
            hotspot_zone=state["hotspot_zone"],
            people_count=state["people_count"],
            stampede_risk=state["stampede_risk"]
        )
        out = agent_advisor.generate_advice(adv_in)
        state["llm_risk_summary"] = out.risk_summary
        state["llm_safety_recommendations"] = out.safety_recommendations
        state["llm_incident_report"] = out.incident_report
    except Exception as e:
        logger.warning(f"Advisor error (likely Ollama missing): {e}")
        state["llm_risk_summary"] = "Ollama connection failed or LLM unavailable."
        state["llm_safety_recommendations"] = "- Manual protocol recommended."
        state["llm_incident_report"] = "System failure in advisor node."
    return state

# --- Build the Graph ---
workflow = StateGraph(CrowdState)

workflow.add_node("detection", node_detection)
workflow.add_node("hotspot", node_hotspot)
workflow.add_node("risk", node_risk)
workflow.add_node("prediction", node_prediction)
workflow.add_node("alert", node_alert)
workflow.add_node("advisor", node_advisor)

workflow.add_edge(START, "detection")
workflow.add_edge("detection", "hotspot")
workflow.add_edge("hotspot", "risk")
workflow.add_edge("risk", "prediction")
workflow.add_edge("prediction", "alert")
workflow.add_edge("alert", "advisor")
workflow.add_edge("advisor", END)

# Compile it into a runnable graph
app_graph = workflow.compile()
