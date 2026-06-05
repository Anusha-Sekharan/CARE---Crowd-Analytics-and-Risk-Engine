import logging
from enum import Enum
from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

# --- Schemas ---

class AlertLevel(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    ORANGE = "ORANGE"
    RED = "RED"

class AlertInput(BaseModel):
    risk_level: str = Field(..., description="Risk Level from Agent 3 (LOW, MODERATE, HIGH, CRITICAL)")
    hotspot_zone: str = Field(None, description="Hotspot zone from Agent 2")
    stampede_risk: int = Field(0, description="Stampede risk score from Agent 4")

class AlertOutput(BaseModel):
    alert_level: AlertLevel = Field(..., description="Mapped alert level (GREEN to RED)")
    warning_message: str = Field(..., description="Generated warning message")
    requires_intervention: bool = Field(..., description="Whether physical intervention is required")

# --- Agent Implementation ---

class AlertAgent:
    """
    Agent 5: Alert Agent
    Responsible for generating actionable alerts based on risk levels and 
    creating standardized warning messages for the dashboard.
    """
    
    def _map_risk_to_alert(self, risk_level: str, stampede_risk: int) -> AlertLevel:
        """
        Maps the Risk Level to standard color-coded Alert Levels.
        Escalates to RED if stampede risk is exceptionally high regardless of base risk.
        """
        if stampede_risk >= 85:
            return AlertLevel.RED
            
        mapping = {
            "LOW": AlertLevel.GREEN,
            "MODERATE": AlertLevel.YELLOW,
            "HIGH": AlertLevel.ORANGE,
            "CRITICAL": AlertLevel.RED
        }
        return mapping.get(risk_level.upper(), AlertLevel.YELLOW)

    def _generate_warning_message(self, alert_level: AlertLevel, hotspot: str) -> str:
        """
        Generates a concise, standardized warning message.
        """
        zone_text = f" in {hotspot}" if hotspot else ""
        
        if alert_level == AlertLevel.GREEN:
            return "Crowd conditions are safe. Normal monitoring."
        elif alert_level == AlertLevel.YELLOW:
            return f"Moderate crowding detected{zone_text}. Maintain standard vigilance."
        elif alert_level == AlertLevel.ORANGE:
            return f"High density detected{zone_text}. Prepare crowd diversion protocols."
        elif alert_level == AlertLevel.RED:
            return f"Critical density reached{zone_text}. Crowd diversion is recommended immediately."
        
        return "Unknown status."

    def generate_alert(self, data: AlertInput) -> AlertOutput:
        """
        Executes the alert generation logic.
        """
        alert_level = self._map_risk_to_alert(data.risk_level, data.stampede_risk)
        message = self._generate_warning_message(alert_level, data.hotspot_zone)
        intervention = alert_level in [AlertLevel.ORANGE, AlertLevel.RED]
        
        logger.info(f"Alert Generated - Level: {alert_level.value}, Intervention: {intervention}")
        
        return AlertOutput(
            alert_level=alert_level,
            warning_message=message,
            requires_intervention=intervention
        )

# --- Example Usage ---
if __name__ == "__main__":
    agent = AlertAgent()
    test_input = AlertInput(risk_level="CRITICAL", hotspot_zone="Zone C", stampede_risk=88)
    output = agent.generate_alert(test_input)
    print(output.model_dump_json(indent=2))
